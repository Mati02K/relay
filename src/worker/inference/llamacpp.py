"""llama.cpp ``llama-server`` subprocess + OpenAI-compatible streaming (primary engine)."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import AsyncIterator, Mapping

import httpx
from loguru import logger

from telemetry.prometheus import find_metric, parse_prometheus_samples
from telemetry.schemas import EngineReportedTelemetry
from worker.inference.base import (
    EngineHealth,
    InferenceEngine,
)

_DEFAULT_PORT = 9081
_METRICS_CANDIDATES_KV: tuple[str, ...] = (
    "llamacpp:kv_cache_usage_ratio",
    "llamacpp_kv_cache_usage_ratio",
)
_METRICS_CANDIDATES_Q: tuple[str, ...] = (
    "llamacpp:requests_processing",
    "llamacpp_requests_processing",
)


class LlamaCppEngine(InferenceEngine):
    """Runs ``llama-server`` locally and exposes generate / health / telemetry."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        server_binary: str | None = None,
        extra_args: list[str] | None = None,
        n_gpu_layers: int = -1,  # All to GPU
        n_ctx: int = 4096,  # Context window
        n_threads: int | None = None,  # CPU threads
        verbose: bool = False,  # Less spam
    ) -> None:
        self._model_path: str = (
            model_path if model_path is not None else os.environ.get("LLAMA_MODEL_PATH", "")
        )
        self._host: str = (
            host if host is not None else os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
        )
        self._port = (
            port if port is not None else int(os.getenv("LLAMA_SERVER_PORT", str(_DEFAULT_PORT)))
        )
        self._binary: str = (
            server_binary
            if server_binary is not None
            else os.environ.get("LLAMA_SERVER_BIN", "llama-server")
        )
        self._extra_args = extra_args or _split_args(os.getenv("LLAMA_SERVER_EXTRA_ARGS", ""))

        cpu_count = os.cpu_count() or 1
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_threads = n_threads if n_threads is not None else max(1, cpu_count // 2)
        self._verbose = verbose

        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._http: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def uses_gpu(self) -> bool:
        """``n_gpu_layers != 0`` means at least one transformer layer runs on GPU."""
        return self._n_gpu_layers != 0

    async def start(self) -> None:
        """Spawn ``llama-server`` if not already running (idempotent)."""
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                return
            if not self._model_path:
                raise RuntimeError(
                    "LLAMA_MODEL_PATH or model_path is required to start llama-server"
                )
            if not shutil.which(self._binary):
                raise RuntimeError(f"llama-server binary not found in PATH: {self._binary}")

            args: list[str] = [
                self._binary,
                "-m",
                self._model_path,
                "--host",
                self._host,
                "--port",
                str(self._port),
                "--metrics",
                "-ngl",
                str(self._n_gpu_layers),
                "-c",
                str(self._n_ctx),
                "-t",
                str(self._n_threads or os.cpu_count()),
            ]
            args.extend(self._extra_args)

            logger.info("Starting llama-server | cmd={}", " ".join(args))
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(600.0))

            deadline = time.monotonic() + float(os.getenv("LLAMA_SERVER_START_TIMEOUT", "120"))
            while time.monotonic() < deadline:
                health = await self.health()
                if health.status:
                    logger.info("llama-server ready | url={}", self.base_url)
                    return
                await asyncio.sleep(0.3)

            err = await self._drain_stderr_tail()
            await self.stop()
            raise RuntimeError(f"llama-server did not become healthy in time. stderr_tail={err!r}")

    async def stop(self) -> None:
        """Terminate the subprocess and close the HTTP client."""
        async with self._lock:
            if self._http:
                await self._http.aclose()
                self._http = None
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=15.0)
                except TimeoutError:
                    self._proc.kill()
            self._proc = None

    async def _drain_stderr_tail(self, max_bytes: int = 4000) -> str:
        if not self._proc or not self._proc.stderr:
            return ""
        data = await self._proc.stderr.read(max_bytes)
        return data.decode(errors="replace")

    async def health(self) -> EngineHealth:
        if self._proc and self._proc.returncode is not None:
            tail = await self._drain_stderr_tail()
            return EngineHealth(
                status=False,
                detail=f"llama-server exited code={self._proc.returncode} stderr={tail[:500]}",
                engine="llama.cpp",
            )
        client = self._http
        if not client:
            return EngineHealth(status=False, detail="llama-server not started", engine="llama.cpp")
        try:
            r = await client.get("/health")
            if r.status_code == 200:
                return EngineHealth(status=True, detail="ok", engine="llama.cpp")
            return EngineHealth(
                status=False,
                detail=f"HTTP {r.status_code}: {r.text[:200]}",
                engine="llama.cpp",
            )
        except httpx.RequestError as e:
            return EngineHealth(status=False, detail=str(e), engine="llama.cpp")

    async def get_engine_telemetry(self) -> EngineReportedTelemetry:
        client = self._http
        if not client:
            return EngineReportedTelemetry(qw=0, mw=0.0)

        try:
            response = await client.get("/metrics")
        except httpx.RequestError:
            return EngineReportedTelemetry(qw=0, mw=0.0)

        if response.status_code != 200:
            return EngineReportedTelemetry(qw=0, mw=0.0)

        samples = parse_prometheus_samples(response.text)
        qv = find_metric(samples, *_METRICS_CANDIDATES_Q)
        kv = find_metric(samples, *_METRICS_CANDIDATES_KV)
        qw = int(qv) if qv is not None else 0
        mw = max(0.0, min(1.0, float(kv))) if kv is not None else 0.0
        return EngineReportedTelemetry(qw=qw, mw=mw)

    async def generate(self, request: Mapping[str, object]) -> AsyncIterator[str]:
        await self.start()
        assert self._http is not None

        body = dict(request)
        body.setdefault("model", "gpt-3.5-turbo")
        body["stream"] = True

        async with self._http.stream(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            if resp.status_code >= 400:
                err_text = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"llama-server error {resp.status_code}: {err_text[:500]}")

            async for line in resp.aiter_lines():
                if not line:
                    continue
                yield line + "\n"


def _split_args(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return raw.split()
