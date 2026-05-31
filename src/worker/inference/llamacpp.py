"""llama.cpp ``llama-server`` subprocess + OpenAI-compatible streaming (primary engine)."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
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
_METRICS_CANDIDATES_Q: tuple[str, ...] = (
    "llamacpp:requests_processing",
    "llamacpp_requests_processing",
)
# Requests accepted but waiting for a free slot. Added to ``requests_processing``
# so qw reports total in-flight (running + waiting), matching the MLX engine and
# letting the scheduler compare backlog across engines.
_METRICS_CANDIDATES_DEFERRED: tuple[str, ...] = (
    "llamacpp:requests_deferred",
    "llamacpp_requests_deferred",
)

# Timeout for the live engine client (shared by the spawn and adopt paths so they
# can never diverge). It must be generous: streaming generation can hold a slow
# CPU prefill for many seconds before the first token, and a CPU-bound node can
# take seconds to answer /metrics while every core is busy decoding.
_ENGINE_HTTP_TIMEOUT = httpx.Timeout(600.0)
# Short timeout for the one-shot "is a server already running?" adoption probe.
_ADOPT_PROBE_TIMEOUT = httpx.Timeout(2.0)


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
        n_ctx: int | None = None,  # Total context window (divided across slots)
        n_threads: int | None = None,  # CPU threads
        n_parallel: int | None = None,  # Concurrent inference slots
        n_threads_http: int | None = None,  # HTTP server worker threads
        cont_batching: bool | None = None,  # Continuous batching across slots
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
        self._n_ctx = n_ctx if n_ctx is not None else int(os.getenv("LLAMA_N_CTX", "16384"))
        self._n_threads = n_threads if n_threads is not None else max(1, cpu_count // 2)
        self._n_parallel = (
            n_parallel if n_parallel is not None else int(os.getenv("LLAMA_N_PARALLEL", "4"))
        )
        self._n_threads_http = (
            n_threads_http
            if n_threads_http is not None
            else int(os.getenv("LLAMA_N_THREADS_HTTP", "8"))
        )
        self._cont_batching = (
            cont_batching
            if cont_batching is not None
            else os.getenv("LLAMA_CONT_BATCHING", "1") not in ("0", "false", "False")
        )
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
            if self._proc and self._proc.returncode is not None:
                await self._stop_locked()
            if not self._model_path:
                raise RuntimeError(
                    "LLAMA_MODEL_PATH or model_path is required to start llama-server"
                )
            if not shutil.which(self._binary):
                raise RuntimeError(f"llama-server binary not found in PATH: {self._binary}")

            if await _port_reachable(self._host, self._port):
                adopted = await self._try_adopt()
                if adopted:
                    return
                # Port is occupied but unresponsive — kill whatever is there.
                killed = await asyncio.get_event_loop().run_in_executor(
                    None, _kill_pids_on_port, self._port
                )
                if killed:
                    logger.info(
                        "Killed stale process(es) on port {} before starting | pids={}",
                        self._port,
                        killed,
                    )
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if not await _port_reachable(self._host, self._port):
                            break
                        await asyncio.sleep(0.2)
                else:
                    raise RuntimeError(
                        f"Port {self._port} is already in use and could not be cleared. "
                        "Free the port manually and retry."
                    )

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
                "--parallel",
                str(self._n_parallel),
                "--threads-http",
                str(self._n_threads_http),
            ]
            if self._cont_batching:
                args.append("--cont-batching")
            args.extend(self._extra_args)

            logger.info("Starting llama-server | cmd={}", " ".join(args))
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=_ENGINE_HTTP_TIMEOUT)

            deadline = time.monotonic() + float(os.getenv("LLAMA_SERVER_START_TIMEOUT", "120"))
            while time.monotonic() < deadline:
                if self._proc.returncode is not None:
                    code = self._proc.returncode
                    err = await self._drain_stderr_tail()
                    await self._stop_locked()
                    raise RuntimeError(
                        f"llama-server exited during startup code={code}. stderr_tail={err!r}"
                    )
                health = await self.health()
                if health.status:
                    logger.info("llama-server ready | url={}", self.base_url)
                    return
                await asyncio.sleep(0.3)

            err = await self._drain_stderr_tail()
            await self._stop_locked()
            raise RuntimeError(f"llama-server did not become healthy in time. stderr_tail={err!r}")

    async def stop(self) -> None:
        """Terminate the subprocess and close the HTTP client."""
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stop backend resources while ``self._lock`` is already held."""
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

    async def _try_adopt(self) -> bool:
        """Adopt an already-running llama-server on our port without spawning.

        Probes with a short timeout, but the adopted live client uses the same
        generous timeout as the spawn path — a 2s timeout on the live client
        would silently zero ``/metrics`` (queue depth) and break slow prefills on
        a CPU-bound node. Returns True and sets ``self._http`` when healthy.
        """
        probe = httpx.AsyncClient(base_url=self.base_url, timeout=_ADOPT_PROBE_TIMEOUT)
        try:
            response = await probe.get("/health")
            adopted = response.status_code == 200
        except httpx.RequestError:
            adopted = False
        finally:
            await probe.aclose()
        if not adopted:
            return False
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=_ENGINE_HTTP_TIMEOUT)
        logger.info(
            "Adopted existing llama-server on port {} | url={}",
            self._port,
            self.base_url,
        )
        return True

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
            return EngineReportedTelemetry(qw=0)

        try:
            response = await client.get("/metrics")
        except httpx.RequestError as e:
            logger.warning(
                "llama-server /metrics unreachable; reporting qw=0 | url={} error={}",
                self.base_url,
                e,
            )
            return EngineReportedTelemetry(qw=0)

        if response.status_code != 200:
            logger.warning(
                "llama-server /metrics returned HTTP {}; reporting qw=0 | url={}",
                response.status_code,
                self.base_url,
            )
            return EngineReportedTelemetry(qw=0)

        samples = parse_prometheus_samples(response.text)
        processing = find_metric(samples, *_METRICS_CANDIDATES_Q)
        deferred = find_metric(samples, *_METRICS_CANDIDATES_DEFERRED)
        qw = int(processing or 0) + int(deferred or 0)
        return EngineReportedTelemetry(qw=qw)

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


async def _port_reachable(host: str, port: int) -> bool:
    """Return True if something is already listening on host:port."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=0.5,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def _kill_pids_on_port(port: int) -> list[int]:
    """Return PIDs found listening on port and send them SIGTERM.

    Tries ``fuser`` first (common on Linux), then ``lsof`` (macOS / Linux).
    Returns the list of PIDs that were signalled.
    """
    pids: list[int] = []

    try:
        r = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        pids = [int(p) for p in r.stdout.split() if p.strip().lstrip("-").isdigit()]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    if not pids:
        try:
            r = subprocess.run(
                ["lsof", "-ti", f"TCP:{port}", "-s", "TCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            pids = [int(p) for p in r.stdout.splitlines() if p.strip().isdigit()]
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    return pids
