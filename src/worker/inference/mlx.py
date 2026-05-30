"""MLX LM server inference engine for Apple Silicon."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping

import httpx
from loguru import logger

from telemetry.schemas import EngineReportedTelemetry
from worker.inference.base import EngineHealth, InferenceEngine

_DEFAULT_PORT = 9081


class MlxEngine(InferenceEngine):
    """Runs mlx_lm.server as a subprocess and exposes generate / health / telemetry.

    The model argument should be a HuggingFace repo ID such as
    ``mlx-community/Qwen2.5-7B-Instruct-4bit`` or a local directory containing
    MLX-format weights. When given a repo ID, the engine pre-downloads the
    model into the HuggingFace cache before spawning the server so the process
    does not crash mid-startup from a failed download.

    Queue depth (``qw``) is tracked by counting active streaming requests
    because mlx_lm.server does not expose Prometheus metrics.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        port: int | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self._model = (
            model or os.environ.get("LLAMA_MODEL_PATH") or os.environ.get("RELAY_MODEL_ID", "")
        )
        self._host = host or os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
        self._port = port or int(os.getenv("LLAMA_SERVER_PORT", str(_DEFAULT_PORT)))
        self._extra_args = extra_args or _split_args(os.getenv("LLAMA_SERVER_EXTRA_ARGS", ""))
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._http: httpx.AsyncClient | None = None
        self._active_requests: int = 0

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def uses_gpu(self) -> bool:
        return True

    async def start(self) -> None:
        """Spawn mlx_lm.server if not already running (idempotent).

        The model must already be present locally — download it with
        ``relay init`` or ``relay pull`` before starting the cluster.
        """
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                return
            if self._proc and self._proc.returncode is not None:
                await self._stop_locked()
            if not self._model:
                raise RuntimeError(
                    "LLAMA_MODEL_PATH or RELAY_MODEL_ID is required to start mlx_lm.server"
                )

            # Kill any stale process that holds our port (e.g. a leftover
            # llama-server from a previous session), so our health check does
            # not get a false-positive 200 from the wrong server.
            if await _port_reachable(self._host, self._port):
                killed = await asyncio.get_event_loop().run_in_executor(
                    None, _kill_pids_on_port, self._port
                )
                if killed:
                    logger.info(
                        "Killed stale process(es) on port {} before mlx_lm.server | pids={}",
                        self._port,
                        killed,
                    )
                    deadline_clear = time.monotonic() + 5.0
                    while time.monotonic() < deadline_clear:
                        if not await _port_reachable(self._host, self._port):
                            break
                        await asyncio.sleep(0.2)
                else:
                    raise RuntimeError(
                        f"Port {self._port} is already in use and could not be cleared. "
                        "Free the port manually and retry."
                    )

            args = [
                sys.executable,
                "-m",
                "mlx_lm.server",
                "--model",
                self._model,
                "--host",
                self._host,
                "--port",
                str(self._port),
            ]
            args.extend(self._extra_args)

            logger.info("Starting mlx_lm.server | cmd={}", " ".join(args))
            # Merge stderr into stdout so all output is captured in one stream.
            # mlx_lm prints startup errors and download progress to stdout.
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(600.0))

            deadline = time.monotonic() + float(os.getenv("LLAMA_SERVER_START_TIMEOUT", "300"))
            while time.monotonic() < deadline:
                if self._proc.returncode is not None:
                    code = self._proc.returncode
                    err = await self._drain_output()
                    await self._stop_locked()
                    raise RuntimeError(
                        f"mlx_lm.server exited during startup code={code}. output={err!r}"
                    )
                health = await self.health()
                if health.status:
                    logger.info("mlx_lm.server ready | url={}", self.base_url)
                    return
                await asyncio.sleep(0.5)

            err = await self._drain_output()
            await self._stop_locked()
            raise RuntimeError(
                f"mlx_lm.server did not become healthy in time. output={err!r}"
            )

    async def stop(self) -> None:
        """Terminate the subprocess and close the HTTP client."""
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
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

    async def _drain_output(self, max_bytes: int = 4000) -> str:
        """Read captured output from the merged stdout+stderr stream."""
        if not self._proc or not self._proc.stdout:
            return ""
        data = await self._proc.stdout.read(max_bytes)
        return data.decode(errors="replace")

    async def health(self) -> EngineHealth:
        if self._proc and self._proc.returncode is not None:
            out = await self._drain_output()
            return EngineHealth(
                status=False,
                detail=f"mlx_lm.server exited code={self._proc.returncode} output={out[:500]}",
                engine="mlx",
            )
        client = self._http
        if not client:
            return EngineHealth(status=False, detail="mlx_lm.server not started", engine="mlx")
        try:
            r = await client.get("/v1/models")
            if r.status_code == 200:
                return EngineHealth(status=True, detail="ok", engine="mlx")
            return EngineHealth(
                status=False,
                detail=f"HTTP {r.status_code}: {r.text[:200]}",
                engine="mlx",
            )
        except httpx.RequestError as e:
            return EngineHealth(status=False, detail=str(e), engine="mlx")

    async def get_engine_telemetry(self) -> EngineReportedTelemetry:
        """Return active request count as queue depth; mw is not available from MLX."""
        return EngineReportedTelemetry(qw=self._active_requests, mw=0.0)

    async def generate(self, request: Mapping[str, object]) -> AsyncIterator[str]:
        await self.start()
        assert self._http is not None

        body = dict(request)
        body["stream"] = True
        body["model"] = self._model

        self._active_requests += 1
        try:
            async with self._http.stream(
                "POST",
                "/v1/chat/completions",
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode(errors="replace")
                    raise RuntimeError(f"mlx_lm.server error {resp.status_code}: {err_text[:500]}")
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    yield line + "\n"
        finally:
            self._active_requests -= 1


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
    """Signal SIGTERM to any process listening on the given TCP port.

    Tries lsof (macOS/Linux) and returns the list of PIDs signalled.
    """
    pids: list[int] = []
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
