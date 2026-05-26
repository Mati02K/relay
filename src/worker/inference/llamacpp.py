"""llama.cpp ``llama-server`` subprocess + OpenAI-compatible streaming (primary engine)."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import time
from collections.abc import AsyncIterator, Mapping
from typing import Final

import httpx
from loguru import logger

from worker.inference.base import (
    EngineHealth,
    InferenceEngine,
    Telemetry,
    ema_update,
    estimate_tokens_from_text,
    extract_usage_token_counts,
    find_metric,
    parse_prometheus_samples,
    prefix_chunk_hashes_from_text,
    prompt_length_bucket_id,
)

_DEFAULT_PORT: Final[int] = 9081
_METRICS_CANDIDATES_KV: Final[tuple[str, ...]] = (
    "llamacpp:kv_cache_usage_ratio",
    "llamacpp_kv_cache_usage_ratio",
)
_METRICS_CANDIDATES_Q: Final[tuple[str, ...]] = (
    "llamacpp:requests_processing",
    "llamacpp_requests_processing",
)
_THERMAL_CACHE_TTL_S: Final[float] = float(os.getenv("RELAY_THERMAL_TTL_S", "2.0"))
_NVIDIA_THROTTLE_THRESHOLD: Final[float] = 0.9

# Single-host heterogeneity knobs used by the ablation study.
# These override or perturb specific telemetry fields so one MacBook can stand in
# for three "different" workers without lying about the underlying engine.
_FAKE_THETA_W: Final[str] = os.getenv("RELAY_FAKE_THETA_W", "")
_FAKE_QW_OFFSET: Final[int] = int(os.getenv("RELAY_FAKE_QW_OFFSET", "0"))
_FAKE_MW: Final[str] = os.getenv("RELAY_FAKE_MW", "")
_FAKE_TELEMETRY_DELAY_MS: Final[float] = float(os.getenv("RELAY_FAKE_TELEMETRY_DELAY_MS", "0"))
# When >0, adds a uniform random sleep in [0, ms] before returning telemetry.
# This injects real variance into coordinator-measured RTT and produces a
# non-trivial j_w EMA so the delta term in the cost function is exercisable
# on a single-host deployment.
_FAKE_TELEMETRY_RANDOM_JITTER_MS: Final[float] = float(
    os.getenv("RELAY_FAKE_TELEMETRY_RANDOM_JITTER_MS", "0")
)


async def _read_thermal_flag() -> int:
    """Return 1 if this host's CPU/GPU appears thermally throttled, else 0.

    Best-effort across platforms; failures silently return 0 (assume healthy).
    """
    system = platform.system()
    try:
        if system == "Darwin":
            return await _read_thermal_darwin()
        if system == "Linux":
            return await _read_thermal_linux()
    except Exception as exc:
        logger.debug("Thermal probe failed | system={} err={}", system, exc)
    return 0


async def _read_thermal_darwin() -> int:
    """macOS: ``pmset -g therm`` exposes CPU_Speed_Limit < 100 when throttled."""
    if not shutil.which("pmset"):
        return 0
    proc = await asyncio.create_subprocess_exec(
        "pmset", "-g", "therm",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
    except TimeoutError:
        proc.kill()
        return 0
    for raw_line in stdout.decode(errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("CPU_Speed_Limit"):
            value = line.split("=", 1)[-1].strip()
            try:
                return 1 if int(value) < 100 else 0
            except ValueError:
                return 0
    return 0


async def _read_thermal_linux() -> int:
    """Linux + NVIDIA: nvidia-smi shows current vs max graphics clock when throttled."""
    if not shutil.which("nvidia-smi"):
        return 0
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=clocks.current.graphics,clocks.max.graphics",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
    except TimeoutError:
        proc.kill()
        return 0
    for raw_line in stdout.decode(errors="replace").splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) != 2:
            continue
        try:
            current = float(parts[0])
            maximum = float(parts[1])
        except ValueError:
            continue
        if maximum > 0 and current / maximum < _NVIDIA_THROTTLE_THRESHOLD:
            return 1
    return 0


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
        n_gpu_layers: int = -1,           # All to GPU
        n_ctx: int = 4096,                # Context window
        n_threads: int | None = None,     # CPU threads
        verbose: bool = False,            # Less spam
    ) -> None:
        self._model_path = model_path or os.getenv("LLAMA_MODEL_PATH", "")
        self._host = host or os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self._port = int(port or os.getenv("LLAMA_SERVER_PORT", str(_DEFAULT_PORT)))
        self._binary = server_binary or os.getenv("LLAMA_SERVER_BIN", "llama-server")
        self._extra_args = extra_args or _split_args(os.getenv("LLAMA_SERVER_EXTRA_ARGS", ""))

        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_threads = n_threads if n_threads is not None else max(1, os.cpu_count() // 2)
        self._verbose = verbose

        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._http: httpx.AsyncClient | None = None

        self._sw_by_bucket: dict[str, float] = {}
        self._sprefill_ema: float = 0.0
        self._prefix_hw: set[str] = set()
        self._max_hw_chunks: int = int(os.getenv("RELAY_KV_HASH_CACHE_CHUNKS", "512"))

        self._thermal_value: int = 0
        self._thermal_last_check: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def start(self) -> None:
        """Spawn ``llama-server`` if not already running (idempotent)."""
        async with self._lock:
            if self._proc and self._proc.returncode is None:
                return
            if not self._model_path:
                raise RuntimeError("LLAMA_MODEL_PATH or model_path is required to start llama-server")
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
                "-ngl", str(self._n_gpu_layers),
                "-c", str(self._n_ctx),
                "-t", str(self._n_threads or os.cpu_count()),
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
                if health.ok:
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
                ok=False,
                detail=f"llama-server exited code={self._proc.returncode} stderr={tail[:500]}",
                engine="llama.cpp",
            )
        client = self._http
        if not client:
            return EngineHealth(ok=False, detail="llama-server not started", engine="llama.cpp")
        try:
            r = await client.get("/health")
            if r.status_code == 200:
                return EngineHealth(ok=True, detail="ok", engine="llama.cpp")
            return EngineHealth(
                ok=False,
                detail=f"HTTP {r.status_code}: {r.text[:200]}",
                engine="llama.cpp",
            )
        except httpx.RequestError as e:
            return EngineHealth(ok=False, detail=str(e), engine="llama.cpp")

    async def get_telemetry(self) -> Telemetry:
        qw = 0
        mw = 0.0
        client = self._http
        if client:
            try:
                r = await client.get("/metrics")
                if r.status_code == 200:
                    samples = parse_prometheus_samples(r.text)
                    qv = find_metric(samples, *_METRICS_CANDIDATES_Q)
                    if qv is not None:
                        qw = int(qv)
                    kv = find_metric(samples, *_METRICS_CANDIDATES_KV)
                    if kv is not None:
                        mw = max(0.0, min(1.0, float(kv)))
            except httpx.RequestError:
                pass

        theta_w = await self._cached_thermal_flag()

        # Ablation knobs: override or perturb fields so co-located workers can
        # play different roles in the experiment. Defaults are no-ops.
        if _FAKE_THETA_W != "":
            try:
                theta_w = int(_FAKE_THETA_W)
            except ValueError:
                pass
        if _FAKE_QW_OFFSET:
            qw = max(0, qw + _FAKE_QW_OFFSET)
        if _FAKE_MW != "":
            try:
                mw = max(0.0, min(1.0, float(_FAKE_MW)))
            except ValueError:
                pass
        if _FAKE_TELEMETRY_DELAY_MS > 0:
            await asyncio.sleep(_FAKE_TELEMETRY_DELAY_MS / 1000.0)
        if _FAKE_TELEMETRY_RANDOM_JITTER_MS > 0:
            import random
            jitter_s = random.uniform(0, _FAKE_TELEMETRY_RANDOM_JITTER_MS) / 1000.0
            await asyncio.sleep(jitter_s)

        return Telemetry(
            qw=qw,
            sw_by_bucket=dict(self._sw_by_bucket),
            mw=mw,
            jw=0.0,
            theta_w=theta_w,
            prefix_chunk_hashes=sorted(self._prefix_hw),
            sprefill_tokens_per_sec=self._sprefill_ema,
        )

    async def _cached_thermal_flag(self) -> int:
        """Probe thermal state at most once every ``_THERMAL_CACHE_TTL_S`` seconds."""
        now = time.monotonic()
        if now - self._thermal_last_check >= _THERMAL_CACHE_TTL_S:
            self._thermal_value = await _read_thermal_flag()
            self._thermal_last_check = now
        return self._thermal_value

    async def generate(self, request: Mapping[str, object]) -> AsyncIterator[str]:
        await self.start()
        assert self._http is not None

        body = dict(request)
        body.setdefault("model", "gpt-3.5-turbo")
        body["stream"] = True
        # Force llama-server to emit a final usage chunk so we can compute decode tok/s.
        existing_opts = body.get("stream_options")
        if isinstance(existing_opts, dict):
            existing_opts.setdefault("include_usage", True)
        else:
            body["stream_options"] = {"include_usage": True}

        messages = body.get("messages")
        prompt_text = _messages_to_text(messages) if isinstance(messages, list) else ""
        est_prompt_tokens = estimate_tokens_from_text(prompt_text)
        bucket = prompt_length_bucket_id(est_prompt_tokens)

        t0 = time.perf_counter()
        first_token_t: float | None = None
        last_usage_prompt: int | None = None
        last_usage_completion: int | None = None

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
                if first_token_t is None and line.startswith("data: ") and line != "data: [DONE]":
                    payload = line.removeprefix("data: ").strip()
                    if payload and payload != "[DONE]":
                        try:
                            obj = json.loads(payload)
                            choices = obj.get("choices")
                            if isinstance(choices, list) and choices:
                                delta = choices[0].get("delta") or {}
                                if isinstance(delta, dict) and (
                                    delta.get("content") or delta.get("reasoning_content")
                                ):
                                    first_token_t = time.perf_counter()
                        except json.JSONDecodeError:
                            pass
                if line.startswith("data: "):
                    payload = line.removeprefix("data: ").strip()
                    if payload and payload != "[DONE]":
                        pt, ct = extract_usage_token_counts(payload)
                        if pt is not None:
                            last_usage_prompt = pt
                        if ct is not None:
                            last_usage_completion = ct

        elapsed = time.perf_counter() - t0
        prompt_tokens = last_usage_prompt if last_usage_prompt is not None else est_prompt_tokens
        completion_tokens = last_usage_completion if last_usage_completion is not None else 0

        if first_token_t is not None and prompt_tokens > 0:
            prefill_s = max(first_token_t - t0, 1e-6)
            sprefill = prompt_tokens / prefill_s
            self._sprefill_ema = (
                ema_update(self._sprefill_ema, sprefill) if self._sprefill_ema > 0 else sprefill
            )
        if first_token_t is not None:
            decode_s = max(elapsed - (first_token_t - t0), 1e-6)
        else:
            decode_s = max(elapsed, 1e-6)
        if completion_tokens > 0:
            decode_tok_s = completion_tokens / decode_s
            prev = self._sw_by_bucket.get(bucket, decode_tok_s)
            self._sw_by_bucket[bucket] = ema_update(prev, decode_tok_s)

        for h in prefix_chunk_hashes_from_text(prompt_text):
            if len(self._prefix_hw) >= self._max_hw_chunks:
                break
            self._prefix_hw.add(h)


def _messages_to_text(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text")
                        if isinstance(t, str):
                            parts.append(t)
    return "\n".join(parts)

def _split_args(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    return raw.split()

