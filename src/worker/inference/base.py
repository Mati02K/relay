"""Inference engine abstraction and telemetry shapes for Relay workers (data plane)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, Field

# Paper §6.2: fixed 16-token chunks; without a tokenizer we approximate chunking (see below).
PREFIX_CHUNK_TOKENS: Final[int] = 16
# Rough bytes-per-token for UTF-8 English-ish text when tokenizing is unavailable.
_BYTES_PER_TOKEN_EST: Final[float] = 4.0


class EngineHealth(BaseModel):
    """Whether the backend process and HTTP endpoint are usable."""

    model_config = {"frozen": True}

    ok: bool
    detail: str = ""
    engine: str = "unknown"


class Telemetry(BaseModel):
    """Worker → coordinator telemetry (paper §6.1, §6.7).

    Field names follow the paper; ``sw_by_bucket`` is s_w(b) keyed by prompt-length bucket id.
    """

    model_config = {"frozen": False}

    qw: int = Field(0, description="Queue depth (tokens waiting, engine-provided or 0)")
    sw_by_bucket: dict[str, float] = Field(
        default_factory=dict,
        description="EMA decode tokens/s by prompt-length bucket b",
    )
    mw: float = Field(0.0, ge=0.0, le=1.0, description="Memory / KV-cache pressure in [0, 1]")
    jw: float = Field(0.0, ge=0.0, description="RTT jitter EMA (ms); shim fills from heartbeats")
    theta_w: int = Field(0, ge=0, le=1, description="Thermal throttling flag")
    prefix_chunk_hashes: list[str] = Field(
        default_factory=list,
        description="H_w: prefix chunk hashes resident in KV (approximation via shim)",
    )
    sprefill_tokens_per_sec: float = Field(
        0.0,
        ge=0.0,
        description="s_prefill^w from last completed request (tok/s)",
    )


def prompt_length_bucket_id(prompt_token_estimate: int) -> str:
    """Discrete bucket b for s_w(b) keyed by prompt length (tokens, approximate)."""
    t = max(0, prompt_token_estimate)
    if t <= 256:
        return "<=256"
    if t <= 1024:
        return "<=1024"
    if t <= 4096:
        return "<=4096"
    return ">4096"


def estimate_tokens_from_text(text: str) -> int:
    """Token count proxy when the engine has not yet reported usage."""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text.encode("utf-8")) / _BYTES_PER_TOKEN_EST)))


def prefix_chunk_hashes_from_text(text: str) -> list[str]:
    """16-token-chunk hashes (paper §6.2) using a byte window proxy for PREFIX_CHUNK_TOKENS."""
    if not text:
        return []
    data = text.encode("utf-8")
    step = max(1, int(PREFIX_CHUNK_TOKENS * _BYTES_PER_TOKEN_EST))
    out: list[str] = []
    for i in range(0, len(data), step):
        chunk = data[i : i + step]
        out.append(hashlib.sha256(chunk).hexdigest())
    return out


def prefix_overlap_fraction(prompt_hashes: Sequence[str], hw: set[str]) -> float:
    """Fraction of contiguous prefix chunks present in H_w (paper §6.2, Eq. 3 style)."""
    n = len(prompt_hashes)
    if n == 0:
        return 1.0
    first_miss = n
    for i, h in enumerate(prompt_hashes):
        if h not in hw:
            first_miss = i
            break
    else:
        return 1.0
    return first_miss / n


def ema_update(previous: float, observed: float, alpha: float = 0.15) -> float:
    """Exponential moving average for decode and prefill speed trackers."""
    return alpha * observed + (1.0 - alpha) * previous


def parse_prometheus_samples(body: str) -> dict[str, float]:
    """Parse a minimal subset of Prometheus text format into metric name → value."""
    result: dict[str, float] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric, value_s = parts
        try:
            value = float(value_s)
        except ValueError:
            continue
        # Strip labels: metric_name{...} -> metric_name
        base = metric.split("{", 1)[0]
        result[base] = value
        result[metric] = value
    return result


def find_metric(samples: Mapping[str, float], *candidates: str) -> float | None:
    """Return the first matching metric value from alternate llama.cpp naming styles."""
    for name in candidates:
        if name in samples:
            return float(samples[name])
    stripped = {k.split("{", 1)[0]: v for k, v in samples.items()}
    for name in candidates:
        if name in stripped:
            return float(stripped[name])
    tail_by_candidate = [c.split(":", 1)[-1].replace(":", "_") for c in candidates]
    for key, val in stripped.items():
        for tail in tail_by_candidate:
            if key == tail or key.endswith(tail):
                return float(val)
    return None


_TOKEN_RE = re.compile(r'"prompt_tokens"\s*:\s*(\d+)|"completion_tokens"\s*:\s*(\d+)')


def extract_usage_token_counts(json_line: str) -> tuple[int | None, int | None]:
    """Best-effort parse of prompt_tokens / completion_tokens from an SSE JSON fragment."""
    try:
        obj = json.loads(json_line)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        usage = obj.get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            p_ok = isinstance(pt, int)
            c_ok = isinstance(ct, int)
            if p_ok or c_ok:
                return (int(pt) if p_ok else None, int(ct) if c_ok else None)
    prompt: int | None = None
    completion: int | None = None
    for m in _TOKEN_RE.finditer(json_line):
        if m.group(1) is not None:
            prompt = int(m.group(1))
        if m.group(2) is not None:
            completion = int(m.group(2))
    return prompt, completion


class InferenceEngine(ABC):
    """Data-plane inference backend (paper §2.2)."""

    @abstractmethod
    async def generate(self, request: Mapping[str, object]) -> AsyncIterator[str]:
        """Stream one chat completion as newline-delimited SSE ``data:`` lines (OpenAI style)."""

    @abstractmethod
    async def health(self) -> EngineHealth:
        """Process + HTTP readiness for the active model server."""

    @abstractmethod
    async def get_telemetry(self) -> Telemetry:
        """Latest telemetry snapshot for the coordinator scheduler."""
