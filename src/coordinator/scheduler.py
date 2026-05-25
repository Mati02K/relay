"""Cache-aware worker scheduler."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import (
    compute_prefix_hashes_from_text,
    longest_prefix_match,
    prefix_hit_tokens,
    request_to_prefix_text,
)
from telemetry.request_metrics import prompt_length_bucket_id
from telemetry.schemas import Telemetry

DEFAULT_EXPECTED_OUTPUT_TOKENS = int(os.getenv("RELAY_DEFAULT_EXPECTED_OUTPUT_TOKENS", "128"))
DEFAULT_PREFILL_TOKENS_PER_SEC = float(os.getenv("RELAY_DEFAULT_PREFILL_TOKENS_PER_SEC", "100"))
DEFAULT_DECODE_TOKENS_PER_SEC = float(os.getenv("RELAY_DEFAULT_DECODE_TOKENS_PER_SEC", "10"))
QUEUE_COST_SECONDS = float(os.getenv("RELAY_SCHED_QUEUE_COST_SECONDS", "0.05"))
MEMORY_COST_SECONDS = float(os.getenv("RELAY_SCHED_MEMORY_COST_SECONDS", "2.0"))
THERMAL_COST_SECONDS = float(os.getenv("RELAY_SCHED_THERMAL_COST_SECONDS", "10.0"))
JITTER_COST_SECONDS_PER_MS = float(os.getenv("RELAY_SCHED_JITTER_COST_SECONDS_PER_MS", "0.001"))


class SchedulingError(RuntimeError):
    """Raised when the coordinator cannot select a worker."""


@dataclass(frozen=True)
class WorkerChoice:
    """Selected worker plus scheduler diagnostics."""

    worker: WorkerSnapshot
    cost: float
    matched_blocks: int
    matched_tokens: int
    prompt_tokens: int
    uncached_prompt_tokens: int


def choose_worker(
    request: Mapping[str, object],
    workers: Sequence[WorkerSnapshot],
) -> WorkerChoice:
    """Choose the lowest-cost worker for an OpenAI-style chat completion request."""
    if not workers:
        raise SchedulingError("No workers registered")
    requested_model = _requested_model(request)
    eligible_workers = [worker for worker in workers if worker.supports_model(requested_model)]
    if not eligible_workers:
        if requested_model:
            raise SchedulingError(f"No workers can serve requested model '{requested_model}'")
        raise SchedulingError("No workers can serve this request")

    prompt_text = request_to_prefix_text(request)
    expected_output_tokens = _expected_output_tokens(request)
    candidates = [
        _score_worker(prompt_text, expected_output_tokens, worker) for worker in eligible_workers
    ]
    return min(candidates, key=lambda choice: (choice.cost, choice.worker.node_id))


def _score_worker(
    prompt_text: str,
    expected_output_tokens: int,
    worker: WorkerSnapshot,
) -> WorkerChoice:
    telemetry = worker.telemetry
    prefix_cache = telemetry.prefix_cache
    prefix_config = prefix_cache.to_hash_config()
    request_prefix = compute_prefix_hashes_from_text(prompt_text, prefix_config)

    matched_blocks = longest_prefix_match(
        request_prefix.block_hashes,
        prefix_cache.block_hashes,
    )
    matched_tokens = prefix_hit_tokens(matched_blocks, prefix_config.block_size_tokens)
    prompt_tokens = request_prefix.estimated_tokens
    uncached_prompt_tokens = max(0, prompt_tokens - matched_tokens)

    bucket = prompt_length_bucket_id(prompt_tokens)
    prefill_seconds = uncached_prompt_tokens / _prefill_speed(telemetry)
    decode_seconds = expected_output_tokens / _decode_speed(telemetry, bucket)
    load_penalty = telemetry.qw * QUEUE_COST_SECONDS
    memory_penalty = telemetry.mw * MEMORY_COST_SECONDS
    jitter_penalty = telemetry.jw * JITTER_COST_SECONDS_PER_MS
    thermal_penalty = telemetry.theta_w * THERMAL_COST_SECONDS

    return WorkerChoice(
        worker=worker,
        cost=prefill_seconds
        + decode_seconds
        + load_penalty
        + memory_penalty
        + jitter_penalty
        + thermal_penalty,
        matched_blocks=matched_blocks,
        matched_tokens=matched_tokens,
        prompt_tokens=prompt_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
    )


def _prefill_speed(telemetry: Telemetry) -> float:
    if telemetry.sprefill_tokens_per_sec > 0:
        return telemetry.sprefill_tokens_per_sec
    return max(DEFAULT_PREFILL_TOKENS_PER_SEC, 1e-6)


def _decode_speed(telemetry: Telemetry, bucket: str) -> float:
    bucket_speed = telemetry.sw_by_bucket.get(bucket)
    if bucket_speed is not None and bucket_speed > 0:
        return bucket_speed
    observed_speeds = [speed for speed in telemetry.sw_by_bucket.values() if speed > 0]
    if observed_speeds:
        return max(observed_speeds)
    return max(DEFAULT_DECODE_TOKENS_PER_SEC, 1e-6)


def _expected_output_tokens(request: Mapping[str, object]) -> int:
    for key in ("max_completion_tokens", "max_tokens"):
        value = request.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_EXPECTED_OUTPUT_TOKENS


def _requested_model(request: Mapping[str, object]) -> str | None:
    value = request.get("model")
    if isinstance(value, str) and value:
        return value
    return None
