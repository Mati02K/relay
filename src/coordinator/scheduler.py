"""Cache-aware worker scheduler.

The coordinator uses this module to choose one worker for each incoming
OpenAI-style request. The scheduler follows the paper-style cost function:

``queue_weight * q_w / s_w(b)
+ prefix_miss_weight * (1 - overlap(w, r))
+ memory_weight * m_w
+ jitter_weight * j_w / j_max
+ thermal_weight * theta_w``

Lower cost is better. All weights are configurable through environment
variables so the scheduler can be tuned without changing code.
"""

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

DEFAULT_DECODE_TOKENS_PER_SEC = float(os.getenv("RELAY_DEFAULT_DECODE_TOKENS_PER_SEC", "10"))

# Paper formula weights. Defaults keep terms on similar order for current local
# telemetry; production tuning should be based on real cluster measurements.
QUEUE_WEIGHT = float(os.getenv("RELAY_SCHED_QUEUE_WEIGHT", "1.0"))
PREFIX_MISS_WEIGHT = float(os.getenv("RELAY_SCHED_PREFIX_MISS_WEIGHT", "1.0"))
MEMORY_WEIGHT = float(os.getenv("RELAY_SCHED_MEMORY_WEIGHT", "1.0"))
JITTER_WEIGHT = float(os.getenv("RELAY_SCHED_JITTER_WEIGHT", "1.0"))
THERMAL_WEIGHT = float(os.getenv("RELAY_SCHED_THERMAL_WEIGHT", "1.0"))
DEFAULT_JITTER_MAX_MS = float(os.getenv("RELAY_SCHED_JITTER_MAX_MS", "1.0"))


class SchedulingError(RuntimeError):
    """Raised when the coordinator cannot select a worker."""


@dataclass(frozen=True)
class WorkerChoice:
    """Selected worker plus scheduler diagnostics.

    Attributes:
        worker: Worker selected by the scheduler.
        cost: Final comparable cost; lower is better.
        matched_blocks: Number of request prefix blocks found on the worker.
        matched_tokens: Approximate prompt tokens already reusable on worker.
        prompt_tokens: Approximate total prompt tokens in the request.
        uncached_prompt_tokens: Prompt tokens expected to require prefill work.
        overlap: Fraction of prompt tokens matched by the worker prefix cache.
    """

    worker: WorkerSnapshot
    cost: float
    matched_blocks: int
    matched_tokens: int
    prompt_tokens: int
    uncached_prompt_tokens: int
    overlap: float


def choose_worker(
    request: Mapping[str, object],
    workers: Sequence[WorkerSnapshot],
) -> WorkerChoice:
    """Choose the lowest-cost worker for an OpenAI-style chat completion request.

    The function first filters out workers that do not advertise the requested
    model. It then scores every eligible worker and returns the one with the
    lowest cost. Ties are broken by node id to keep choices deterministic.
    """
    if not workers:
        raise SchedulingError("No workers registered")
    requested_model = _requested_model(request)
    eligible_workers = [worker for worker in workers if worker.supports_model(requested_model)]
    if not eligible_workers:
        if requested_model:
            raise SchedulingError(f"No workers can serve requested model '{requested_model}'")
        raise SchedulingError("No workers can serve this request")

    prompt_text = request_to_prefix_text(request)
    jitter_max = _jitter_max(eligible_workers)
    candidates = [_score_worker(prompt_text, jitter_max, worker) for worker in eligible_workers]
    return min(candidates, key=lambda choice: (choice.cost, choice.worker.node_id))


def _score_worker(
    prompt_text: str,
    jitter_max: float,
    worker: WorkerSnapshot,
) -> WorkerChoice:
    """Compute the paper-style scheduler cost for one worker."""
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
    overlap = _prefix_overlap(matched_tokens, prompt_tokens)

    bucket = prompt_length_bucket_id(prompt_tokens)
    decode_speed = _decode_speed(telemetry, bucket)
    queue_term = QUEUE_WEIGHT * telemetry.qw / decode_speed
    prefix_term = PREFIX_MISS_WEIGHT * (1.0 - overlap)
    memory_term = MEMORY_WEIGHT * telemetry.mw
    jitter_term = JITTER_WEIGHT * telemetry.jw / jitter_max
    thermal_term = THERMAL_WEIGHT * telemetry.theta_w

    return WorkerChoice(
        worker=worker,
        cost=queue_term + prefix_term + memory_term + jitter_term + thermal_term,
        matched_blocks=matched_blocks,
        matched_tokens=matched_tokens,
        prompt_tokens=prompt_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
        overlap=overlap,
    )


def _decode_speed(telemetry: Telemetry, bucket: str) -> float:
    """Return decode speed for a prompt-length bucket with fallback behavior.

    Prefer the speed observed for this request's prompt bucket. If that bucket
    has no data yet, use the best observed decode speed from any bucket. If this
    worker has no request history, use the default decode speed.
    """
    bucket_speed = telemetry.sw_by_bucket.get(bucket)
    if bucket_speed is not None and bucket_speed > 0:
        return bucket_speed
    observed_speeds = [speed for speed in telemetry.sw_by_bucket.values() if speed > 0]
    if observed_speeds:
        return max(observed_speeds)
    return max(DEFAULT_DECODE_TOKENS_PER_SEC, 1e-6)


def _prefix_overlap(matched_tokens: int, prompt_tokens: int) -> float:
    """Return ``overlap(w, r)`` as a value in ``[0, 1]``."""
    if prompt_tokens <= 0:
        return 1.0
    return max(0.0, min(1.0, matched_tokens / prompt_tokens))


def _jitter_max(workers: Sequence[WorkerSnapshot]) -> float:
    """Return ``j_max`` for the current candidate set."""
    observed = max((worker.telemetry.jw for worker in workers), default=0.0)
    return max(observed, DEFAULT_JITTER_MAX_MS, 1e-6)


def _requested_model(request: Mapping[str, object]) -> str | None:
    """Extract the requested model id from an OpenAI-style request body."""
    value = request.get("model")
    if isinstance(value, str) and value:
        return value
    return None
