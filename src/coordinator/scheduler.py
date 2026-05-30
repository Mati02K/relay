"""Cache-aware worker scheduler.

The coordinator uses this module to choose one worker for each incoming
OpenAI-style request. The scheduler follows the paper-style cost function
extended with a RouteLLM-style quality-routing term:

``queue_weight    * q_w
+ prefix_miss_weight * (1 - overlap(w, r))
+ memory_weight     * m_w
+ jitter_weight     * j_w / j_max
+ thermal_weight    * theta_w
+ nu                * complexity(r) * (1 - model_quality_w)
- worker_weight(w)``

Lower cost is better. The global weights are configurable through environment
variables so the scheduler can be tuned without changing code. The
quality term is sourced from :mod:`coordinator.router`. ``worker_weight(w)``
is a per-worker preference advertised by each worker in its metadata;
default ``0.0`` makes it inert.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from coordinator.router import (
    detect_request_modalities,
    estimate_complexity_score,
    quality_routing_term,
    worker_model_quality,
    worker_supports_modalities,
)
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import (
    compute_prefix_hashes_from_text,
    longest_prefix_match,
    prefix_hit_tokens,
    request_to_prefix_text,
)

DEFAULT_JITTER_MAX_MS = float(os.getenv("RELAY_SCHED_JITTER_MAX_MS", "1.0"))


@dataclass
class SchedulerWeights:
    """Mutable container for the paper-style cost-function weights.

    Held as a single object instead of module-level constants so the dashboard
    can hot-swap values via ``POST /v1/scheduler/weights`` without a restart.
    All weights are clamped to ``[0.0, 1.0]`` at the API layer.
    """

    queue: float = 1.0
    prefix_miss: float = 1.0
    memory: float = 1.0
    jitter: float = 1.0
    thermal: float = 1.0
    nu: float = 0.0

    @classmethod
    def from_env(cls) -> SchedulerWeights:
        """Initialise from ``RELAY_SCHED_*_WEIGHT`` env vars.

        ``RELAY_SCHED_NU_WEIGHT`` controls the RouteLLM-style quality
        routing term; default ``0.0`` keeps the scheduler behaviour
        backwards-compatible until quality routing is opted into.
        """
        return cls(
            queue=float(os.getenv("RELAY_SCHED_QUEUE_WEIGHT", "1.0")),
            prefix_miss=float(os.getenv("RELAY_SCHED_PREFIX_MISS_WEIGHT", "1.0")),
            memory=float(os.getenv("RELAY_SCHED_MEMORY_WEIGHT", "1.0")),
            jitter=float(os.getenv("RELAY_SCHED_JITTER_WEIGHT", "1.0")),
            thermal=float(os.getenv("RELAY_SCHED_THERMAL_WEIGHT", "1.0")),
            nu=float(os.getenv("RELAY_SCHED_NU_WEIGHT", "0.0")),
        )

    def update(self, **fields: float) -> None:
        """Replace named fields in-place. Unknown names raise ``KeyError``."""
        for name, value in fields.items():
            if not hasattr(self, name):
                raise KeyError(f"Unknown weight field: {name}")
            setattr(self, name, float(value))

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-friendly snapshot."""
        return {
            "queue": self.queue,
            "prefix_miss": self.prefix_miss,
            "memory": self.memory,
            "jitter": self.jitter,
            "thermal": self.thermal,
            "nu": self.nu,
        }


# Live, mutable weights. Reads in `_score_worker` go through this object so
# `.update(...)` from the API changes scheduling on the very next request.
WEIGHTS = SchedulerWeights.from_env()

# Live, in-memory overrides for per-worker preference. Keyed by node id.
# The worker's metadata still carries the baseline weight set at init; this
# map lets operators temporarily boost or penalise specific workers from the
# dashboard without re-running ``relay init`` or restarting the worker. The
# override is not persisted across coordinator restarts by design — it's a
# tuning knob, not configuration.
WORKER_WEIGHT_OVERRIDES: dict[str, float] = {}

# Live, in-memory overrides for the per-worker router knobs (``model_quality``
# and ``modalities``). Same lifecycle as ``WORKER_WEIGHT_OVERRIDES``: not
# persisted, populated by the dashboard, consumed at routing time. Each value
# is a partial metadata dict whose keys shadow the worker's advertised
# metadata for the duration of the coordinator process.
WORKER_ROUTER_OVERRIDES: dict[str, dict[str, Any]] = {}


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
        complexity: RouteLLM-style complexity score for the request.
        model_quality: Quality value advertised by this worker.
        quality_term: Final ``nu * complexity * (1 - model_quality)`` cost.
    """

    worker: WorkerSnapshot
    cost: float
    matched_blocks: int
    matched_tokens: int
    prompt_tokens: int
    uncached_prompt_tokens: int
    overlap: float
    complexity: float = 0.0
    model_quality: float = 0.0
    quality_term: float = 0.0


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
    required_modalities = detect_request_modalities(request)
    eligible_workers = [
        worker
        for worker in workers
        if worker.supports_model(requested_model)
        and worker_supports_modalities(_effective_metadata(worker), required_modalities)
    ]
    if not eligible_workers:
        non_text = sorted(required_modalities - {"text"})
        if non_text:
            raise SchedulingError(
                f"No workers can serve required modalities {non_text}"
                + (f" for model '{requested_model}'" if requested_model else "")
            )
        if requested_model:
            raise SchedulingError(f"No workers can serve requested model '{requested_model}'")
        raise SchedulingError("No workers can serve this request")

    prompt_text = request_to_prefix_text(request)
    jitter_max = _jitter_max(eligible_workers)
    complexity = estimate_complexity_score(prompt_text)
    candidates = [
        _score_worker(prompt_text, jitter_max, complexity, worker) for worker in eligible_workers
    ]
    return min(candidates, key=lambda choice: (choice.cost, choice.worker.node_id))


def _score_worker(
    prompt_text: str,
    jitter_max: float,
    complexity: float,
    worker: WorkerSnapshot,
) -> WorkerChoice:
    """Compute the scheduler cost for one worker.

    Cost = base 5-term paper formula (queue, prefix-miss, memory, jitter,
    thermal) plus the RouteLLM-style ``nu * complexity * (1 - quality)``
    term contributed by :mod:`coordinator.router`.
    """
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

    weights = WEIGHTS
    queue_term = weights.queue * telemetry.qw
    prefix_term = weights.prefix_miss * (1.0 - overlap)
    memory_term = weights.memory * telemetry.mw
    jitter_term = weights.jitter * telemetry.jw / jitter_max
    thermal_term = weights.thermal * telemetry.theta_w
    worker_weight = _worker_weight(worker)

    model_quality = worker_model_quality(_effective_metadata(worker))
    quality_term = quality_routing_term(weights.nu, complexity, model_quality)

    cost = (
        queue_term
        + prefix_term
        + memory_term
        + jitter_term
        + thermal_term
        + quality_term
        - worker_weight
    )

    logger.debug(
        "scheduler cost | nodeId={} "
        "| queue: w={} qw={} term={} "
        "| prefix_miss: w={} miss={} term={} "
        "| memory: w={} mw={} term={} "
        "| jitter: w={} jw={} jmax={} term={} "
        "| thermal: w={} theta_w={} term={} "
        "| nu={} complexity={} model_quality={} quality_term={} "
        "| worker_weight={} "
        "| cost={}",
        worker.node_id,
        weights.queue,
        telemetry.qw,
        queue_term,
        weights.prefix_miss,
        1.0 - overlap,
        prefix_term,
        weights.memory,
        telemetry.mw,
        memory_term,
        weights.jitter,
        telemetry.jw,
        jitter_max,
        jitter_term,
        weights.thermal,
        telemetry.theta_w,
        thermal_term,
        weights.nu,
        complexity,
        model_quality,
        quality_term,
        worker_weight,
        cost,
    )

    return WorkerChoice(
        worker=worker,
        cost=cost,
        matched_blocks=matched_blocks,
        matched_tokens=matched_tokens,
        prompt_tokens=prompt_tokens,
        uncached_prompt_tokens=uncached_prompt_tokens,
        overlap=overlap,
        complexity=complexity,
        model_quality=model_quality,
        quality_term=quality_term,
    )


def _effective_metadata(worker: WorkerSnapshot) -> dict[str, Any]:
    """Return worker metadata with live router overrides applied.

    Dashboard-driven overrides in :data:`WORKER_ROUTER_OVERRIDES` win over the
    metadata the worker advertised at registration time, so changes from
    ``POST /v1/scheduler/worker_router`` take effect on the next request
    without restarting either process.
    """
    override = WORKER_ROUTER_OVERRIDES.get(worker.node_id)
    if not override:
        return dict(worker.metadata)
    merged = dict(worker.metadata)
    merged.update(override)
    return merged


def _worker_weight(worker: WorkerSnapshot) -> float:
    """Return the worker's effective preference, clamped to ``[-1.0, 1.0]``.

    Live coordinator overrides win over the worker's advertised metadata so
    dashboard tweaks take effect on the next request without a restart.
    """
    override = WORKER_WEIGHT_OVERRIDES.get(worker.node_id)
    raw = override if override is not None else worker.metadata.get("weight", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, value))


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
