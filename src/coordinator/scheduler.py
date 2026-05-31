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

The model chart is gated entirely by ``nu`` (the RouteLLM quality knob). When
``nu == 0`` the chart is not consulted for routing at all — no model-id filter,
no input-modality filter, no skill filter — so workers are interchangeable
compute scored by the base physical-signal cost only. When ``nu > 0`` the chart
layer turns on and routing runs, before scoring:

1. A hard model-id filter — workers that don't serve the requested model id
   are dropped.
2. A hard input-modality filter (text / image / audio / video) — anything
   the chart says a model can't accept is dropped.
3. A soft skill filter (coding / reasoning / instruct) — if at least one
   worker advertises every needed skill, only those are scored;
   otherwise scoring proceeds with the entire modality-eligible set so the
   request still goes somewhere.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from coordinator.router import (
    Classification,
    ModelEntry,
    classify,
    detect_request_modalities,
    lookup,
    quality_routing_term,
)
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import (
    compute_prefix_hashes_from_text,
    longest_prefix_match,
    prefix_hit_tokens,
    request_to_prefix_text,
)

DEFAULT_JITTER_MAX_MS = float(os.getenv("RELAY_SCHED_JITTER_MAX_MS", "1.0"))
# Queue depth (q_w) is a raw count of in-flight requests (0..parallel slots).
# Normalize it to [0, 1] by dividing by this capacity so the queue term shares
# the same scale as the other [0, 1] signals; values above capacity clamp to 1.0.
# Defaults to the llama.cpp ``--parallel`` default; set RELAY_SCHED_QUEUE_MAX to
# match a non-default slot count.
DEFAULT_QUEUE_MAX = max(1.0, float(os.getenv("RELAY_SCHED_QUEUE_MAX", "4")))


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


SCHEDULER_MODES: tuple[str, ...] = ("cost", "round_robin")


def _initial_mode() -> str:
    """Resolve the scheduler mode from ``RELAY_SCHED_MODE``, default ``cost``."""
    raw = os.getenv("RELAY_SCHED_MODE", "cost").strip().lower()
    return raw if raw in SCHEDULER_MODES else "cost"


# Live scheduler mode. ``cost`` runs the full cost-based scoring path; ``round_robin``
# bypasses scoring and rotates across eligible workers. Flipped at runtime via
# ``POST /v1/scheduler/mode``; not persisted across coordinator restarts.
SCHEDULER_MODE: str = _initial_mode()

# Process-wide counter that drives the rotation in round-robin mode. Reset on
# coordinator restart by design (in-memory only). Not exposed externally.
_RR_COUNTER: itertools.count[int] = itertools.count()


def get_mode() -> str:
    """Return the current scheduler mode."""
    return SCHEDULER_MODE


def set_mode(mode: str) -> str:
    """Set the scheduler mode. Raises ``ValueError`` on an unknown mode."""
    global SCHEDULER_MODE, _RR_COUNTER
    normalised = mode.strip().lower() if isinstance(mode, str) else ""
    if normalised not in SCHEDULER_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Allowed: {list(SCHEDULER_MODES)}")
    if normalised != SCHEDULER_MODE:
        # Reset rotation when entering round-robin so a flip on stage starts
        # from the first node — predictable, demo-friendly.
        _RR_COUNTER = itertools.count()
    SCHEDULER_MODE = normalised
    return SCHEDULER_MODE


def _reset_round_robin() -> None:
    """Reset the round-robin counter. Test-only helper."""
    global _RR_COUNTER
    _RR_COUNTER = itertools.count()


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
        model_quality: Quality value from the model chart for this worker.
        quality_term: Final ``nu * complexity * (1 - model_quality)`` cost.
        model_id: Model id this worker was matched on, for routing audit.
        skills_needed: Skill tags the classifier asked for.
        skills_available: Skill tags the chosen worker's chart entry advertises.
        skill_match: Whether ``skills_needed ⊆ skills_available``.
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
    model_id: str = ""
    skills_needed: frozenset[str] = frozenset()
    skills_available: frozenset[str] = frozenset()
    skill_match: bool = False


def choose_worker(
    request: Mapping[str, object],
    workers: Sequence[WorkerSnapshot],
) -> WorkerChoice:
    """Choose the lowest-cost worker for an OpenAI-style chat completion request.

    ``nu`` is the master switch for the model chart. When ``nu == 0`` the chart
    has zero influence: there is **no model-id filter, no input-modality filter,
    and no skill filter** — every worker is an interchangeable compute target
    scored by the base physical-signal cost (queue, prefix, memory, jitter,
    thermal) alone, and round-robin rotates across all of them. When ``nu > 0``
    the whole chart layer turns on: hard model-id and input-modality filters,
    the soft skill filter, and the RouteLLM quality term. Surviving candidates
    are scored and the minimum-cost worker wins, ties broken by node id.
    """
    if not workers:
        raise SchedulingError("No workers registered")
    classification = classify(request)

    worker_entries: list[tuple[WorkerSnapshot, ModelEntry]] = [
        (worker, _entry_for_worker(worker)) for worker in workers
    ]

    if WEIGHTS.nu > 0.0:
        # Chart on: enforce model-id then input-modality eligibility (hard).
        requested_model = _requested_model(request)
        required_modalities = detect_request_modalities(request)

        model_eligible = [
            (worker, entry)
            for worker, entry in worker_entries
            if worker.supports_model(requested_model)
        ]
        if not model_eligible:
            if requested_model:
                raise SchedulingError(f"No workers can serve requested model '{requested_model}'")
            raise SchedulingError("No workers can serve this request")

        eligible = [
            (worker, entry)
            for worker, entry in model_eligible
            if required_modalities.issubset(entry.input_modalities)
        ]
        if not eligible:
            non_text = sorted(required_modalities - {"text"})
            if non_text:
                raise SchedulingError(
                    f"No workers can serve required modalities {non_text}"
                    + (f" for model '{requested_model}'" if requested_model else "")
                )
            raise SchedulingError("No workers can serve this request")
    else:
        # Chart off: workers are interchangeable, no model/modality filter.
        eligible = worker_entries

    if SCHEDULER_MODE == "round_robin":
        return _round_robin_choice(eligible)

    # Soft skill filter is part of the chart layer, so it too is nu-gated.
    if WEIGHTS.nu > 0.0:
        skill_preferred = [
            (worker, entry)
            for worker, entry in eligible
            if classification.skills_needed.issubset(entry.skills)
        ]
        scoring_pool = skill_preferred if skill_preferred else eligible
    else:
        scoring_pool = eligible

    prompt_text = request_to_prefix_text(request)
    jitter_max = _jitter_max([worker for worker, _ in scoring_pool])
    candidates = [
        _score_worker(prompt_text, jitter_max, classification, worker, entry)
        for worker, entry in scoring_pool
    ]
    return min(candidates, key=lambda choice: (choice.cost, choice.worker.node_id))


def _round_robin_choice(
    eligible: list[tuple[WorkerSnapshot, ModelEntry]],
) -> WorkerChoice:
    """Pick the next worker by rotating index in ``eligible`` (stable order).

    The selector ignores telemetry, the model chart's quality, and the soft
    skill filter — that's the whole point of the round-robin mode. Hard
    eligibility (model id + input modality) has already been enforced by the
    caller.
    """
    ordered = sorted(eligible, key=lambda pair: pair[0].node_id)
    idx = next(_RR_COUNTER) % len(ordered)
    worker, entry = ordered[idx]
    logger.debug(
        "scheduler round_robin | nodeId={} modelId={} index={}/{}",
        worker.node_id,
        entry.model_id,
        idx,
        len(ordered),
    )
    return WorkerChoice(
        worker=worker,
        cost=0.0,
        matched_blocks=0,
        matched_tokens=0,
        prompt_tokens=0,
        uncached_prompt_tokens=0,
        overlap=0.0,
        complexity=0.0,
        model_quality=entry.quality,
        quality_term=0.0,
        model_id=entry.model_id,
        skills_needed=frozenset(),
        skills_available=entry.skills,
        skill_match=False,
    )


def _score_worker(
    prompt_text: str,
    jitter_max: float,
    classification: Classification,
    worker: WorkerSnapshot,
    entry: ModelEntry,
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
    normalized_queue = min(1.0, max(0, telemetry.qw) / DEFAULT_QUEUE_MAX)
    queue_term = weights.queue * normalized_queue
    prefix_term = weights.prefix_miss * (1.0 - overlap)
    memory_term = weights.memory * telemetry.mw
    jitter_term = weights.jitter * telemetry.jw / jitter_max
    thermal_term = weights.thermal * telemetry.theta_w
    worker_weight = _worker_weight(worker)

    quality_term = quality_routing_term(weights.nu, classification.complexity, entry.quality)

    cost = (
        queue_term
        + prefix_term
        + memory_term
        + jitter_term
        + thermal_term
        + quality_term
        - worker_weight
    )

    skill_match = classification.skills_needed.issubset(entry.skills)

    logger.debug(
        "scheduler cost | nodeId={} modelId={} "
        "| queue: w={} qw={} term={} "
        "| prefix_miss: w={} miss={} term={} "
        "| memory: w={} mw={} term={} "
        "| jitter: w={} jw={} jmax={} term={} "
        "| thermal: w={} theta_w={} term={} "
        "| nu={} complexity={} quality={} skills_needed={} skills_avail={} "
        "skill_match={} quality_term={} "
        "| worker_weight={} "
        "| cost={}",
        worker.node_id,
        entry.model_id,
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
        classification.complexity,
        entry.quality,
        sorted(classification.skills_needed),
        sorted(entry.skills),
        skill_match,
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
        complexity=classification.complexity,
        model_quality=entry.quality,
        quality_term=quality_term,
        model_id=entry.model_id,
        skills_needed=classification.skills_needed,
        skills_available=entry.skills,
        skill_match=skill_match,
    )


def _entry_for_worker(worker: WorkerSnapshot) -> ModelEntry:
    """Resolve the model chart entry for the worker's first loaded model."""
    model_id, quant = _primary_model_descriptor(worker.metadata)
    return lookup(model_id, quant_hint=quant)


def _primary_model_descriptor(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(model_id, quant)`` for the worker's first loaded model."""
    models = metadata.get("models")
    if not isinstance(models, list):
        return None, None
    for model in models:
        if not isinstance(model, dict):
            continue
        if not model.get("loaded", True):
            continue
        model_id = model.get("id")
        quant = model.get("quant")
        return (
            model_id if isinstance(model_id, str) else None,
            quant if isinstance(quant, str) else None,
        )
    return None, None


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
