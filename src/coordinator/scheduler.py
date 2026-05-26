"""Request scheduler implementing the §3.2 cost function from the project proposal.

Given a chat request and the current ``WorkerRegistry`` snapshot, ``pick_worker``
returns the node id that minimizes:

    cost(w, r) = alpha * (q_w / s_w(b))
               + beta  * (1 - overlap(w, r))
               + gamma * m_w
               + delta * (j_w / j_max)
               + epsilon * theta_w
               + phi   * 1[T > K] * (1 - c_w)              # phase-aware bias
               + nu    * complexity(r) * (1 - quality_w)   # quality-aware (RouteLLM-style)

All weights are configurable via env vars so the ablation study can zero out
any single term without touching this code.

The final ``nu`` term implements the RouteLLM idea (Ong et al., 2024) inside
our cost function: when a prompt is complex, prefer workers running a higher
quality model. ``quality_w`` is self-reported by each worker in its
membership metadata (``model_quality`` 0..1). ``complexity(r)`` is a cheap
text heuristic today; swapping in a BERT-style classifier later changes only
``estimate_complexity_score`` and nothing else.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

from loguru import logger

from coordinator.worker_registry import WorkerState
from worker.inference.base import (
    estimate_tokens_from_text,
    prefix_chunk_hashes_from_text,
    prefix_overlap_fraction,
    prompt_length_bucket_id,
)

_DEFAULT_DECODE_TOKS_PER_SEC: float = 20.0
_DEFAULT_COMPUTE_STRENGTH: float = 0.5
_DEFAULT_MODEL_QUALITY: float = 0.5

# Keywords that bump the heuristic complexity score. Kept very small on
# purpose; the contract is "swap me out for a learned classifier later".
_COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "explain", "compare", "analyze", "derive", "prove", "why",
    "step by step", "reasoning", "design",
)


@dataclass(frozen=True)
class CostWeights:
    """Scheduler weights; defaults read from env vars so ablations need no code changes."""

    alpha: float = float(os.getenv("RELAY_ALPHA", "1.0"))
    beta: float = float(os.getenv("RELAY_BETA", "1.0"))
    gamma: float = float(os.getenv("RELAY_GAMMA", "1.0"))
    delta: float = float(os.getenv("RELAY_DELTA", "1.0"))
    epsilon: float = float(os.getenv("RELAY_EPSILON", "1.0"))
    phi: float = float(os.getenv("RELAY_PHI", "0.0"))
    nu: float = float(os.getenv("RELAY_NU", "0.0"))
    long_prompt_threshold: int = int(os.getenv("RELAY_LONG_PROMPT_TOKENS", "1024"))
    jitter_max_ms: float = float(os.getenv("RELAY_JITTER_MAX_MS", "100.0"))
    complexity_length_norm_tokens: int = int(
        os.getenv("RELAY_COMPLEXITY_LENGTH_NORM_TOKENS", "2048")
    )

    @classmethod
    def from_env(cls) -> "CostWeights":
        """Construct from current env vars (re-read on every call)."""
        return cls()

    @classmethod
    def round_robin(cls) -> "CostWeights":
        """Degenerate setting: all weights zero, so cost(w) is identical and choice is arbitrary."""
        return cls(
            alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, epsilon=0.0, phi=0.0, nu=0.0,
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-worker cost decomposition used for logging and ablation analysis."""

    node_id: str
    total: float
    queue_term: float
    cache_term: float
    memory_term: float
    jitter_term: float
    thermal_term: float
    phase_term: float
    quality_term: float
    overlap: float
    complexity: float
    bucket: str


def messages_to_prompt_text(messages: list[dict]) -> str:
    """Concatenate message ``content`` fields for hashing and bucketing."""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
    return "\n".join(parts)


def estimate_complexity_score(
    prompt_text: str,
    length_norm_tokens: int = 2048,
) -> float:
    """Return a [0, 1] heuristic complexity score for a prompt.

    Stand-in for the BERT-style classifier used in RouteLLM (Ong et al., 2024).
    The score combines:

      * length signal: prompt token count divided by ``length_norm_tokens``
      * lexical signal: occurrences of reasoning keywords + code fences
      * structural signal: number of question marks

    The two halves are averaged, so each contributes equally. We deliberately
    keep this cheap and side-effect-free: it is recomputed for every routing
    decision and must add negligible overhead to the hot path.
    """
    if not prompt_text:
        return 0.0

    tokens = estimate_tokens_from_text(prompt_text)
    length_score = min(1.0, tokens / max(1, length_norm_tokens))

    lowered = prompt_text.lower()
    keyword_hits = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in lowered)
    code_blocks = lowered.count("```")
    questions = prompt_text.count("?")
    signal_raw = keyword_hits + 2 * code_blocks + 0.5 * questions
    signal_score = min(1.0, signal_raw / 10.0)

    return min(1.0, 0.5 * length_score + 0.5 * signal_score)


def compute_cost(
    prompt_text: str,
    state: WorkerState,
    weights: CostWeights,
) -> ScoreBreakdown:
    """Score a single worker for one request. Lower is better."""
    tel = state.telemetry
    prompt_hashes = prefix_chunk_hashes_from_text(prompt_text)
    prompt_tokens = estimate_tokens_from_text(prompt_text)
    bucket = prompt_length_bucket_id(prompt_tokens)

    decode_speed = tel.sw_by_bucket.get(bucket, _DEFAULT_DECODE_TOKS_PER_SEC)
    decode_speed = max(decode_speed, 1.0)
    queue_term = weights.alpha * (tel.qw / decode_speed)

    overlap = prefix_overlap_fraction(prompt_hashes, set(tel.prefix_chunk_hashes))
    cache_term = weights.beta * (1.0 - overlap)

    memory_term = weights.gamma * float(tel.mw)

    jitter_norm = min(1.0, tel.jw / weights.jitter_max_ms) if weights.jitter_max_ms > 0 else 0.0
    jitter_term = weights.delta * jitter_norm

    thermal_term = weights.epsilon * float(tel.theta_w)

    compute_strength = float(state.metadata.get("compute_strength", _DEFAULT_COMPUTE_STRENGTH))
    compute_strength = max(0.0, min(1.0, compute_strength))
    is_long_prompt = prompt_tokens > weights.long_prompt_threshold
    phase_term = weights.phi * (1.0 - compute_strength) if is_long_prompt else 0.0

    model_quality = float(state.metadata.get("model_quality", _DEFAULT_MODEL_QUALITY))
    model_quality = max(0.0, min(1.0, model_quality))
    complexity = estimate_complexity_score(prompt_text, weights.complexity_length_norm_tokens)
    quality_term = weights.nu * complexity * (1.0 - model_quality)

    total = (
        queue_term + cache_term + memory_term + jitter_term + thermal_term
        + phase_term + quality_term
    )

    return ScoreBreakdown(
        node_id=state.node_id,
        total=total,
        queue_term=queue_term,
        cache_term=cache_term,
        memory_term=memory_term,
        jitter_term=jitter_term,
        thermal_term=thermal_term,
        phase_term=phase_term,
        quality_term=quality_term,
        overlap=overlap,
        complexity=complexity,
        bucket=bucket,
    )


def pick_worker(
    prompt_text: str,
    workers: dict[str, WorkerState],
    weights: CostWeights | None = None,
) -> tuple[str | None, list[ScoreBreakdown]]:
    """Return ``(winning_node_id, all_score_breakdowns)``.

    Returns ``(None, [])`` if no worker is currently online.
    The breakdown list is useful for ``/v1/cluster``-style debugging and the
    final ablation report.
    """
    weights = weights or CostWeights.from_env()
    online = {nid: s for nid, s in workers.items() if s.online}
    if not online:
        return None, []

    scores = [compute_cost(prompt_text, s, weights) for s in online.values()]
    # Break exact ties (e.g. the all-zero "round-robin" ablation, or two equally
    # idle workers at startup) with a deterministic per-request jitter so we
    # don't always send to the first-iterated worker. The jitter magnitude is
    # well below any real cost difference.
    winner = min(scores, key=lambda sb: (sb.total, random.random()))

    logger.debug(
        "Scheduler picked worker | nodeId={} totalCost={:.4f} overlap={:.3f} bucket={}",
        winner.node_id,
        winner.total,
        winner.overlap,
        winner.bucket,
    )
    return winner.node_id, scores


def estimate_ttft_ms(state: WorkerState, prompt_tokens: int) -> float:
    """Coarse TTFT estimate used by SLO admission control (paper §3.2)."""
    tel = state.telemetry
    prefill_speed = tel.sprefill_tokens_per_sec
    if prefill_speed <= 0:
        prefill_speed = _DEFAULT_DECODE_TOKS_PER_SEC
    queue_wait_s = tel.qw / max(prefill_speed, 1.0)
    prefill_s = prompt_tokens / max(prefill_speed, 1.0)
    return (queue_wait_s + prefill_s) * 1000.0 + state.rtt_ms_ema


def _safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(den):
        return math.inf
    return num / den
