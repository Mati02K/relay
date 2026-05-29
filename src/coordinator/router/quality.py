"""Per-worker quality lookup and the quality-routing cost term.

A worker self-advertises a ``model_quality`` value in [0.0, 1.0] via
its registration metadata. ``1.0`` means "strongest model available
in the cluster"; ``0.0`` means "weakest". The scheduler combines this
with the prompt's complexity score to produce a cost term that biases
hard prompts toward high-quality workers.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_QUALITY = 0.5


def worker_model_quality(metadata: dict[str, Any]) -> float:
    """Return the worker's advertised ``model_quality`` in [0, 1].

    Falls back to :data:`DEFAULT_MODEL_QUALITY` if the worker did not
    advertise a value or advertised something out of range. Workers can
    set the value via the ``RELAY_MODEL_QUALITY`` environment variable
    at registration time.
    """
    raw = metadata.get("model_quality")
    if raw is None:
        return DEFAULT_MODEL_QUALITY
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MODEL_QUALITY
    if value != value or value < 0.0:
        return DEFAULT_MODEL_QUALITY
    return min(1.0, value)


def quality_routing_term(
    nu: float,
    complexity: float,
    model_quality: float,
) -> float:
    """Return the RouteLLM-style quality cost term.

    ``cost += nu * complexity * (1 - model_quality)``

    The term is zero for trivial prompts (``complexity == 0``) and zero
    for the strongest worker (``model_quality == 1``). It penalises
    sending hard prompts to weak models while leaving easy prompts
    free to land on whichever worker is otherwise cheapest.
    """
    if nu <= 0.0 or complexity <= 0.0:
        return 0.0
    quality = max(0.0, min(1.0, model_quality))
    return nu * complexity * (1.0 - quality)
