"""Quality-routing cost term.

The cost term is the RouteLLM-style ``nu * complexity * (1 - quality)``
contribution to the scheduler. Quality is now sourced from the
coordinator-side model chart (:mod:`coordinator.router.model_chart`)
keyed by the worker's loaded model id, not from worker-advertised
metadata.
"""

from __future__ import annotations

DEFAULT_MODEL_QUALITY = 0.4


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
