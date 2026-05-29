"""Quality-aware request router.

This subpackage owns the RouteLLM-style routing logic: it scores how
"hard" a user prompt is, reads each worker's advertised model quality,
and returns a cost-function term that biases routing toward higher-
quality workers when the prompt warrants it.

The scheduler in :mod:`coordinator.scheduler` plugs the term into the
existing additive cost model via :func:`quality_routing_term`.
"""

from coordinator.router.classifier import (
    DEFAULT_LENGTH_NORM_TOKENS,
    estimate_complexity_score,
)
from coordinator.router.modality import (
    DEFAULT_REQUEST_MODALITIES,
    DEFAULT_WORKER_MODALITIES,
    detect_request_modalities,
    worker_modalities,
    worker_supports_modalities,
)
from coordinator.router.quality import (
    DEFAULT_MODEL_QUALITY,
    quality_routing_term,
    worker_model_quality,
)

__all__ = [
    "DEFAULT_LENGTH_NORM_TOKENS",
    "DEFAULT_MODEL_QUALITY",
    "DEFAULT_REQUEST_MODALITIES",
    "DEFAULT_WORKER_MODALITIES",
    "detect_request_modalities",
    "estimate_complexity_score",
    "quality_routing_term",
    "worker_modalities",
    "worker_model_quality",
    "worker_supports_modalities",
]
