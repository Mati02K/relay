"""Chart-driven request router.

The router decides three things for every chat request:

* which input modalities the request needs (text / image / audio / video) — a
  hard constraint enforced before scoring
* which skills the prompt benefits from (coding / reasoning / instruct) — a
  soft preference that narrows the candidate set when at least one worker
  matches, falls through to "all eligible" otherwise
* how hard the prompt is, used by the RouteLLM-style ``nu`` quality term

The coordinator-side :mod:`coordinator.router.model_chart` is the
authority on each model's skills, quality, and input modalities;
workers only publish their loaded model id.
"""

from coordinator.router.classifier import (
    DEFAULT_LENGTH_NORM_TOKENS,
    Classification,
    classify,
    detect_skills,
    estimate_complexity_score,
)
from coordinator.router.modality import (
    DEFAULT_REQUEST_MODALITIES,
    detect_request_modalities,
)
from coordinator.router.model_chart import (
    DEFAULT_INPUT_MODALITIES,
    DEFAULT_QUALITY,
    DEFAULT_SKILLS,
    ModelEntry,
    lookup,
    reload_chart,
)
from coordinator.router.quality import (
    DEFAULT_MODEL_QUALITY,
    quality_routing_term,
)

__all__ = [
    "Classification",
    "DEFAULT_INPUT_MODALITIES",
    "DEFAULT_LENGTH_NORM_TOKENS",
    "DEFAULT_MODEL_QUALITY",
    "DEFAULT_QUALITY",
    "DEFAULT_REQUEST_MODALITIES",
    "DEFAULT_SKILLS",
    "ModelEntry",
    "classify",
    "detect_request_modalities",
    "detect_skills",
    "estimate_complexity_score",
    "lookup",
    "quality_routing_term",
    "reload_chart",
]
