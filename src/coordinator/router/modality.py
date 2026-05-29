"""Modality detection and worker-capability filtering.

A vision-capable worker should serve image+text prompts; a text-only
worker should not. This module gives the scheduler the two primitives
it needs to enforce that:

* :func:`detect_request_modalities` — walk an OpenAI-style request body
  and return the set of modalities it contains
* :func:`worker_supports_modalities` — read a worker's advertised
  capability set and check whether it covers every required modality

Workers self-advertise their modality set via metadata, typically via
the ``RELAY_MODALITIES`` environment variable at registration time
(comma-separated, e.g. ``RELAY_MODALITIES=text,image``).

The classifier deliberately defaults to ``{"text"}`` for both an
unannotated request and an unannotated worker so that any existing
worker that pre-dates this module keeps serving text requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_REQUEST_MODALITIES: frozenset[str] = frozenset({"text"})
DEFAULT_WORKER_MODALITIES: frozenset[str] = frozenset({"text"})

# OpenAI-style content-part `type` values, grouped by modality. Anything
# unknown falls through to ``"text"`` so a new content type cannot
# silently route to a worker that does not understand it.
_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image_url",
        "input_image",
        "image",
    }
)
_AUDIO_CONTENT_TYPES = frozenset(
    {
        "input_audio",
        "audio",
    }
)
_VIDEO_CONTENT_TYPES = frozenset(
    {
        "input_video",
        "video",
    }
)


def detect_request_modalities(request: Mapping[str, Any]) -> set[str]:
    """Return the modalities present in an OpenAI-style chat request.

    Text is always included so a plain-text request resolves to
    ``{"text"}``. Image, audio, and video parts add their own keys.
    Any request body that the function does not recognise resolves to
    the safe default ``{"text"}`` — never empty, so the filter still
    yields a sensible candidate set downstream.
    """
    modalities: set[str] = {"text"}
    messages = request.get("messages")
    if not isinstance(messages, list):
        return modalities
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in _IMAGE_CONTENT_TYPES:
                    modalities.add("image")
                elif part_type in _AUDIO_CONTENT_TYPES:
                    modalities.add("audio")
                elif part_type in _VIDEO_CONTENT_TYPES:
                    modalities.add("video")
    return modalities


def worker_modalities(metadata: dict[str, Any]) -> set[str]:
    """Return the modality set a worker advertises.

    Accepts either ``modalities`` (preferred) or ``capabilities`` as a
    list of strings. Falls back to :data:`DEFAULT_WORKER_MODALITIES`
    when neither is present so existing workers keep working.
    """
    for key in ("modalities", "capabilities"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            normalised = {str(item).strip().lower() for item in raw if item}
            normalised.discard("")
            if normalised:
                return normalised
    return set(DEFAULT_WORKER_MODALITIES)


def worker_supports_modalities(
    metadata: dict[str, Any],
    required: set[str],
) -> bool:
    """Return ``True`` iff the worker advertises every required modality."""
    if not required:
        return True
    return required.issubset(worker_modalities(metadata))
