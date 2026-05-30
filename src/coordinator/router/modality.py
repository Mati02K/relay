"""Input-modality detection for OpenAI-style chat requests.

This module owns the *hard* input-modality constraint only: detecting
that a request contains image / audio / video parts so the scheduler
can reject it on workers whose chart entry doesn't include the
corresponding ``input_modalities`` flag.

Skill specialization (coding / reasoning / chat) is detected by
:mod:`coordinator.router.classifier` and applied as a *soft* preference
in the scheduler. Worker-advertised modality lists are no longer
honoured — capabilities come from the model chart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_REQUEST_MODALITIES: frozenset[str] = frozenset({"text"})

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
    """Return the input modalities present in an OpenAI-style chat request.

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
