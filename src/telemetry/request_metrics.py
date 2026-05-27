"""Telemetry calculated by the worker around generation requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from telemetry.prefix_cache import (
    PrefixHashConfig,
    PrefixHashLRU,
    compute_prefix_hashes_from_text,
    messages_to_prefix_text,
)
from telemetry.schemas import PrefixCacheTelemetry, RequestComputedTelemetry

_TOKEN_RE = re.compile(r'"prompt_tokens"\s*:\s*(\d+)|"prompt_n"\s*:\s*(\d+)')


@dataclass(frozen=True)
class PromptTelemetryContext:
    """Prompt details needed to update request-computed telemetry after generation."""

    text: str
    estimated_tokens: int
    prefix_hashes: list[str]


def messages_to_prompt_text(messages: object) -> str:
    """Extract canonical prompt text from OpenAI-style chat messages."""
    return messages_to_prefix_text(messages)


def prompt_context_from_messages(
    messages: object,
    prefix_config: PrefixHashConfig | None = None,
) -> PromptTelemetryContext:
    """Build prompt telemetry context from an OpenAI-style message list."""
    text = messages_to_prompt_text(messages)
    prefix_result = compute_prefix_hashes_from_text(text, prefix_config)
    return PromptTelemetryContext(
        text=text,
        estimated_tokens=prefix_result.estimated_tokens,
        prefix_hashes=prefix_result.block_hashes,
    )


def ema_update(previous: float, observed: float, alpha: float = 0.15) -> float:
    """Update an exponential moving average."""
    return alpha * observed + (1.0 - alpha) * previous


def extract_usage_prompt_tokens(json_line: str) -> int | None:
    """Best-effort parse of prompt_tokens from an SSE JSON fragment."""
    try:
        obj = json.loads(json_line)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        usage = obj.get("usage")
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            if isinstance(pt, int):
                return pt
        timings = obj.get("timings")
        if isinstance(timings, dict):
            pt = timings.get("prompt_n")
            if isinstance(pt, int):
                return pt

    for match in _TOKEN_RE.finditer(json_line):
        if match.group(1) is not None or match.group(2) is not None:
            return int(match.group(1) or match.group(2))
    return None


class RequestTelemetryTracker:
    """Tracks telemetry that Relay computes around completed generation calls."""

    def __init__(self, *, prefix_config: PrefixHashConfig | None = None) -> None:
        self._prefix_config = prefix_config or PrefixHashConfig.from_env()
        self._sprefill_ema: float = 0.0
        self._prefix_hashes = PrefixHashLRU(self._prefix_config.max_blocks)

    def context_from_messages(self, messages: object) -> PromptTelemetryContext:
        """Build the prompt context needed before the request is sent."""
        return prompt_context_from_messages(messages, self._prefix_config)

    def observe_completion(
        self,
        *,
        first_token_seconds: float | None,
        prompt_tokens: int,
        prefix_hashes: Sequence[str],
    ) -> None:
        """Update request-computed telemetry from one completed generation."""
        if first_token_seconds is not None and prompt_tokens > 0:
            prefill_s = max(first_token_seconds, 1e-6)
            observed_prefill = prompt_tokens / prefill_s
            self._sprefill_ema = (
                ema_update(self._sprefill_ema, observed_prefill)
                if self._sprefill_ema > 0
                else observed_prefill
            )

        self._prefix_hashes.observe(prefix_hashes)

    def snapshot(self) -> RequestComputedTelemetry:
        """Return current request-computed telemetry."""
        return RequestComputedTelemetry(
            prefix_cache=PrefixCacheTelemetry.from_config(
                self._prefix_config,
                self._prefix_hashes.snapshot(),
            ),
            sprefill_tokens_per_sec=self._sprefill_ema,
        )
