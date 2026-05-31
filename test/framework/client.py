"""Async Relay client that captures routing headers and TTFT from SSE streams."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class RoutingRecord:
    """All routing and latency data captured from a single request."""

    timestamp: float
    scenario: str
    phase: str
    worker: str
    model: str
    ttft_ms: float
    total_ms: float
    cost: float
    overlap: float
    complexity: float
    prompt_tokens: int
    matched_tokens: int
    attempts: int
    conversation_id: str = ""
    turn: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "phase": self.phase,
            "worker": self.worker,
            "model": self.model,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "cost": self.cost,
            "overlap": self.overlap,
            "complexity": self.complexity,
            "prompt_tokens": self.prompt_tokens,
            "matched_tokens": self.matched_tokens,
            "attempts": self.attempts,
            "conversation_id": self.conversation_id,
            "turn": self.turn,
            "error": self.error,
        }


class RelayClient:
    """Async OpenAI-compatible client targeting the Relay coordinator.

    Streams SSE responses, records TTFT on the first content delta, and
    captures all ``X-Relay-*`` routing headers.
    """

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return a shared async client, recreating it if it was closed.

        Reusing one client (and its connection pool) across requests avoids the
        cost of opening a fresh TCP/TLS connection for every completion.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the shared async client if one is open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 64,
        scenario: str = "unknown",
        phase: str = "default",
        conversation_id: str = "",
        turn: int = 0,
    ) -> RoutingRecord:
        """Send one chat completion request and return a :class:`RoutingRecord`.

        TTFT is measured from the moment the TCP connection opens until the
        first SSE chunk that carries a non-empty ``choices[0].delta.content``.
        Total time is measured until the ``[DONE]`` sentinel arrives or the
        stream closes.
        """
        body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "stream": True}
        if model:
            body["model"] = model

        url = f"{self.base_url}/v1/chat/completions"
        t_start = time.perf_counter()
        ttft_ms = 0.0
        first_content_seen = False
        worker = ""
        model_id = ""
        cost = 0.0
        overlap = 0.0
        complexity = 0.0
        prompt_tokens = 0
        matched_tokens = 0
        attempts = 1
        error: str | None = None

        try:
            client = self._get_client()
            async with client.stream("POST", url, json=body) as response:
                if response.status_code >= 400:
                    body_text = (await response.aread()).decode("utf-8", errors="replace")
                    error = f"HTTP {response.status_code}: {body_text[:200]}"
                else:
                    worker = response.headers.get("x-relay-worker", "")
                    model_id = response.headers.get("x-relay-model", "")
                    cost = _parse_float(response.headers.get("x-relay-cost", "0"))
                    overlap = _parse_float(response.headers.get("x-relay-overlap", "0"))
                    complexity = _parse_float(response.headers.get("x-relay-complexity", "0"))
                    prompt_tokens = int(
                        _parse_float(response.headers.get("x-relay-prompt-tokens", "0"))
                    )
                    matched_tokens = int(
                        _parse_float(response.headers.get("x-relay-matched-tokens", "0"))
                    )
                    attempts = int(
                        _parse_float(response.headers.get("x-relay-attempts", "1"))
                    )

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload == "[DONE]":
                            break
                        if not first_content_seen and _has_content_delta(payload):
                            ttft_ms = (time.perf_counter() - t_start) * 1000.0
                            first_content_seen = True
        except Exception as exc:
            error = str(exc)[:300]

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return RoutingRecord(
            timestamp=t_start,
            scenario=scenario,
            phase=phase,
            worker=worker,
            model=model_id,
            ttft_ms=ttft_ms if first_content_seen else total_ms,
            total_ms=total_ms,
            cost=cost,
            overlap=overlap,
            complexity=complexity,
            prompt_tokens=prompt_tokens,
            matched_tokens=matched_tokens,
            attempts=attempts,
            conversation_id=conversation_id,
            turn=turn,
            error=error,
        )


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _has_content_delta(payload: str) -> bool:
    try:
        obj = json.loads(payload)
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        delta = choices[0].get("delta") or {}
        return bool(delta.get("content") or delta.get("reasoning_content"))
    except (json.JSONDecodeError, AttributeError, KeyError):
        return False
