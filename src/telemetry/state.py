"""Long-lived mutable telemetry state owned by a worker daemon."""

from __future__ import annotations

import asyncio

from telemetry.request_metrics import (
    PromptTelemetryContext,
    RequestTelemetryTracker,
    prompt_context_from_messages,
)
from telemetry.schemas import (
    EngineReportedTelemetry,
    RequestComputedTelemetry,
    SystemTelemetry,
    Telemetry,
)


class WorkerTelemetryState:
    """Mutable worker telemetry assembled from engine, request, system, and memory sources."""

    def __init__(self) -> None:
        self._engine = EngineReportedTelemetry(qw=0)
        self._system = SystemTelemetry(jw=0.0, theta_w=0)
        self._memory_mw: float = 0.0
        self._request_tracker = RequestTelemetryTracker()
        self._lock = asyncio.Lock()

    def context_from_request(self, request: dict[str, object]) -> PromptTelemetryContext:
        """Build prompt telemetry context before a request is sent to the engine."""
        return prompt_context_from_messages(request.get("messages"))

    async def update_engine(self, telemetry: EngineReportedTelemetry) -> None:
        """Replace engine-reported telemetry with the latest backend snapshot."""
        async with self._lock:
            self._engine = telemetry

    async def update_system(self, telemetry: SystemTelemetry) -> None:
        """Replace worker-level system telemetry with the latest collector snapshot."""
        async with self._lock:
            self._system = telemetry

    async def update_memory_pressure(self, mw: float) -> None:
        """Replace the system-level memory pressure value."""
        async with self._lock:
            self._memory_mw = max(0.0, min(1.0, float(mw)))

    async def observe_request_completion(
        self,
        *,
        prompt_context: PromptTelemetryContext,
        first_token_seconds: float | None,
        prompt_tokens: int,
    ) -> None:
        """Update request-computed telemetry from one completed generation."""
        async with self._lock:
            self._request_tracker.observe_completion(
                first_token_seconds=first_token_seconds,
                prompt_tokens=prompt_tokens,
                prefix_hashes=prompt_context.prefix_hashes,
            )

    async def snapshot(self) -> Telemetry:
        """Return a coordinator-facing telemetry snapshot."""
        async with self._lock:
            request = self._request_tracker.snapshot()
            return _combine(self._engine, request, self._system, self._memory_mw)


def _combine(
    engine: EngineReportedTelemetry,
    request: RequestComputedTelemetry,
    system: SystemTelemetry,
    memory_mw: float,
) -> Telemetry:
    return Telemetry.from_parts(
        engine=engine, request=request, system=system, memory_mw=memory_mw
    )
