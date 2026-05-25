"""Inference engine abstraction for Relay workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping

from pydantic import BaseModel

from telemetry.schemas import EngineReportedTelemetry


class EngineHealth(BaseModel):
    """Whether the Inference Engine is alive."""

    model_config = {"frozen": True}

    status: bool
    detail: str = ""
    engine: str = "unknown"


class InferenceEngine(ABC):
    """Data-plane inference backend."""

    @abstractmethod
    def generate(self, request: Mapping[str, object]) -> AsyncIterator[str]:
        """Stream one chat completion as newline-delimited SSE ``data:`` lines (OpenAI style)."""

    @abstractmethod
    async def health(self) -> EngineHealth:
        """Process + HTTP readiness for the active model server."""

    @abstractmethod
    async def stop(self) -> None:
        """Release inference backend resources."""

    @abstractmethod
    async def get_engine_telemetry(self) -> EngineReportedTelemetry:
        """Latest telemetry that can be reported directly by the inference backend."""
