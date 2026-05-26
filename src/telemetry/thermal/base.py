"""Thermal collector protocol, null implementation, and OR aggregator.

A ``ThermalCollector`` produces a binary ``theta_w`` signal where ``1`` means
the device is currently throttling (or imminently will) and ``0`` means normal.
The worker daemon wires the per-platform collectors into a single aggregator
and writes the OR-reduced value into ``SystemTelemetry.theta_w``.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger


class ThermalCollector(Protocol):
    """One thermal source. Returns 1 when throttling, 0 otherwise."""

    name: str

    async def sample(self) -> int:
        """Return 0 or 1 for the latest observation."""
        ...

    async def close(self) -> None:
        """Release any resources held by the collector."""
        ...


class NullThermalCollector:
    """Fallback collector for platforms or hardware we cannot read."""

    name = "null"

    async def sample(self) -> int:
        """Always return 0."""
        return 0

    async def close(self) -> None:
        """No-op."""
        return None


class ThermalAggregator:
    """Composes multiple collectors into a single ``theta_w`` value.

    The aggregator is intentionally defensive: a collector raising an exception
    is treated as ``0`` for that tick, so a broken sensor never causes the
    worker to advertise itself as throttled.
    """

    def __init__(self, collectors: list[ThermalCollector]) -> None:
        self._collectors = list(collectors)

    @property
    def collector_names(self) -> list[str]:
        """Return the names of the underlying collectors for diagnostics."""
        return [collector.name for collector in self._collectors]

    async def sample(self) -> int:
        """Return ``1`` if any collector reports throttling, else ``0``."""
        throttled = 0
        for collector in self._collectors:
            try:
                value = await collector.sample()
            except Exception:
                logger.exception("Thermal collector raised | name={}", collector.name)
                continue
            if value:
                throttled = 1
        return throttled

    async def close(self) -> None:
        """Close every underlying collector, ignoring individual failures."""
        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.exception("Thermal collector close failed | name={}", collector.name)
