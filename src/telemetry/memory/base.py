"""Memory pressure collector protocol, null implementation, and aggregator.

Collectors return a normalized pressure sample in ``[0, 1]``:
  ``0.0`` = device memory is completely free / idle
  ``1.0`` = device memory is fully exhausted

Unlike the KV-cache ratio reported by llama-server (which only sees memory
allocated by the engine itself), these collectors read from the OS or GPU
driver and therefore capture all consumers on the same device — other
processes, games, CUDA runtime overhead, etc.

Aggregation mirrors the thermal module's two-signal design:

  GPU primary (model on GPU):
      mw = max(vram_pressure, ram_pressure × ram_weight)

      ``ram_weight`` escalates as RAM fills up — a near-full RAM pool
      causes OS swapping, stalls PCIe DMA transfers, and squeezes the
      llama-server process heap, all of which degrade GPU inference even
      though VRAM itself may look healthy.  Weight tiers:

        ram_pressure ≤ 0.65  →  0.25  (plenty free, GPU fully dominant)
        ram_pressure  0.65–0.85 → 0.50  (getting tight, meaningful risk)
        ram_pressure > 0.85  →  0.75  (swap imminent, nearly co-primary)

  CPU primary (model on CPU, no GPU):
      mw = ram_pressure  (GPU absent — RAM is the only memory signal)

  Unknown / unified (macOS):
      mw = max(all source pressures)

After aggregation the raw value is passed through an asymmetric EMA so
sudden allocation spikes (e.g. a game loading a new level) are reflected
immediately while brief transient dips do not drop the apparent pressure
at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

# Asymmetric EMA decay rate applied when pressure is *falling*.
# 0.4 means new pressure contributes 40 % to the smoothed value each sample.
# At a 5-second poll interval a spike decays to ~10 % of its peak in ~25 s.
MEMORY_EMA_ALPHA: float = 0.4

# RAM pressure tiers used to weight RAM contribution when GPU is the primary
# compute device.  Unlike thermal headroom (which is in °C below the trip
# point), memory pressure is already a normalized ratio in [0, 1] so the
# tiers are defined directly on the pressure value.
RAM_PRESSURE_MID: float = 0.65   # above this RAM is getting tight
RAM_PRESSURE_HIGH: float = 0.85  # above this swap is imminent
RAM_WEIGHT_DEFAULT: float = 0.25  # RAM well within limits — GPU dominant
RAM_WEIGHT_MID: float = 0.50      # moderate RAM pressure — meaningful risk
RAM_WEIGHT_HIGH: float = 0.75     # near-swap — RAM nearly co-primary


@dataclass(frozen=True)
class MemorySnapshot:
    """One normalized memory pressure reading from one device or source."""

    source: str
    device_type: str
    pressure: float
    used_bytes: int | None = None
    total_bytes: int | None = None
    details: dict[str, object] = field(default_factory=dict)

    def normalized(self) -> MemorySnapshot:
        """Clamp ``pressure`` to ``[0, 1]``."""
        return MemorySnapshot(
            source=self.source,
            device_type=self.device_type,
            pressure=clamp01(self.pressure),
            used_bytes=self.used_bytes,
            total_bytes=self.total_bytes,
            details=dict(self.details),
        )


@dataclass(frozen=True)
class MemoryPressureSnapshot:
    """Aggregated memory pressure returned by :class:`MemoryAggregator`."""

    pressure: float
    raw_pressure: float
    gpu_pressure: float
    cpu_pressure: float
    ram_weight: float
    sources: list[MemorySnapshot]

    @property
    def mw(self) -> float:
        """Scheduler-facing memory pressure in ``[0, 1]``."""
        return self.pressure


class MemoryCollector(Protocol):
    """One memory source.  Returns a normalized pressure sample or ``None``."""

    name: str

    async def sample(self) -> MemorySnapshot | None:
        """Return the latest memory pressure observation, or ``None`` on error."""
        ...

    async def close(self) -> None:
        """Release any resources held by the collector."""
        ...


class NullMemoryCollector:
    """Fallback collector when no platform source is readable."""

    name = "null"

    async def sample(self) -> MemorySnapshot | None:
        return MemorySnapshot(
            source=self.name,
            device_type="unknown",
            pressure=0.0,
        )

    async def close(self) -> None:
        return None


class MemoryAggregator:
    """Combines multiple collectors into one ``mw`` pressure value.

    When ``primary_device="gpu"`` (model running on GPU) VRAM pressure is
    dominant but system RAM still contributes through a dynamic weight that
    escalates as RAM fills toward swap territory — mirroring the thermal
    module's two-signal design.

    The aggregated raw value is then smoothed with an asymmetric EMA so
    sudden allocation spikes are reflected immediately while brief dips do
    not drop the apparent pressure at once.
    """

    def __init__(
        self,
        collectors: list[MemoryCollector],
        *,
        primary_device: str = "unknown",
        ema_alpha: float = MEMORY_EMA_ALPHA,
    ) -> None:
        self._collectors = list(collectors)
        self._primary_device = primary_device
        self._ema_alpha = max(0.0, min(1.0, ema_alpha))
        self._ema: float = 0.0

    @property
    def collector_names(self) -> list[str]:
        return [c.name for c in self._collectors]

    async def sample(self) -> MemoryPressureSnapshot:
        """Return the latest aggregated memory pressure snapshot."""
        sources: list[MemorySnapshot] = []
        for collector in self._collectors:
            try:
                result = await collector.sample()
            except Exception:
                logger.exception("Memory collector raised | name={}", collector.name)
                continue
            if result is None:
                continue
            s = result.normalized()
            logger.debug(
                "Memory source sample | source={} device={} pressure={:.3f} "
                "usedBytes={} totalBytes={}",
                s.source,
                s.device_type,
                s.pressure,
                s.used_bytes,
                s.total_bytes,
            )
            sources.append(s)

        raw, gpu_p, cpu_p, rw = aggregate_memory_samples(
            sources, primary_device=self._primary_device
        )

        if raw > self._ema:
            self._ema = clamp01(raw)
        else:
            self._ema = clamp01(self._ema_alpha * raw + (1.0 - self._ema_alpha) * self._ema)

        logger.debug(
            "Memory aggregate | raw={:.3f} ema={:.3f} gpu={:.3f} cpu={:.3f} "
            "ramWeight={:.2f} primary={} collectors={}",
            raw,
            self._ema,
            gpu_p,
            cpu_p,
            rw,
            self._primary_device,
            self.collector_names,
        )
        return MemoryPressureSnapshot(
            pressure=self._ema,
            raw_pressure=raw,
            gpu_pressure=gpu_p,
            cpu_pressure=cpu_p,
            ram_weight=rw,
            sources=sources,
        )

    async def close(self) -> None:
        """Close every underlying collector, ignoring individual failures."""
        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.exception("Memory collector close failed | name={}", collector.name)


def _ram_weight(ram_pressure: float) -> float:
    """Return the RAM contribution coefficient when GPU is the primary device.

    RAM pressure is already a normalized ratio in ``[0, 1]`` so the tiers
    are defined directly on the value rather than on headroom from a fixed
    trip point (unlike thermal which uses degrees below the thermal limit).

    * ``≤ RAM_PRESSURE_MID``   → ``RAM_WEIGHT_DEFAULT`` (0.25) — GPU dominant
    * ``(MID, HIGH]``          → ``RAM_WEIGHT_MID``     (0.50) — notable risk
    * ``> RAM_PRESSURE_HIGH``  → ``RAM_WEIGHT_HIGH``    (0.75) — swap imminent
    """
    if ram_pressure > RAM_PRESSURE_HIGH:
        return RAM_WEIGHT_HIGH
    if ram_pressure > RAM_PRESSURE_MID:
        return RAM_WEIGHT_MID
    return RAM_WEIGHT_DEFAULT


def aggregate_memory_samples(
    sources: list[MemorySnapshot],
    *,
    primary_device: str = "unknown",
) -> tuple[float, float, float, float]:
    """Fuse source samples into ``(pressure, gpu_pressure, cpu_pressure, ram_weight)``.

    The returned ``pressure`` is the raw aggregate before EMA smoothing.
    ``gpu_pressure`` and ``cpu_pressure`` are per-pool maximums retained for
    diagnostics and dashboard display.  ``ram_weight`` is the coefficient
    applied to RAM in the GPU-primary case.
    """
    normalized = [s.normalized() for s in sources]

    gpu_pressure = max(
        (s.pressure for s in normalized if s.device_type == "gpu"),
        default=0.0,
    )
    cpu_pressure = max(
        (s.pressure for s in normalized if s.device_type == "cpu"),
        default=0.0,
    )
    has_gpu = any(s.device_type == "gpu" for s in normalized)
    has_cpu = any(s.device_type == "cpu" for s in normalized)

    if primary_device == "gpu" and has_gpu:
        rw = _ram_weight(cpu_pressure)
        pressure = max(gpu_pressure, cpu_pressure * rw)
    elif primary_device == "cpu" and has_cpu:
        rw = 1.0
        pressure = cpu_pressure
    else:
        rw = 1.0
        pressure = max((s.pressure for s in normalized), default=0.0)

    return clamp01(pressure), gpu_pressure, cpu_pressure, rw


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))
