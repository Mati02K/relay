"""Thermal collector protocol, null implementation, and pressure aggregator.

Collectors return normalized pressure samples instead of raw vendor values.
``0.0`` means normal, ``1.0`` means severe/critical thermal pressure. The
aggregator combines CPU, GPU, and system-level sources into one scheduler-facing
``theta_w`` value while retaining raw source details for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

THERMAL_STATE_NORMAL = "normal"
THERMAL_STATE_WARM = "warm"
THERMAL_STATE_THROTTLING = "throttling"
THERMAL_STATE_CRITICAL = "critical"
THERMAL_STATE_UNKNOWN = "unknown"

# Thermal headroom thresholds used to dynamically weight CPU pressure when the
# model runs on GPU.  "Headroom" = (limit_c - temperature_c).  As the CPU
# closes in on its thermal trip point it increasingly bottlenecks PCIe
# dispatch, tokenisation, and llama-server process scheduling even though the
# GPU is the primary compute device.
CPU_HEADROOM_CRITICAL_C: float = 5.0   # < 5 °C left → treat CPU as primary
CPU_HEADROOM_WARN_C: float = 15.0      # < 15 °C left → raise coefficient
CPU_WEIGHT_DEFAULT: float = 0.35       # safe headroom, GPU clearly dominant
CPU_WEIGHT_WARN: float = 0.60          # closing in on limit
CPU_WEIGHT_CRITICAL: float = 1.0       # about to throttle, CPU is a bottleneck
GPU_WEIGHT_CPU_PRIMARY: float = 0.50   # GPU contribution when CPU is primary


@dataclass(frozen=True)
class ThermalSourceSample:
    """One normalized thermal observation from one device or OS source."""

    source: str
    device_type: str
    pressure: float
    state: str
    temperature_c: float | None = None
    limit_c: float | None = None
    throttle_active: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> ThermalSourceSample:
        """Clamp numeric fields into stable ranges."""
        return ThermalSourceSample(
            source=self.source,
            device_type=self.device_type,
            pressure=clamp01(self.pressure),
            state=self.state,
            temperature_c=self.temperature_c,
            limit_c=self.limit_c,
            throttle_active=self.throttle_active,
            details=dict(self.details),
        )


@dataclass(frozen=True)
class ThermalSnapshot:
    """Aggregate thermal pressure sent to worker telemetry."""

    pressure: float
    state: str
    cpu_pressure: float
    gpu_pressure: float
    sources: list[ThermalSourceSample]
    cpu_weight: float = CPU_WEIGHT_DEFAULT

    @property
    def theta_w(self) -> float:
        """Scheduler-facing thermal pressure in ``[0, 1]``."""
        return self.pressure


class ThermalCollector(Protocol):
    """One thermal source. Returns normalized pressure samples."""

    name: str

    async def sample(self) -> list[ThermalSourceSample]:
        """Return zero or more observations for the latest sample window."""
        ...

    async def close(self) -> None:
        """Release any resources held by the collector."""
        ...


class NullThermalCollector:
    """Fallback collector for platforms or hardware we cannot read."""

    name = "null"

    async def sample(self) -> list[ThermalSourceSample]:
        """Return an explicit unknown/normal sample."""
        return [
            ThermalSourceSample(
                source=self.name,
                device_type="system",
                pressure=0.0,
                state=THERMAL_STATE_UNKNOWN,
            )
        ]

    async def close(self) -> None:
        """No-op."""
        return None


class ThermalAggregator:
    """Composes multiple collectors into one normalized ``theta_w`` pressure.

    The final pressure is the highest source pressure. This is conservative:
    any device in the inference path can make the worker a bad scheduling
    target. The scheduler still gets a gradient rather than a binary flag.
    """

    def __init__(
        self,
        collectors: list[ThermalCollector],
        *,
        primary_device: str = "unknown",
    ) -> None:
        self._collectors = list(collectors)
        self._primary_device = primary_device

    @property
    def collector_names(self) -> list[str]:
        """Return the names of the underlying collectors for diagnostics."""
        return [collector.name for collector in self._collectors]

    async def sample(self) -> ThermalSnapshot:
        """Return the latest aggregate thermal pressure."""
        samples: list[ThermalSourceSample] = []
        for collector in self._collectors:
            try:
                values = await collector.sample()
            except Exception:
                logger.exception("Thermal collector raised | name={}", collector.name)
                continue
            for value in values:
                sample = value.normalized()
                logger.debug(
                    "Thermal source sample | source={} device={} pressure={:.3f} "
                    "state={} tempC={} limitC={} throttled={} details={}",
                    sample.source,
                    sample.device_type,
                    sample.pressure,
                    sample.state,
                    sample.temperature_c,
                    sample.limit_c,
                    sample.throttle_active,
                    sample.details,
                )
                samples.append(sample)
        snapshot = aggregate_thermal_samples(samples, primary_device=self._primary_device)
        logger.debug(
            "Thermal aggregate sample | pressure={:.3f} state={} cpu={:.3f} "
            "gpu={:.3f} cpuWeight={:.2f} collectors={}",
            snapshot.pressure,
            snapshot.state,
            snapshot.cpu_pressure,
            snapshot.gpu_pressure,
            snapshot.cpu_weight,
            self.collector_names,
        )
        return snapshot

    async def close(self) -> None:
        """Close every underlying collector, ignoring individual failures."""
        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.exception("Thermal collector close failed | name={}", collector.name)


def _cpu_headroom_weight(cpu_samples: list[ThermalSourceSample]) -> float:
    """Return the CPU pressure coefficient for GPU-primary aggregation.

    When the model runs on GPU the CPU is not the bottleneck under normal
    conditions, so its thermal pressure is down-weighted.  However, a CPU
    that is about to throttle still degrades PCIe dispatch, tokenisation, and
    the llama-server process itself, so the coefficient escalates as the CPU
    closes in on its thermal trip point.

    The headroom tiers map to:
    * ``>= CPU_HEADROOM_WARN_C``     → ``CPU_WEIGHT_DEFAULT`` (0.35)
    * ``[CPU_HEADROOM_CRITICAL_C, CPU_HEADROOM_WARN_C)`` → ``CPU_WEIGHT_WARN`` (0.60)
    * ``< CPU_HEADROOM_CRITICAL_C``  → ``CPU_WEIGHT_CRITICAL`` (1.0)

    If no sample supplies temperature *and* limit data the default weight is
    used so the fallback is conservative rather than silently inflating cost.
    """
    min_headroom: float | None = None
    for sample in cpu_samples:
        if (
            sample.temperature_c is not None
            and sample.limit_c is not None
            and sample.limit_c > 0
        ):
            headroom = sample.limit_c - sample.temperature_c
            if min_headroom is None or headroom < min_headroom:
                min_headroom = headroom
    if min_headroom is None:
        return CPU_WEIGHT_DEFAULT
    if min_headroom < CPU_HEADROOM_CRITICAL_C:
        return CPU_WEIGHT_CRITICAL
    if min_headroom < CPU_HEADROOM_WARN_C:
        return CPU_WEIGHT_WARN
    return CPU_WEIGHT_DEFAULT


def aggregate_thermal_samples(
    samples: list[ThermalSourceSample],
    *,
    primary_device: str = "unknown",
) -> ThermalSnapshot:
    """Fuse source samples into one scheduler-facing thermal snapshot."""
    normalized = [sample.normalized() for sample in samples]
    if not normalized:
        return ThermalSnapshot(
            pressure=0.0,
            state=THERMAL_STATE_UNKNOWN,
            cpu_pressure=0.0,
            gpu_pressure=0.0,
            sources=[],
        )

    cpu_pressure = max(
        (sample.pressure for sample in normalized if sample.device_type == "cpu"),
        default=0.0,
    )
    gpu_pressure = max(
        (sample.pressure for sample in normalized if sample.device_type == "gpu"),
        default=0.0,
    )
    has_gpu = any(sample.device_type == "gpu" for sample in normalized)
    has_cpu = any(sample.device_type == "cpu" for sample in normalized)
    if primary_device == "gpu" and has_gpu:
        cpu_samples = [s for s in normalized if s.device_type == "cpu"]
        cpu_weight = _cpu_headroom_weight(cpu_samples)
        pressure = max(gpu_pressure, cpu_pressure * cpu_weight)
    elif primary_device == "cpu" and has_cpu:
        cpu_weight = 1.0
        pressure = max(cpu_pressure, gpu_pressure * GPU_WEIGHT_CPU_PRIMARY)
    else:
        cpu_weight = 1.0
        pressure = max((sample.pressure for sample in normalized), default=0.0)
    return ThermalSnapshot(
        pressure=pressure,
        state=pressure_to_state(pressure),
        cpu_pressure=cpu_pressure,
        gpu_pressure=gpu_pressure,
        sources=normalized,
        cpu_weight=cpu_weight,
    )


def pressure_to_state(pressure: float) -> str:
    """Map a normalized pressure value to a dashboard-friendly state."""
    value = clamp01(pressure)
    if value >= 0.9:
        return THERMAL_STATE_CRITICAL
    if value >= 0.65:
        return THERMAL_STATE_THROTTLING
    if value >= 0.35:
        return THERMAL_STATE_WARM
    return THERMAL_STATE_NORMAL


def pressure_from_temperature(
    temperature_c: float | None,
    limit_c: float | None,
    *,
    warm_margin_c: float = 10.0,
) -> float:
    """Convert temperature and limit into normalized pressure.

    Returns ``0`` unless ``temperature_c`` is within ``warm_margin_c`` of
    ``limit_c``. Inside the margin we use a quadratic curve so being mildly
    warm contributes only modest pressure — only readings close to the
    actual throttle limit ramp the score up.

    The defaults are deliberately conservative: a chip cruising at 85 °C
    against a 100 °C trip point reads ``0``, because it isn't being
    thermally clocked. Real throttling signals (explicit slowdown bits,
    sysfs counter deltas) dominate this term anyway.
    """
    if temperature_c is None or limit_c is None or limit_c <= 0:
        return 0.0
    start_c = max(0.0, limit_c - warm_margin_c)
    if temperature_c <= start_c:
        return 0.0
    ratio = clamp01((temperature_c - start_c) / max(1.0, limit_c - start_c))
    # Quadratic: mid-margin ≈ 0.25 instead of 0.5; only near-limit pushes high.
    return ratio * ratio


def pressure_from_throttle_rate(events_per_second: float) -> float:
    """Map Linux CPU throttle-event rate into normalized thermal pressure.

    Linux throttle counters can increment for very brief firmware events that
    are not visible as sustained user-facing slowdown. Treat low event rates as
    transient hints, and reserve throttling/critical pressure for sustained
    event rates.
    """
    if events_per_second <= 0:
        return 0.0
    if events_per_second < 2.0:
        return 0.15
    if events_per_second < 10.0:
        return 0.35
    if events_per_second < 50.0:
        return 0.65
    return 0.9


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))
