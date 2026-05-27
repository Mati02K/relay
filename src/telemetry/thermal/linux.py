"""Linux CPU thermal collector.

Combines two Linux-native sources:

* ``/sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count`` for
  monotonic CPU throttle events.
* ``/sys/class/thermal/thermal_zone*`` and ``/sys/class/hwmon`` temperatures
  with their published trip/critical limits when available.

The collector returns normalized CPU pressure and raw details so operators can
verify the value from a shell without relying on hidden heuristics.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from telemetry.thermal.base import (
    ThermalSourceSample,
    pressure_from_temperature,
    pressure_from_throttle_rate,
    pressure_to_state,
)

DEFAULT_SYSFS_ROOT = Path("/sys/devices/system/cpu")
DEFAULT_THERMAL_ROOT = Path("/sys/class/thermal")
DEFAULT_HWMON_ROOT = Path("/sys/class/hwmon")
_COUNTER_FILES = ("core_throttle_count", "package_throttle_count")
_CPU_LABEL_HINTS = ("cpu", "package", "core", "tctl", "tdie", "k10temp", "x86_pkg_temp")


class LinuxCpuThrottleCollector:
    """Detects CPU thermal pressure on Linux via sysfs counters and temperatures."""

    name = "linux-cpu-throttle"

    def __init__(
        self,
        sysfs_root: Path = DEFAULT_SYSFS_ROOT,
        thermal_root: Path = DEFAULT_THERMAL_ROOT,
        hwmon_root: Path = DEFAULT_HWMON_ROOT,
        clock: Callable[[], float] = time.monotonic,
        throttle_ema_alpha: float = 0.30,
    ) -> None:
        self._sysfs_root = sysfs_root
        self._thermal_root = thermal_root
        self._hwmon_root = hwmon_root
        self._clock = clock
        self._throttle_ema_alpha = max(0.0, min(1.0, throttle_ema_alpha))
        self._previous_total: int | None = None
        self._previous_sample_time: float | None = None
        self._throttle_pressure_ema = 0.0

    async def sample(self) -> list[ThermalSourceSample]:
        """Return one normalized CPU pressure sample."""
        sampled_at = self._clock()
        total, temp = await asyncio.to_thread(
            _read_linux_cpu_thermal,
            self._sysfs_root,
            self._thermal_root,
            self._hwmon_root,
        )
        temp_pressure = pressure_from_temperature(temp.temperature_c, temp.limit_c)
        if total is None:
            logger.debug(
                "Linux CPU thermal sample | root={} countersReadable=false tempC={} "
                "limitC={} tempPressure={:.3f}",
                self._sysfs_root,
                temp.temperature_c,
                temp.limit_c,
                temp_pressure,
            )
            pressure = temp_pressure
            delta = None
            sample_window_seconds = None
            throttle_rate = None
            throttle_pressure_raw = 0.0
            throttle_pressure = 0.0
            throttle_active = None
        else:
            previous = self._previous_total
            previous_sample_time = self._previous_sample_time
            self._previous_total = total
            self._previous_sample_time = sampled_at
            if previous is None:
                delta = 0
            else:
                delta = max(0, total - previous)
            if previous_sample_time is None:
                sample_window_seconds = None
                throttle_rate = 0.0
            else:
                sample_window_seconds = max(1e-6, sampled_at - previous_sample_time)
                throttle_rate = delta / sample_window_seconds
            throttle_pressure_raw = pressure_from_throttle_rate(throttle_rate)
            throttle_pressure = self._smoothed_throttle_pressure(throttle_pressure_raw)
            pressure = max(temp_pressure, throttle_pressure)
            throttle_active = delta > 0
            logger.debug(
                "Linux CPU thermal sample | total={} previous={} delta={} tempC={} "
                "limitC={} tempPressure={:.3f} windowSec={} throttleRate={} "
                "rawThrottlePressure={:.3f} smoothedThrottlePressure={:.3f} "
                "pressure={:.3f}",
                total,
                previous,
                delta,
                temp.temperature_c,
                temp.limit_c,
                temp_pressure,
                sample_window_seconds,
                throttle_rate,
                throttle_pressure_raw,
                throttle_pressure,
                pressure,
            )
        return [
            ThermalSourceSample(
                source=self.name,
                device_type="cpu",
                pressure=pressure,
                state=pressure_to_state(pressure),
                temperature_c=temp.temperature_c,
                limit_c=temp.limit_c,
                throttle_active=throttle_active,
                details={
                    "throttle_total": total,
                    "throttle_delta": delta,
                    "sample_window_seconds": sample_window_seconds,
                    "throttle_rate_per_sec": throttle_rate,
                    "throttle_pressure_raw": throttle_pressure_raw,
                    "throttle_pressure_smoothed": throttle_pressure,
                    "temperature_source": temp.source,
                    "temperature_label": temp.label,
                },
            )
        ]

    async def close(self) -> None:
        """No resources to release."""
        return None

    def _smoothed_throttle_pressure(self, raw_pressure: float) -> float:
        alpha = self._throttle_ema_alpha
        self._throttle_pressure_ema = (
            self._throttle_pressure_ema * (1.0 - alpha) + raw_pressure * alpha
        )
        return self._throttle_pressure_ema


def _read_total_throttle_count(sysfs_root: Path) -> int | None:
    """Sum every available throttle counter under ``sysfs_root``.

    Returns ``None`` if no readable counters exist so the caller can keep
    treating the device as untested rather than artificially healthy.
    """
    if not sysfs_root.is_dir():
        return None
    total = 0
    found = False
    for cpu_dir in sysfs_root.glob("cpu[0-9]*"):
        thermal_dir = cpu_dir / "thermal_throttle"
        if not thermal_dir.is_dir():
            continue
        for counter_name in _COUNTER_FILES:
            counter_path = thermal_dir / counter_name
            value = _read_counter(counter_path)
            if value is None:
                continue
            total += value
            found = True
    return total if found else None


class _TemperatureReading:
    def __init__(
        self,
        *,
        temperature_c: float | None = None,
        limit_c: float | None = None,
        source: str | None = None,
        label: str | None = None,
    ) -> None:
        self.temperature_c = temperature_c
        self.limit_c = limit_c
        self.source = source
        self.label = label


def _read_linux_cpu_thermal(
    cpu_root: Path,
    thermal_root: Path,
    hwmon_root: Path,
) -> tuple[int | None, _TemperatureReading]:
    return (
        _read_total_throttle_count(cpu_root),
        _max_temperature_reading(
            _read_thermal_zone_temperatures(thermal_root)
            + _read_hwmon_cpu_temperatures(hwmon_root)
        ),
    )


def _read_thermal_zone_temperatures(root: Path) -> list[_TemperatureReading]:
    readings: list[_TemperatureReading] = []
    if not root.is_dir():
        return readings
    for zone in root.glob("thermal_zone*"):
        temp_c = _read_millicelsius(zone / "temp")
        if temp_c is None:
            continue
        zone_type = _read_text(zone / "type") or zone.name
        limit_c = _read_thermal_zone_limit(zone)
        readings.append(
            _TemperatureReading(
                temperature_c=temp_c,
                limit_c=limit_c,
                source=str(zone),
                label=zone_type,
            )
        )
    return readings


def _read_thermal_zone_limit(zone: Path) -> float | None:
    limits: list[float] = []
    for trip_path in zone.glob("trip_point_*_temp"):
        value = _read_millicelsius(trip_path)
        if value is not None and value > 0:
            limits.append(value)
    if not limits:
        return None
    return max(limits)


def _read_hwmon_cpu_temperatures(root: Path) -> list[_TemperatureReading]:
    readings: list[_TemperatureReading] = []
    if not root.is_dir():
        return readings
    for hwmon in root.glob("hwmon*"):
        chip_name = _read_text(hwmon / "name") or hwmon.name
        for temp_input in hwmon.glob("temp*_input"):
            stem = temp_input.stem.removesuffix("_input")
            label = _read_text(hwmon / f"{stem}_label") or chip_name
            combined = f"{chip_name} {label}".lower()
            if not any(hint in combined for hint in _CPU_LABEL_HINTS):
                continue
            temp_c = _read_millicelsius(temp_input)
            if temp_c is None:
                continue
            limit_c = _read_millicelsius(hwmon / f"{stem}_crit") or _read_millicelsius(
                hwmon / f"{stem}_max"
            )
            readings.append(
                _TemperatureReading(
                    temperature_c=temp_c,
                    limit_c=limit_c,
                    source=str(temp_input),
                    label=label,
                )
            )
    return readings


def _max_temperature_reading(readings: list[_TemperatureReading]) -> _TemperatureReading:
    if not readings:
        return _TemperatureReading()
    return max(readings, key=lambda reading: reading.temperature_c or -273.15)


def _read_millicelsius(path: Path) -> float | None:
    try:
        raw = float(path.read_text().strip())
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None
    return raw / 1000.0 if abs(raw) > 1000 else raw


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_counter(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError):
        return None
    except (OSError, ValueError) as e:
        logger.debug("Throttle counter unreadable | path={} error={}", path, e)
        return None
