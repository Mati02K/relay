"""AMD GPU thermal collector using Linux ``amdgpu`` hwmon sysfs.

ROCm/AMD-SMI availability varies across consumer machines, but the amdgpu
kernel driver commonly exposes GPU temperatures through ``/sys/class/hwmon``.
This collector uses those sysfs readings without adding a Python dependency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from telemetry.thermal.base import (
    ThermalSourceSample,
    pressure_from_temperature,
    pressure_to_state,
)

DEFAULT_HWMON_ROOT = Path("/sys/class/hwmon")


class AmdGpuSysfsCollector:
    """Reads AMD GPU edge/junction temperatures from Linux hwmon."""

    name = "amd-gpu-sysfs"

    def __init__(self, hwmon_root: Path = DEFAULT_HWMON_ROOT) -> None:
        self._hwmon_root = hwmon_root

    @classmethod
    def try_create(cls, hwmon_root: Path = DEFAULT_HWMON_ROOT) -> AmdGpuSysfsCollector | None:
        """Return a collector when an ``amdgpu`` hwmon device is present."""
        if not hwmon_root.is_dir():
            return None
        for hwmon in hwmon_root.glob("hwmon*"):
            if (_read_text(hwmon / "name") or "").lower() == "amdgpu":
                return cls(hwmon_root=hwmon_root)
        return None

    async def sample(self) -> list[ThermalSourceSample]:
        """Return one sample per readable AMD GPU temperature input."""
        return await asyncio.to_thread(self._sample_sync)

    def _sample_sync(self) -> list[ThermalSourceSample]:
        samples: list[ThermalSourceSample] = []
        for hwmon in self._hwmon_root.glob("hwmon*"):
            if (_read_text(hwmon / "name") or "").lower() != "amdgpu":
                continue
            for temp_input in hwmon.glob("temp*_input"):
                stem = temp_input.stem.removesuffix("_input")
                temp_c = _read_millicelsius(temp_input)
                if temp_c is None:
                    continue
                label = _read_text(hwmon / f"{stem}_label") or stem
                limit_c = _read_millicelsius(hwmon / f"{stem}_crit") or _read_millicelsius(
                    hwmon / f"{stem}_max"
                )
                pressure = pressure_from_temperature(temp_c, limit_c or 100.0)
                logger.debug(
                    "AMD GPU sysfs sample | source={} label={} tempC={} limitC={} "
                    "pressure={:.3f}",
                    temp_input,
                    label,
                    temp_c,
                    limit_c,
                    pressure,
                )
                samples.append(
                    ThermalSourceSample(
                        source=self.name,
                        device_type="gpu",
                        pressure=pressure,
                        state=pressure_to_state(pressure),
                        confidence=0.75,
                        temperature_c=temp_c,
                        limit_c=limit_c or 100.0,
                        throttle_active=pressure >= 0.65,
                        details={"path": str(temp_input), "label": label},
                    )
                )
        return samples

    async def close(self) -> None:
        """No persistent resources."""
        return None


def _read_millicelsius(path: Path) -> float | None:
    try:
        raw = float(path.read_text().strip())
    except (FileNotFoundError, PermissionError, OSError, ValueError) as e:
        logger.debug("AMD GPU temperature unreadable | path={} error={}", path, e)
        return None
    return raw / 1000.0 if abs(raw) > 1000 else raw


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
