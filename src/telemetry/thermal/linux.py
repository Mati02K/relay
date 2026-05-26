"""Linux CPU thermal-throttle collector.

Reads the monotonic per-core and per-package throttle event counters exposed
under ``/sys/devices/system/cpu/cpu*/thermal_throttle/``. The kernel
documentation guarantees these counters are read-only and only increase, so
``theta_w`` for a sample window becomes ``1`` if and only if the total grew
since the previous read.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

DEFAULT_SYSFS_ROOT = Path("/sys/devices/system/cpu")
_COUNTER_FILES = ("core_throttle_count", "package_throttle_count")


class LinuxCpuThrottleCollector:
    """Detects CPU thermal throttling on Linux via sysfs throttle counters."""

    name = "linux-cpu-throttle"

    def __init__(self, sysfs_root: Path = DEFAULT_SYSFS_ROOT) -> None:
        self._sysfs_root = sysfs_root
        self._previous_total: int | None = None

    async def sample(self) -> int:
        """Return 1 if the cumulative throttle count grew since the last sample."""
        total = await asyncio.to_thread(_read_total_throttle_count, self._sysfs_root)
        if total is None:
            return 0
        previous = self._previous_total
        self._previous_total = total
        if previous is None:
            return 0
        return 1 if total > previous else 0

    async def close(self) -> None:
        """No resources to release."""
        return None


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


def _read_counter(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError):
        return None
    except (OSError, ValueError) as e:
        logger.debug("Throttle counter unreadable | path={} error={}", path, e)
        return None
