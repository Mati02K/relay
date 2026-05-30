"""System RAM pressure collector for Linux CPU-only inference workers.

Reads ``/proc/meminfo`` for ``MemTotal`` and ``MemAvailable``.  The kernel
updates these atomically so no locking is required.

``MemAvailable`` (added in kernel 3.14) is the right denominator for "how
much memory can new allocations use": it includes free pages, reclaimable
page-cache, and reclaimable slab — not just the raw ``MemFree`` which omits
pages the kernel would release under pressure.

  pressure = 1.0 - MemAvailable / MemTotal
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from telemetry.memory.base import MemorySnapshot

DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")


class LinuxRamCollector:
    """Measures system RAM pressure from ``/proc/meminfo``."""

    name = "linux-ram"

    def __init__(self, meminfo_path: Path = DEFAULT_MEMINFO_PATH) -> None:
        self._meminfo_path = meminfo_path

    async def sample(self) -> MemorySnapshot | None:
        """Return current RAM pressure or ``None`` if unreadable."""
        return await asyncio.to_thread(self._sample_sync)

    def _sample_sync(self) -> MemorySnapshot | None:
        total_kb, available_kb = _read_meminfo(self._meminfo_path)
        if total_kb is None or total_kb <= 0:
            logger.debug("Linux RAM: MemTotal unreadable or zero | path={}", self._meminfo_path)
            return None
        if available_kb is None:
            logger.debug(
                "Linux RAM: MemAvailable missing (kernel < 3.14?) | path={}", self._meminfo_path
            )
            return None
        pressure = 1.0 - available_kb / total_kb
        logger.debug(
            "Linux RAM sample | totalKiB={} availableKiB={} pressure={:.3f}",
            total_kb,
            available_kb,
            pressure,
        )
        return MemorySnapshot(
            source=self.name,
            device_type="cpu",
            pressure=pressure,
            used_bytes=(total_kb - available_kb) * 1024,
            total_bytes=total_kb * 1024,
        )

    async def close(self) -> None:
        """No persistent resources."""
        return None


def _read_meminfo(path: Path) -> tuple[int | None, int | None]:
    """Return ``(MemTotal_kB, MemAvailable_kB)`` from a ``/proc/meminfo`` file."""
    total: int | None = None
    available: int | None = None
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].rstrip(":")
            try:
                value = int(parts[1])
            except ValueError:
                continue
            if key == "MemTotal":
                total = value
            elif key == "MemAvailable":
                available = value
            if total is not None and available is not None:
                break
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.debug("Linux RAM meminfo read failed | path={} error={}", path, exc)
    return total, available
