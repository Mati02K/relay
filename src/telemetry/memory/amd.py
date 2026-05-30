"""AMD GPU VRAM pressure collector via Linux DRM sysfs.

The ``amdgpu`` kernel driver exposes per-card VRAM statistics under
``/sys/class/drm/card*/device/``:

  ``mem_info_vram_used``   — bytes currently allocated
  ``mem_info_vram_total``  — total VRAM on the card

These files are readable without elevated privileges on any kernel with
``amdgpu`` loaded (5.x+).  No ROCm or AMD-SMI dependency is required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from telemetry.memory.base import MemorySnapshot

DEFAULT_DRM_ROOT = Path("/sys/class/drm")


class AmdVramCollector:
    """Reads per-card VRAM utilisation from the amdgpu DRM sysfs interface."""

    name = "amd-vram-sysfs"

    def __init__(self, drm_root: Path = DEFAULT_DRM_ROOT) -> None:
        self._drm_root = drm_root

    @classmethod
    def try_create(cls, drm_root: Path = DEFAULT_DRM_ROOT) -> AmdVramCollector | None:
        """Return a collector when at least one amdgpu VRAM file is present."""
        if not drm_root.is_dir():
            return None
        for card in drm_root.glob("card*"):
            if not (card / "device").is_dir():
                continue
            if (card / "device" / "mem_info_vram_total").exists():
                return cls(drm_root=drm_root)
        return None

    async def sample(self) -> MemorySnapshot | None:
        """Return the most-pressured AMD GPU sample across all amdgpu cards."""
        return await asyncio.to_thread(self._sample_sync)

    def _sample_sync(self) -> MemorySnapshot | None:
        worst: MemorySnapshot | None = None
        for card in sorted(self._drm_root.glob("card*")):
            device = card / "device"
            if not device.is_dir():
                continue
            used = _read_bytes(device / "mem_info_vram_used")
            total = _read_bytes(device / "mem_info_vram_total")
            if used is None or total is None or total <= 0:
                continue
            pressure = used / total
            logger.debug(
                "AMD VRAM sysfs | card={} usedBytes={} totalBytes={} pressure={:.3f}",
                card.name,
                used,
                total,
                pressure,
            )
            snapshot = MemorySnapshot(
                source=self.name,
                device_type="gpu",
                pressure=pressure,
                used_bytes=used,
                total_bytes=total,
                details={"card": card.name},
            )
            if worst is None or snapshot.pressure > worst.pressure:
                worst = snapshot
        return worst

    async def close(self) -> None:
        """No persistent resources."""
        return None


def _read_bytes(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        logger.debug("AMD VRAM sysfs read failed | path={} error={}", path, exc)
        return None
