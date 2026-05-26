"""NVIDIA GPU thermal-throttle collector backed by NVML.

The ``pynvml`` / ``nvidia-ml-py`` package is imported softly: if it is not
installed or no NVIDIA driver is present, ``try_create`` returns ``None`` and
the factory falls back to skipping NVIDIA monitoring on this worker.

The collector reads ``nvmlDeviceGetCurrentClocksEventReasons`` (formerly
``ClocksThrottleReasons``). The returned bitmask is checked against the
thermal-related flags only — power capping and idle downclocking are not
reported as throttling because they are not heat-driven.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

# Bit values from nvml.h. Hard-coded so the file can be imported even when
# pynvml is missing; the actual call uses pynvml's symbols when available.
_REASON_HW_SLOWDOWN = 0x0000000000000008
_REASON_SW_THERMAL_SLOWDOWN = 0x0000000000000020
_REASON_HW_THERMAL_SLOWDOWN = 0x0000000000000040

_THERMAL_REASON_MASK = (
    _REASON_HW_SLOWDOWN | _REASON_SW_THERMAL_SLOWDOWN | _REASON_HW_THERMAL_SLOWDOWN
)


def is_thermal_throttle(reason_bits: int) -> int:
    """Return 1 if the bitmask carries any thermal slowdown flag."""
    return 1 if (reason_bits & _THERMAL_REASON_MASK) else 0


class NvidiaNvmlCollector:
    """Polls NVML for thermal-slowdown bits across all visible NVIDIA GPUs."""

    name = "nvidia-nvml"

    def __init__(self, pynvml: Any, handles: list[Any]) -> None:
        self._pynvml = pynvml
        self._handles = handles

    @classmethod
    def try_create(cls) -> NvidiaNvmlCollector | None:
        """Initialize NVML and return a collector, or ``None`` if unavailable."""
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError:
            return None

        try:
            pynvml.nvmlInit()
        except Exception as e:
            logger.debug("NVML init failed | error={}", e)
            return None

        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            logger.debug("NVML device count failed | error={}", e)
            _safe_shutdown(pynvml)
            return None

        handles = []
        for index in range(count):
            try:
                handles.append(pynvml.nvmlDeviceGetHandleByIndex(index))
            except Exception as e:
                logger.debug("NVML handle failed | index={} error={}", index, e)
        if not handles:
            _safe_shutdown(pynvml)
            return None
        logger.info("NVIDIA thermal collector ready | deviceCount={}", len(handles))
        return cls(pynvml, handles)

    async def sample(self) -> int:
        """Query throttle reasons for every GPU; return 1 on any thermal flag."""
        return await asyncio.to_thread(self._sample_sync)

    def _sample_sync(self) -> int:
        for handle in self._handles:
            reasons = self._read_reasons(handle)
            if reasons is None:
                continue
            if is_thermal_throttle(reasons):
                return 1
        return 0

    def _read_reasons(self, handle: Any) -> int | None:
        getter = getattr(
            self._pynvml,
            "nvmlDeviceGetCurrentClocksEventReasons",
            None,
        ) or getattr(
            self._pynvml,
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            None,
        )
        if getter is None:
            return None
        try:
            return int(getter(handle))
        except Exception as e:
            logger.debug("NVML throttle read failed | error={}", e)
            return None

    async def close(self) -> None:
        """Shut down NVML."""
        await asyncio.to_thread(_safe_shutdown, self._pynvml)


def _safe_shutdown(pynvml: Any) -> None:
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
