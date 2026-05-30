"""Per-platform memory-collector factory.

Selection logic:

* **Linux GPU worker**  → NVIDIA NVML (VRAM), then AMD DRM sysfs (VRAM).
  Falls back to system RAM if neither GPU collector is readable — better to
  have a proxy signal than none at all.
* **Linux CPU worker** → system RAM via ``/proc/meminfo``.
* **macOS**           → unified memory via ``sysctl`` + ``vm_stat``.
* **Windows GPU**     → NVIDIA NVML (VRAM).
* **Anything else**   → ``NullMemoryCollector`` (pressure always 0.0).

When no platform source succeeds ``NullMemoryCollector`` is inserted so
:class:`MemoryAggregator` always has at least one collector.
"""

from __future__ import annotations

import platform

from loguru import logger

from telemetry.memory.amd import AmdVramCollector
from telemetry.memory.apple import MacOSMemoryCollector
from telemetry.memory.base import MemoryCollector, NullMemoryCollector
from telemetry.memory.linux_cpu import LinuxRamCollector
from telemetry.memory.nvidia import NvidiaVramCollector


def detect_memory_collectors(
    *,
    uses_gpu: bool,
    system_name: str | None = None,
) -> list[MemoryCollector]:
    """Return the memory collectors that apply to the current host.

    ``uses_gpu`` should reflect the engine's effective configuration — e.g.
    for llama.cpp, ``n_gpu_layers != 0``.  When ``False``, VRAM collectors
    are skipped and system RAM is used instead.
    """
    system = (system_name or platform.system()).lower()
    collectors: list[MemoryCollector] = []

    if system == "linux":
        if uses_gpu:
            nvml = NvidiaVramCollector.try_create()
            if nvml is not None:
                collectors.append(nvml)
            amd = AmdVramCollector.try_create()
            if amd is not None:
                collectors.append(amd)
        # Always include RAM — for GPU workers it is the secondary signal
        # weighted by _ram_weight(); for CPU workers it is the only signal.
        collectors.append(LinuxRamCollector())
    elif system == "darwin":
        macos = MacOSMemoryCollector.try_create()
        if macos is not None:
            collectors.append(macos)
    elif system == "windows":
        if uses_gpu:
            nvml = NvidiaVramCollector.try_create()
            if nvml is not None:
                collectors.append(nvml)
        if not collectors:
            collectors.append(NullMemoryCollector())
    else:
        logger.debug("Unknown system for memory collection | system={}", system)

    if not collectors:
        collectors.append(NullMemoryCollector())

    logger.info(
        "Memory collectors selected | system={} usesGpu={} names={}",
        system,
        uses_gpu,
        [c.name for c in collectors],
    )
    return collectors
