"""Per-platform thermal-collector factory.

Maps ``(os, gpu_in_use)`` to the right set of concrete collectors:

* Linux  : CPU sysfs counters + NVIDIA NVML when the model offloads to GPU.
* macOS  : ``NSProcessInfo.thermalState`` only — CPU and GPU share the envelope.
* Windows: WMI thermal zones (admin) + NVIDIA NVML when the model offloads to GPU.

Anything else, or a host where no source is readable, returns
``[NullThermalCollector]`` so the aggregator still produces a well-defined ``0``.
"""

from __future__ import annotations

import platform

from loguru import logger

from telemetry.thermal.apple import MacOSThermalStateCollector
from telemetry.thermal.base import NullThermalCollector, ThermalCollector
from telemetry.thermal.linux import LinuxCpuThrottleCollector
from telemetry.thermal.nvidia import NvidiaNvmlCollector
from telemetry.thermal.windows import WindowsCpuThermalCollector


def detect_thermal_collectors(
    *,
    uses_gpu: bool,
    system_name: str | None = None,
) -> list[ThermalCollector]:
    """Return the thermal collectors that apply to the current host.

    ``uses_gpu`` should reflect the engine's effective config — e.g. for
    llama.cpp, ``n_gpu_layers != 0``. When False, GPU collectors are skipped
    even on hosts that have an NVIDIA card, since an idle GPU's thermal state
    has no bearing on the model running on the CPU.
    """
    system = (system_name or platform.system()).lower()
    collectors: list[ThermalCollector] = []

    if system == "linux":
        collectors.append(LinuxCpuThrottleCollector())
        if uses_gpu:
            nvml = NvidiaNvmlCollector.try_create()
            if nvml is not None:
                collectors.append(nvml)
    elif system == "darwin":
        macos = MacOSThermalStateCollector.try_create()
        if macos is not None:
            collectors.append(macos)
    elif system == "windows":
        windows = WindowsCpuThermalCollector.try_create()
        if windows is not None:
            collectors.append(windows)
        if uses_gpu:
            nvml = NvidiaNvmlCollector.try_create()
            if nvml is not None:
                collectors.append(nvml)
    else:
        logger.debug("Unknown system for thermal collection | system={}", system)

    if not collectors:
        collectors.append(NullThermalCollector())
    logger.info(
        "Thermal collectors selected | system={} usesGpu={} names={}",
        system,
        uses_gpu,
        [collector.name for collector in collectors],
    )
    return collectors
