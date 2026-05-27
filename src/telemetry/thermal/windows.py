"""Windows CPU thermal collector via WMI ``MSAcpi_ThermalZoneTemperature``.

The ACPI thermal zone interface requires the Relay worker process to run with
Administrator privileges. The factory does **not** check for elevation — if the
query returns nothing the collector simply reports ``0``, the same as an
unsupported machine. Document this requirement to users running on Windows.

CurrentTemperature is reported in tenths of a Kelvin. ``CurrentTemperature``
of ``2982`` therefore decodes to ``25.05 C``. We compare the maximum reading
against a configurable Celsius threshold (default 90 C). NVIDIA GPUs on
Windows are covered by :mod:`telemetry.thermal.nvidia`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil

from loguru import logger

from telemetry.thermal.base import (
    ThermalSourceSample,
    pressure_from_temperature,
    pressure_to_state,
)

_POWERSHELL_COMMAND = (
    "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
    "| Select-Object -ExpandProperty CurrentTemperature "
    "| ConvertTo-Json -Compress"
)
_PROBE_TIMEOUT_SECONDS = 3.0
DEFAULT_THRESHOLD_CELSIUS = float(os.getenv("RELAY_WIN_THERMAL_THRESHOLD_C", "90.0"))


def parse_powershell_temperatures(raw: str) -> list[float]:
    """Parse the JSON emitted by the PowerShell query into Celsius readings."""
    text = raw.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        values = payload
    else:
        values = [payload]
    celsius: list[float] = []
    for value in values:
        if not isinstance(value, int | float):
            continue
        celsius.append(_tenth_kelvin_to_celsius(float(value)))
    return celsius


def temperatures_to_theta_w(celsius_values: list[float], threshold_c: float) -> int:
    """Return 1 if any temperature meets or exceeds ``threshold_c``."""
    if not celsius_values:
        return 0
    return 1 if max(celsius_values) >= threshold_c else 0


class WindowsCpuThermalCollector:
    """Reads ACPI thermal-zone temperatures on Windows via PowerShell + WMI."""

    name = "windows-cpu-thermal"

    def __init__(
        self,
        powershell_binary: str = "powershell",
        threshold_celsius: float = DEFAULT_THRESHOLD_CELSIUS,
    ) -> None:
        self._powershell_binary = powershell_binary
        self._threshold_celsius = threshold_celsius

    @classmethod
    def try_create(
        cls,
        threshold_celsius: float = DEFAULT_THRESHOLD_CELSIUS,
    ) -> WindowsCpuThermalCollector | None:
        """Return a collector when PowerShell is on PATH; ``None`` otherwise."""
        binary = shutil.which("powershell") or shutil.which("pwsh")
        if not binary:
            logger.debug("powershell not found, Windows thermal collector disabled")
            return None
        return cls(powershell_binary=binary, threshold_celsius=threshold_celsius)

    async def sample(self) -> list[ThermalSourceSample]:
        """Return normalized thermal-zone pressure samples."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._powershell_binary,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _POWERSHELL_COMMAND,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError) as e:
            logger.debug("PowerShell subprocess failed | error={}", e)
            return []

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return []

        if process.returncode != 0:
            return []
        temperatures = parse_powershell_temperatures(stdout.decode(errors="replace"))
        samples: list[ThermalSourceSample] = []
        for index, temp_c in enumerate(temperatures):
            pressure = pressure_from_temperature(temp_c, self._threshold_celsius)
            logger.debug(
                "Windows thermal sample | zone={} tempC={} limitC={} pressure={:.3f}",
                index,
                temp_c,
                self._threshold_celsius,
                pressure,
            )
            samples.append(
                ThermalSourceSample(
                    source=self.name,
                    device_type="system",
                    pressure=pressure,
                    state=pressure_to_state(pressure),
                    temperature_c=temp_c,
                    limit_c=self._threshold_celsius,
                    throttle_active=temp_c >= self._threshold_celsius,
                    details={"zone_index": index},
                )
            )
        return samples

    async def close(self) -> None:
        """No persistent resources."""
        return None


def _tenth_kelvin_to_celsius(value: float) -> float:
    return value / 10.0 - 273.15
