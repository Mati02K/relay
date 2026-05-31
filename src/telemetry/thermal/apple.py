"""macOS thermal pressure collector via ``NSProcessInfo.thermalState``.

Apple Silicon shares a thermal envelope between CPU and GPU, so a single
process-info reading covers both. We invoke a Swift one-liner because it
requires no sudo, no Xcode-only frameworks, and works without any pip
dependencies. ``swift`` ships with the Xcode Command Line Tools which are a
near-universal install on developer machines; if it is missing the factory
falls back to ``NullThermalCollector``.

Thermal state values:
    0 nominal   ->  not throttled
    1 fair      ->  not throttled (advisory; reduce non-visible work)
    2 serious   ->  throttled
    3 critical  ->  throttled
"""

from __future__ import annotations

import asyncio
import shutil

from loguru import logger

from telemetry.thermal.base import ThermalSourceSample

_SWIFT_SCRIPT = "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)"
_THROTTLE_THRESHOLD = 2  # serious or critical
_PROBE_TIMEOUT_SECONDS = 2.0


def parse_thermal_state_output(raw: str) -> int | None:
    """Parse the integer raw value printed by the Swift one-liner."""
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text.splitlines()[0])
    except ValueError:
        return None


def thermal_state_to_theta_w(value: int) -> int:
    """Map the four-bucket thermal state to the binary ``theta_w`` flag."""
    return 1 if value >= _THROTTLE_THRESHOLD else 0


def thermal_state_to_pressure(value: int) -> float:
    """Map Apple's four-bucket thermal state to normalized pressure."""
    if value <= 0:
        return 0.0
    if value == 1:
        return 0.35
    if value == 2:
        return 0.75
    return 1.0


def thermal_state_to_label(value: int) -> str:
    """Return a dashboard-friendly label for Apple's thermal state."""
    if value <= 0:
        return "normal"
    if value == 1:
        return "warm"
    if value == 2:
        return "throttling"
    return "critical"


class MacOSThermalStateCollector:
    """Reads ``NSProcessInfo.thermalState`` via a non-privileged Swift subprocess."""

    name = "macos-thermal-state"

    def __init__(self, swift_binary: str = "swift") -> None:
        self._swift_binary = swift_binary

    @classmethod
    def try_create(cls) -> MacOSThermalStateCollector | None:
        """Return a collector when ``swift`` is on PATH; ``None`` otherwise."""
        swift_binary = shutil.which("swift")
        if not swift_binary:
            logger.debug("swift binary not found, macOS thermal collector disabled")
            return None
        return cls(swift_binary=swift_binary)

    async def sample(self) -> list[ThermalSourceSample]:
        """Return normalized thermal pressure from ``NSProcessInfo``."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._swift_binary,
                "-e",
                _SWIFT_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError) as e:
            logger.debug("Swift subprocess failed | error={}", e)
            return []

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return []

        if process.returncode != 0:
            return []
        parsed = parse_thermal_state_output(stdout.decode(errors="replace"))
        if parsed is None:
            return []
        pressure = thermal_state_to_pressure(parsed)
        logger.debug(
            "macOS thermal sample | rawState={} pressure={:.3f}",
            parsed,
            pressure,
        )
        return [
            ThermalSourceSample(
                source=self.name,
                device_type="system",
                pressure=pressure,
                state=thermal_state_to_label(parsed),
                throttle_active=parsed >= _THROTTLE_THRESHOLD,
                details={"raw_state": parsed},
            )
        ]

    async def close(self) -> None:
        """No persistent resources."""
        return None
