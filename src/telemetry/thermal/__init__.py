"""Thermal throttling collectors for the ``theta_w`` scheduler term."""

from telemetry.thermal.base import (
    NullThermalCollector,
    ThermalAggregator,
    ThermalCollector,
)
from telemetry.thermal.factory import detect_thermal_collectors

__all__ = [
    "NullThermalCollector",
    "ThermalAggregator",
    "ThermalCollector",
    "detect_thermal_collectors",
]
