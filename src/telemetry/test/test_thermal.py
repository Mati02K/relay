from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from telemetry.thermal.apple import (
    parse_thermal_state_output,
    thermal_state_to_theta_w,
)
from telemetry.thermal.base import (
    NullThermalCollector,
    ThermalAggregator,
)
from telemetry.thermal.factory import detect_thermal_collectors
from telemetry.thermal.linux import LinuxCpuThrottleCollector
from telemetry.thermal.nvidia import is_thermal_throttle
from telemetry.thermal.windows import (
    parse_powershell_temperatures,
    temperatures_to_theta_w,
)


class _StubCollector:
    """In-memory collector used to drive the aggregator from tests."""

    def __init__(self, name: str, values: list[int]) -> None:
        self.name = name
        self._values = list(values)
        self.closed = False

    async def sample(self) -> int:
        return self._values.pop(0)

    async def close(self) -> None:
        self.closed = True


class _RaisingCollector:
    name = "raises"

    async def sample(self) -> int:
        raise RuntimeError("sensor broken")

    async def close(self) -> None:
        return None


def test_null_collector_always_zero() -> None:
    assert asyncio.run(NullThermalCollector().sample()) == 0


def test_aggregator_ors_collectors() -> None:
    aggregator = ThermalAggregator(
        [_StubCollector("a", [0]), _StubCollector("b", [1])],
    )

    assert asyncio.run(aggregator.sample()) == 1


def test_aggregator_returns_zero_when_all_zero() -> None:
    aggregator = ThermalAggregator(
        [_StubCollector("a", [0]), _StubCollector("b", [0])],
    )

    assert asyncio.run(aggregator.sample()) == 0


def test_aggregator_treats_raising_collector_as_zero() -> None:
    raising = _RaisingCollector()
    healthy = _StubCollector("healthy", [0])
    aggregator = ThermalAggregator([raising, healthy])

    assert asyncio.run(aggregator.sample()) == 0


def test_aggregator_close_closes_each_collector() -> None:
    a = _StubCollector("a", [])
    b = _StubCollector("b", [])
    aggregator = ThermalAggregator([a, b])

    asyncio.run(aggregator.close())

    assert a.closed
    assert b.closed


def test_linux_cpu_collector_first_sample_is_zero(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(sysfs_root=tmp_path)

    assert asyncio.run(collector.sample()) == 0


def test_linux_cpu_collector_detects_increase(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(sysfs_root=tmp_path)
    asyncio.run(collector.sample())

    _write_throttle(tmp_path, cpu="cpu0", core=7, package=3)

    assert asyncio.run(collector.sample()) == 1


def test_linux_cpu_collector_stable_count_returns_zero(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(sysfs_root=tmp_path)
    asyncio.run(collector.sample())

    assert asyncio.run(collector.sample()) == 0


def test_linux_cpu_collector_missing_sysfs_returns_zero(tmp_path: Path) -> None:
    collector = LinuxCpuThrottleCollector(sysfs_root=tmp_path / "absent")

    assert asyncio.run(collector.sample()) == 0


def test_linux_cpu_collector_sums_multiple_cpus(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=1, package=0)
    _write_throttle(tmp_path, cpu="cpu1", core=2, package=0)
    collector = LinuxCpuThrottleCollector(sysfs_root=tmp_path)
    asyncio.run(collector.sample())

    _write_throttle(tmp_path, cpu="cpu1", core=4, package=0)

    assert asyncio.run(collector.sample()) == 1


@pytest.mark.parametrize(
    ("reasons_bits", "expected"),
    [
        (0x0, 0),
        (0x1, 0),  # GPU idle, not thermal
        (0x4, 0),  # SW power cap, not thermal
        (0x8, 1),  # HW slowdown
        (0x20, 1),  # SW thermal slowdown
        (0x40, 1),  # HW thermal slowdown
        (0x60, 1),  # both thermal bits set
    ],
)
def test_nvidia_bit_decoder(reasons_bits: int, expected: int) -> None:
    assert is_thermal_throttle(reasons_bits) == expected


def test_macos_thermal_state_parser_handles_each_bucket() -> None:
    assert parse_thermal_state_output("0\n") == 0
    assert parse_thermal_state_output("3") == 3
    assert parse_thermal_state_output("") is None
    assert parse_thermal_state_output("oops") is None


@pytest.mark.parametrize(
    ("state", "expected"),
    [(0, 0), (1, 0), (2, 1), (3, 1)],
)
def test_macos_thermal_state_mapping(state: int, expected: int) -> None:
    assert thermal_state_to_theta_w(state) == expected


def test_windows_parser_decodes_tenth_kelvin_json_list() -> None:
    # 3032 -> 30.05 C, 3631 -> 89.95 C
    celsius = parse_powershell_temperatures("[3032, 3631]")
    assert celsius == pytest.approx([30.05, 89.95], abs=0.01)


def test_windows_parser_decodes_single_value() -> None:
    assert parse_powershell_temperatures("3032") == pytest.approx([30.05], abs=0.01)


def test_windows_threshold_compare() -> None:
    assert temperatures_to_theta_w([30.0, 89.99], threshold_c=90.0) == 0
    assert temperatures_to_theta_w([30.0, 90.0], threshold_c=90.0) == 1
    assert temperatures_to_theta_w([], threshold_c=90.0) == 0


def test_factory_unknown_system_returns_null() -> None:
    collectors = detect_thermal_collectors(uses_gpu=True, system_name="haiku-os")

    assert len(collectors) == 1
    assert collectors[0].name == "null"


def test_factory_linux_without_gpu_excludes_nvidia() -> None:
    collectors = detect_thermal_collectors(uses_gpu=False, system_name="Linux")

    names = [collector.name for collector in collectors]
    assert "linux-cpu-throttle" in names
    assert "nvidia-nvml" not in names


def _write_throttle(root: Path, *, cpu: str, core: int, package: int) -> None:
    cpu_dir = root / cpu / "thermal_throttle"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    (cpu_dir / "core_throttle_count").write_text(str(core))
    (cpu_dir / "package_throttle_count").write_text(str(package))
