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
    ThermalSourceSample,
    pressure_from_temperature,
    pressure_from_throttle_rate,
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

    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name
        self._values = list(values)
        self.closed = False

    async def sample(self) -> list[ThermalSourceSample]:
        pressure = self._values.pop(0)
        return [
            ThermalSourceSample(
                source=self.name,
                device_type="cpu",
                pressure=pressure,
                state="normal",
                confidence=1.0,
            )
        ]

    async def close(self) -> None:
        self.closed = True


class _RaisingCollector:
    name = "raises"

    async def sample(self) -> list[ThermalSourceSample]:
        raise RuntimeError("sensor broken")

    async def close(self) -> None:
        return None


class _TypedStubCollector:
    def __init__(self, name: str, device_type: str, pressure: float) -> None:
        self.name = name
        self._device_type = device_type
        self._pressure = pressure

    async def sample(self) -> list[ThermalSourceSample]:
        return [
            ThermalSourceSample(
                source=self.name,
                device_type=self._device_type,
                pressure=self._pressure,
                state="normal",
                confidence=1.0,
            )
        ]

    async def close(self) -> None:
        return None


class _FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_null_collector_always_zero() -> None:
    samples = asyncio.run(NullThermalCollector().sample())

    assert samples[0].pressure == 0


def test_aggregator_ors_collectors() -> None:
    aggregator = ThermalAggregator(
        [_StubCollector("a", [0]), _StubCollector("b", [1])],
    )

    assert asyncio.run(aggregator.sample()).pressure == 1


def test_aggregator_returns_zero_when_all_zero() -> None:
    aggregator = ThermalAggregator(
        [_StubCollector("a", [0]), _StubCollector("b", [0])],
    )

    assert asyncio.run(aggregator.sample()).pressure == 0


def test_aggregator_downweights_cpu_pressure_for_gpu_primary() -> None:
    aggregator = ThermalAggregator(
        [
            _TypedStubCollector("cpu", "cpu", 0.9),
            _TypedStubCollector("gpu", "gpu", 0.0),
        ],
        primary_device="gpu",
    )

    snapshot = asyncio.run(aggregator.sample())

    assert snapshot.cpu_pressure == 0.9
    assert snapshot.gpu_pressure == 0.0
    assert snapshot.pressure == pytest.approx(0.315)


def test_aggregator_uses_gpu_pressure_for_gpu_primary() -> None:
    aggregator = ThermalAggregator(
        [
            _TypedStubCollector("cpu", "cpu", 0.9),
            _TypedStubCollector("gpu", "gpu", 0.7),
        ],
        primary_device="gpu",
    )

    snapshot = asyncio.run(aggregator.sample())

    assert snapshot.pressure == pytest.approx(0.7)


def test_aggregator_treats_raising_collector_as_zero() -> None:
    raising = _RaisingCollector()
    healthy = _StubCollector("healthy", [0])
    aggregator = ThermalAggregator([raising, healthy])

    assert asyncio.run(aggregator.sample()).pressure == 0


def test_aggregator_close_closes_each_collector() -> None:
    a = _StubCollector("a", [])
    b = _StubCollector("b", [])
    aggregator = ThermalAggregator([a, b])

    asyncio.run(aggregator.close())

    assert a.closed
    assert b.closed


def test_linux_cpu_collector_first_sample_is_zero(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path,
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
    )

    assert asyncio.run(collector.sample())[0].pressure == 0


def test_linux_cpu_collector_detects_increase(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path,
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
    )
    asyncio.run(collector.sample())

    _write_throttle(tmp_path, cpu="cpu0", core=7, package=3)

    assert asyncio.run(collector.sample())[0].pressure > 0


def test_linux_cpu_collector_treats_single_small_delta_as_transient(tmp_path: Path) -> None:
    clock = _FakeClock()
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path,
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
        clock=clock,
    )
    asyncio.run(collector.sample())

    clock.advance(5.0)
    _write_throttle(tmp_path, cpu="cpu0", core=6, package=3)
    sample = asyncio.run(collector.sample())[0]

    assert sample.pressure == pytest.approx(0.045)
    assert sample.state == "normal"
    assert sample.details["throttle_rate_per_sec"] == pytest.approx(0.2)
    assert sample.details["throttle_pressure_raw"] == pytest.approx(0.15)


def test_linux_cpu_collector_requires_sustained_delta_to_throttle(tmp_path: Path) -> None:
    clock = _FakeClock()
    _write_throttle(tmp_path, cpu="cpu0", core=0, package=0)
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path,
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
        clock=clock,
    )
    asyncio.run(collector.sample())

    sample = None
    for total in (300, 600, 900, 1200):
        clock.advance(5.0)
        _write_throttle(tmp_path, cpu="cpu0", core=total, package=0)
        sample = asyncio.run(collector.sample())[0]

    assert sample is not None
    assert sample.pressure > 0.65
    assert sample.state == "throttling"


def test_linux_cpu_collector_stable_count_returns_zero(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=5, package=3)
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path,
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
    )
    asyncio.run(collector.sample())

    assert asyncio.run(collector.sample())[0].pressure == 0


def test_linux_cpu_collector_missing_sysfs_returns_zero(tmp_path: Path) -> None:
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path / "absent",
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
    )

    assert asyncio.run(collector.sample())[0].pressure == 0


def test_linux_cpu_collector_sums_multiple_cpus(tmp_path: Path) -> None:
    _write_throttle(tmp_path, cpu="cpu0", core=1, package=0)
    _write_throttle(tmp_path, cpu="cpu1", core=2, package=0)
    collector = LinuxCpuThrottleCollector(
        sysfs_root=tmp_path,
        thermal_root=tmp_path / "thermal",
        hwmon_root=tmp_path / "hwmon",
    )
    asyncio.run(collector.sample())

    _write_throttle(tmp_path, cpu="cpu1", core=4, package=0)

    assert asyncio.run(collector.sample())[0].pressure > 0


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0, 0.0),
        (1.0, 0.15),
        (5.0, 0.35),
        (20.0, 0.65),
        (60.0, 0.9),
    ],
)
def test_cpu_throttle_rate_mapping(rate: float, expected: float) -> None:
    assert pressure_from_throttle_rate(rate) == expected


def test_pressure_from_temperature_normal_temps_return_zero() -> None:
    # Default 8C warm margin. A chip cruising well below the limit must
    # report no pressure — being warm is not the same as being throttled.
    assert pressure_from_temperature(70.0, 100.0) == 0.0
    assert pressure_from_temperature(88.0, 100.0) == 0.0
    assert pressure_from_temperature(91.99, 100.0) == 0.0


def test_pressure_from_temperature_quadratic_inside_margin() -> None:
    # At the limit, pressure saturates at 1.0.
    assert pressure_from_temperature(100.0, 100.0) == pytest.approx(1.0, abs=1e-6)
    # Mid-margin (4C inside an 8C window) → ratio 0.5 → quadratic 0.25.
    assert pressure_from_temperature(96.0, 100.0) == pytest.approx(0.25, abs=1e-6)
    # 2C below limit → ratio 0.75 → quadratic ~0.5625.
    assert pressure_from_temperature(98.0, 100.0) == pytest.approx(0.5625, abs=1e-6)


def test_pressure_from_temperature_unknown_inputs_return_zero() -> None:
    assert pressure_from_temperature(None, 100.0) == 0.0
    assert pressure_from_temperature(95.0, None) == 0.0
    assert pressure_from_temperature(95.0, 0.0) == 0.0


def test_pressure_from_temperature_at_or_above_limit_saturates() -> None:
    assert pressure_from_temperature(105.0, 100.0) == 1.0
    assert pressure_from_temperature(200.0, 100.0) == 1.0


@pytest.mark.parametrize(
    ("reasons_bits", "expected"),
    [
        (0x0, 0),
        (0x1, 0),  # GPU idle, not thermal
        (0x4, 0),  # SW power cap, not thermal
        (0x8, 0),  # HW slowdown — non-specific; could be power brake / sync boost
        (0x20, 1),  # SW thermal slowdown — specifically thermal
        (0x40, 1),  # HW thermal slowdown — specifically thermal
        (0x60, 1),  # both thermal bits set
        (0x48, 1),  # HW slowdown + HW thermal — still thermal (the thermal bit)
        (0x44, 1),  # SW power cap + HW thermal — still thermal
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
