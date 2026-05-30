from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from telemetry.memory.amd import AmdVramCollector
from telemetry.memory.base import (
    MEMORY_EMA_ALPHA,
    RAM_PRESSURE_HIGH,
    RAM_PRESSURE_MID,
    RAM_WEIGHT_DEFAULT,
    RAM_WEIGHT_HIGH,
    RAM_WEIGHT_MID,
    MemoryAggregator,
    MemorySnapshot,
    NullMemoryCollector,
    aggregate_memory_samples,
)
from telemetry.memory.factory import detect_memory_collectors
from telemetry.memory.linux_cpu import LinuxRamCollector, _read_meminfo


# ── stub helpers ─────────────────────────────────────────────────────────────


class _FixedCollector:
    """Returns a fixed pressure value every sample."""

    def __init__(self, name: str, pressure: float, device_type: str = "gpu") -> None:
        self.name = name
        self._pressure = pressure
        self._device_type = device_type
        self.closed = False

    async def sample(self) -> MemorySnapshot | None:
        return MemorySnapshot(
            source=self.name,
            device_type=self._device_type,
            pressure=self._pressure,
        )

    async def close(self) -> None:
        self.closed = True


class _SequenceCollector:
    """Returns pressures from a list in order."""

    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name
        self._values = list(values)

    async def sample(self) -> MemorySnapshot | None:
        if not self._values:
            return None
        pressure = self._values.pop(0)
        return MemorySnapshot(source=self.name, device_type="gpu", pressure=pressure)

    async def close(self) -> None:
        return None


class _RaisingCollector:
    name = "raises"

    async def sample(self) -> MemorySnapshot | None:
        raise RuntimeError("sensor broken")

    async def close(self) -> None:
        return None


class _NoneCollector:
    """Returns None to simulate a collector that cannot read its source."""

    name = "none"

    async def sample(self) -> MemorySnapshot | None:
        return None

    async def close(self) -> None:
        return None


# ── NullMemoryCollector ───────────────────────────────────────────────────────


def test_null_collector_returns_zero_pressure() -> None:
    snapshot = asyncio.run(NullMemoryCollector().sample())

    assert snapshot is not None
    assert snapshot.pressure == 0.0


# ── MemoryAggregator: basic behavior ─────────────────────────────────────────


def test_aggregator_single_collector_pressure() -> None:
    agg = MemoryAggregator([_FixedCollector("a", 0.6)])

    result = asyncio.run(agg.sample())

    assert result.pressure == pytest.approx(0.6)
    assert result.raw_pressure == pytest.approx(0.6)


def test_aggregator_returns_zero_when_all_zero() -> None:
    agg = MemoryAggregator(
        [_FixedCollector("a", 0.0), _FixedCollector("b", 0.0)],
    )

    assert asyncio.run(agg.sample()).pressure == pytest.approx(0.0)


def test_aggregator_takes_max_across_collectors() -> None:
    agg = MemoryAggregator(
        [_FixedCollector("low", 0.3), _FixedCollector("high", 0.8)],
    )

    result = asyncio.run(agg.sample())

    assert result.raw_pressure == pytest.approx(0.8)


def test_aggregator_skips_raising_collector() -> None:
    agg = MemoryAggregator([_RaisingCollector(), _FixedCollector("healthy", 0.5)])

    result = asyncio.run(agg.sample())

    assert result.pressure == pytest.approx(0.5)


def test_aggregator_none_collector_contributes_zero() -> None:
    agg = MemoryAggregator([_NoneCollector()])

    result = asyncio.run(agg.sample())

    assert result.raw_pressure == pytest.approx(0.0)


def test_aggregator_close_closes_each_collector() -> None:
    a = _FixedCollector("a", 0.0)
    b = _FixedCollector("b", 0.0)
    agg = MemoryAggregator([a, b])

    asyncio.run(agg.close())

    assert a.closed
    assert b.closed


# ── Asymmetric EMA ────────────────────────────────────────────────────────────


def test_aggregator_spike_is_reflected_immediately() -> None:
    """A sudden pressure spike must not be smoothed away — instant rise."""
    agg = MemoryAggregator([_SequenceCollector("s", [0.0, 0.0, 0.9])])

    asyncio.run(agg.sample())  # 0.0
    asyncio.run(agg.sample())  # 0.0
    result = asyncio.run(agg.sample())  # 0.9

    assert result.pressure == pytest.approx(0.9)


def test_aggregator_pressure_drop_decays_gradually() -> None:
    """After a spike the EMA should still be above the new low on the next sample."""
    agg = MemoryAggregator([_SequenceCollector("s", [0.9, 0.0])])

    asyncio.run(agg.sample())  # spike → ema = 0.9
    result = asyncio.run(agg.sample())  # raw = 0.0 → ema decays but stays above 0

    # ema = ALPHA * 0.0 + (1 - ALPHA) * 0.9
    expected = (1.0 - MEMORY_EMA_ALPHA) * 0.9
    assert result.pressure == pytest.approx(expected, rel=1e-4)
    assert result.pressure > 0.0


def test_aggregator_ema_converges_to_low_over_many_samples() -> None:
    """After many low samples the EMA should decay close to zero."""
    agg = MemoryAggregator([_SequenceCollector("s", [0.9] + [0.0] * 30)])

    for _ in range(31):
        result = asyncio.run(agg.sample())

    assert result.pressure < 0.05


def test_aggregator_ema_alpha_zero_holds_peak_forever() -> None:
    """Alpha=0 means EMA never decays — peak pressure is permanent."""
    agg = MemoryAggregator([_SequenceCollector("s", [0.8, 0.0, 0.0])], ema_alpha=0.0)

    asyncio.run(agg.sample())  # spike
    asyncio.run(agg.sample())  # should stay at 0.8
    result = asyncio.run(agg.sample())

    assert result.pressure == pytest.approx(0.8)


def test_aggregator_ema_alpha_one_always_tracks_raw() -> None:
    """Alpha=1.0 means EMA == raw on every sample (no memory)."""
    agg = MemoryAggregator([_SequenceCollector("s", [0.9, 0.1])], ema_alpha=1.0)

    asyncio.run(agg.sample())  # 0.9
    result = asyncio.run(agg.sample())  # 0.1

    assert result.pressure == pytest.approx(0.1)


def test_aggregator_collector_names() -> None:
    agg = MemoryAggregator([_FixedCollector("nvidia-vram", 0.5), _FixedCollector("amd-vram", 0.3)])

    assert agg.collector_names == ["nvidia-vram", "amd-vram"]


# ── AMD sysfs collector ───────────────────────────────────────────────────────


def _write_vram(root: Path, card: str, used: int, total: int) -> None:
    device = root / card / "device"
    device.mkdir(parents=True, exist_ok=True)
    (device / "mem_info_vram_used").write_text(str(used))
    (device / "mem_info_vram_total").write_text(str(total))


def test_amd_collector_reads_single_card(tmp_path: Path) -> None:
    _write_vram(tmp_path, "card0", used=4_000_000_000, total=8_000_000_000)
    collector = AmdVramCollector(drm_root=tmp_path)

    result = asyncio.run(collector.sample())

    assert result is not None
    assert result.pressure == pytest.approx(0.5)
    assert result.used_bytes == 4_000_000_000
    assert result.total_bytes == 8_000_000_000


def test_amd_collector_returns_most_pressured_card(tmp_path: Path) -> None:
    _write_vram(tmp_path, "card0", used=2_000_000_000, total=8_000_000_000)
    _write_vram(tmp_path, "card1", used=7_000_000_000, total=8_000_000_000)
    collector = AmdVramCollector(drm_root=tmp_path)

    result = asyncio.run(collector.sample())

    assert result is not None
    assert result.pressure == pytest.approx(7 / 8)


def test_amd_collector_missing_root_returns_none(tmp_path: Path) -> None:
    collector = AmdVramCollector(drm_root=tmp_path / "absent")

    result = asyncio.run(collector.sample())

    assert result is None


def test_amd_try_create_returns_none_without_vram_files(tmp_path: Path) -> None:
    (tmp_path / "card0" / "device").mkdir(parents=True)

    assert AmdVramCollector.try_create(drm_root=tmp_path) is None


def test_amd_try_create_returns_collector_with_vram_files(tmp_path: Path) -> None:
    _write_vram(tmp_path, "card0", used=1, total=8_000_000_000)

    assert AmdVramCollector.try_create(drm_root=tmp_path) is not None


# ── Linux RAM collector ───────────────────────────────────────────────────────


def _write_meminfo(path: Path, *, total_kb: int, available_kb: int) -> None:
    path.write_text(
        f"MemTotal:       {total_kb} kB\n"
        f"MemFree:        {available_kb // 2} kB\n"
        f"MemAvailable:   {available_kb} kB\n"
        "Buffers:           128 kB\n"
    )


def test_linux_ram_collector_basic_pressure(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_kb=16_000_000, available_kb=4_000_000)
    collector = LinuxRamCollector(meminfo_path=meminfo)

    result = asyncio.run(collector.sample())

    assert result is not None
    assert result.pressure == pytest.approx(1.0 - 4_000_000 / 16_000_000)


def test_linux_ram_collector_fully_free(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_kb=8_000_000, available_kb=8_000_000)
    collector = LinuxRamCollector(meminfo_path=meminfo)

    result = asyncio.run(collector.sample())

    assert result is not None
    assert result.pressure == pytest.approx(0.0)


def test_linux_ram_collector_missing_file_returns_none(tmp_path: Path) -> None:
    collector = LinuxRamCollector(meminfo_path=tmp_path / "absent")

    assert asyncio.run(collector.sample()) is None


def test_linux_ram_meminfo_parse_returns_none_without_available(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8000000 kB\nMemFree: 4000000 kB\n")
    total, available = _read_meminfo(meminfo)

    assert total == 8_000_000
    assert available is None


# ── factory ───────────────────────────────────────────────────────────────────


def test_factory_unknown_system_returns_null() -> None:
    collectors = detect_memory_collectors(uses_gpu=True, system_name="plan9")

    assert len(collectors) == 1
    assert collectors[0].name == "null"


def test_factory_linux_no_gpu_returns_ram_collector() -> None:
    collectors = detect_memory_collectors(uses_gpu=False, system_name="Linux")

    names = [c.name for c in collectors]
    assert "linux-ram" in names


def test_factory_always_returns_at_least_one_collector() -> None:
    collectors = detect_memory_collectors(uses_gpu=True, system_name="os2warp")

    assert len(collectors) >= 1


def test_factory_linux_gpu_always_includes_ram_collector() -> None:
    # GPU workers always get LinuxRamCollector as the secondary RAM signal
    # regardless of whether an NVML/AMD collector was found.
    collectors = detect_memory_collectors(uses_gpu=True, system_name="Linux")

    names = [c.name for c in collectors]
    assert "linux-ram" in names


# ── aggregate_memory_samples: two-signal GPU/RAM logic ───────────────────────


def _gpu_snap(pressure: float) -> MemorySnapshot:
    return MemorySnapshot(source="test-gpu", device_type="gpu", pressure=pressure)


def _ram_snap(pressure: float) -> MemorySnapshot:
    return MemorySnapshot(source="test-ram", device_type="cpu", pressure=pressure)


def test_gpu_primary_gpu_dominates_when_vram_is_high() -> None:
    pressure, gpu_p, cpu_p, rw = aggregate_memory_samples(
        [_gpu_snap(0.9), _ram_snap(0.2)],
        primary_device="gpu",
    )

    assert gpu_p == pytest.approx(0.9)
    assert cpu_p == pytest.approx(0.2)
    assert rw == pytest.approx(RAM_WEIGHT_DEFAULT)
    assert pressure == pytest.approx(0.9)


def test_gpu_primary_low_vram_high_ram_contributes_with_mid_weight() -> None:
    # RAM at 0.75 → between MID and HIGH thresholds → RAM_WEIGHT_MID (0.50)
    pressure, gpu_p, cpu_p, rw = aggregate_memory_samples(
        [_gpu_snap(0.1), _ram_snap(0.75)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_MID)
    assert pressure == pytest.approx(0.75 * RAM_WEIGHT_MID)


def test_gpu_primary_near_swap_ram_uses_high_weight() -> None:
    # RAM > 0.85 → swap territory → RAM_WEIGHT_HIGH (0.75)
    pressure, _, cpu_p, rw = aggregate_memory_samples(
        [_gpu_snap(0.0), _ram_snap(0.9)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_HIGH)
    assert pressure == pytest.approx(0.9 * RAM_WEIGHT_HIGH)


def test_gpu_primary_safe_ram_uses_default_weight() -> None:
    # RAM ≤ 0.65 → GPU fully dominant → RAM_WEIGHT_DEFAULT (0.25)
    pressure, _, cpu_p, rw = aggregate_memory_samples(
        [_gpu_snap(0.3), _ram_snap(0.5)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_DEFAULT)
    assert pressure == pytest.approx(max(0.3, 0.5 * RAM_WEIGHT_DEFAULT))


def test_gpu_primary_boundary_exactly_at_mid_threshold() -> None:
    # Exactly at RAM_PRESSURE_MID → still default weight (not strictly greater)
    _, _, _, rw = aggregate_memory_samples(
        [_gpu_snap(0.0), _ram_snap(RAM_PRESSURE_MID)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_DEFAULT)


def test_gpu_primary_just_above_mid_threshold_uses_mid_weight() -> None:
    _, _, _, rw = aggregate_memory_samples(
        [_gpu_snap(0.0), _ram_snap(RAM_PRESSURE_MID + 0.01)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_MID)


def test_gpu_primary_boundary_exactly_at_high_threshold() -> None:
    # Exactly at RAM_PRESSURE_HIGH → still mid weight
    _, _, _, rw = aggregate_memory_samples(
        [_gpu_snap(0.0), _ram_snap(RAM_PRESSURE_HIGH)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_MID)


def test_gpu_primary_just_above_high_threshold_uses_high_weight() -> None:
    _, _, _, rw = aggregate_memory_samples(
        [_gpu_snap(0.0), _ram_snap(RAM_PRESSURE_HIGH + 0.01)],
        primary_device="gpu",
    )

    assert rw == pytest.approx(RAM_WEIGHT_HIGH)


def test_cpu_primary_ram_is_full_signal() -> None:
    # CPU-only worker: RAM pressure is returned as-is, weight=1.0
    pressure, gpu_p, cpu_p, rw = aggregate_memory_samples(
        [_ram_snap(0.8)],
        primary_device="cpu",
    )

    assert pressure == pytest.approx(0.8)
    assert rw == pytest.approx(1.0)
    assert gpu_p == pytest.approx(0.0)


def test_cpu_primary_ignores_any_gpu_sources() -> None:
    # For a CPU worker any GPU source should not drive the result
    pressure, _, _, _ = aggregate_memory_samples(
        [_gpu_snap(0.9), _ram_snap(0.4)],
        primary_device="cpu",
    )

    assert pressure == pytest.approx(0.4)


def test_no_sources_returns_zero() -> None:
    pressure, gpu_p, cpu_p, rw = aggregate_memory_samples([], primary_device="gpu")

    assert pressure == pytest.approx(0.0)
    assert gpu_p == pytest.approx(0.0)
    assert cpu_p == pytest.approx(0.0)


def test_aggregator_with_primary_device_propagates_to_snapshot() -> None:
    class _GpuCollector:
        name = "gpu"

        async def sample(self) -> MemorySnapshot | None:
            return MemorySnapshot(source="gpu", device_type="gpu", pressure=0.7)

        async def close(self) -> None:
            return None

    class _RamCollector:
        name = "ram"

        async def sample(self) -> MemorySnapshot | None:
            return MemorySnapshot(source="ram", device_type="cpu", pressure=0.9)

        async def close(self) -> None:
            return None

    # RAM at 0.9 → RAM_WEIGHT_HIGH (0.75); GPU at 0.7 → GPU should win
    agg = MemoryAggregator([_GpuCollector(), _RamCollector()], primary_device="gpu")
    snapshot = asyncio.run(agg.sample())

    assert snapshot.gpu_pressure == pytest.approx(0.7)
    assert snapshot.cpu_pressure == pytest.approx(0.9)
    assert snapshot.ram_weight == pytest.approx(RAM_WEIGHT_HIGH)
    # GPU (0.7) vs RAM×weight (0.9×0.75=0.675) → GPU wins
    assert snapshot.raw_pressure == pytest.approx(0.7)
