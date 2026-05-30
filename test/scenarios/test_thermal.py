"""Thermal pressure routing scenario.

Observes how the scheduler routes requests as real CPU/GPU temperature rises
on one worker.  You create thermal load externally (CPU stress tools, running
a compute-intensive workload) and this test records routing at each phase.

How to apply thermal load
--------------------------
CPU stress (Linux):
  stress-ng --cpu 0 --timeout 120s        # all cores at 100%
  # or
  yes > /dev/null &                        # single core spike

GPU stress (NVIDIA):
  nvidia-smi dmon -s u                     # monitor GPU utilisation
  # Run a GPU workload like stable diffusion or a CUDA benchmark

Monitor temperature while applying load:
  watch -n 1 'sensors | grep -E "Core|GPU"'
  # or
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -l 1

How to run
----------
1. Start the cluster.
2. On worker-0's machine, start the stress tool.
3. Wait until temperature rises noticeably (watch sensors).
4. Run: pytest test/scenarios/test_thermal.py --coordinator http://<host>:8080

The live theta_w from /v1/workers is printed before each phase so you can
confirm real thermal pressure is registering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from framework.client import RelayClient
from framework.cluster import ClusterClient
from framework.metrics import (
    save_records_csv,
    save_records_json,
    worker_share,
)
from framework.report import (
    plot_latency_boxes_phases,
    plot_worker_distribution_phases,
)
from framework.workload import send_batch

SCENARIO = "thermal"
REQUESTS_PER_PHASE = 120
CONCURRENCY = 8


@pytest.mark.asyncio
async def test_thermal_baseline(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Baseline routing — no thermal pressure applied."""
    workers = await cluster.wait_for_workers(min_count=2)
    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=0.0, jitter=0.0, thermal=1.0)
    _log_telemetry(await cluster.get_workers())

    records = await send_batch(
        relay_client, short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO, phase="baseline", concurrency=CONCURRENCY,
    )
    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "baseline")


@pytest.mark.asyncio
async def test_thermal_under_load(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Routing while worker-0 CPU/GPU is thermally stressed.

    Run stress-ng or a heavy GPU workload on worker-0 before this test.
    """
    workers = await cluster.wait_for_workers(min_count=2)
    target = workers[0]

    all_workers = await cluster.get_workers()
    _log_telemetry(all_workers)

    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=0.0, jitter=0.0, thermal=1.0)

    records = await send_batch(
        relay_client, short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO, phase="hot", concurrency=CONCURRENCY,
    )
    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "hot")

    share = worker_share(records, target["node_id"])
    print(f"\n  target ({target['node_id']}) share while hot: {share:.1%}")
    print("  (should be lower than baseline if theta_w is elevated)")


@pytest.mark.asyncio
async def test_thermal_recovery(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Routing after stress tool is stopped and temperature drops."""
    workers = await cluster.wait_for_workers(min_count=2)
    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=0.0, jitter=0.0, thermal=1.0)
    await cluster.wait_telemetry_propagation(seconds=5.0)
    _log_telemetry(await cluster.get_workers())

    records = await send_batch(
        relay_client, short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO, phase="recovery", concurrency=CONCURRENCY,
    )
    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "recovery")


def _log_telemetry(workers: list[Any]) -> None:
    for w in workers:
        t = w.get("telemetry", {})
        print(f"  [{SCENARIO}] {w['node_id']}  "
              f"theta_w={t.get('theta_w', '?'):.3f}  mw={t.get('mw', '?'):.3f}")


def _save_and_plot(records: Any, run_dir: Path) -> None:
    save_records_csv(records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_worker_distribution_phases(records, SCENARIO, plots_dir)
    plot_latency_boxes_phases(records, SCENARIO, plots_dir)


def _print_distribution(records: Any, workers: list[Any], phase: str) -> None:
    from framework.metrics import _percentile
    ttfts = [r.ttft_ms for r in records]
    print(f"\n  [{SCENARIO}] phase={phase}  n={len(records)}  "
          f"P50={_percentile(ttfts, 50):.0f}ms  P99={_percentile(ttfts, 99):.0f}ms")
    for w in workers:
        share = worker_share(records, w["node_id"])
        print(f"    {w['node_id']}: {share:.1%}")
