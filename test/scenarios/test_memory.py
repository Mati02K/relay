"""Memory pressure routing scenario.

Observes how the scheduler routes requests as real VRAM/RAM pressure rises
and falls on one worker.  You apply pressure externally (run a GPU-heavy
workload, a game, or ``stress-ng --vm`` on the target machine) and this test
records the routing distribution at each phase.

How to run
----------
1. Start the cluster (coordinator + ≥ 2 workers).
2. On the machine running worker-0, apply GPU/RAM load:

   # GPU VRAM load (fills ~4 GB of VRAM):
   python -c "import torch; t = torch.zeros(1024,1024,1024, device='cuda'); input('hold...')"

   # Or CPU RAM load:
   stress-ng --vm 1 --vm-bytes 80% --timeout 120s

3. Run this test:
   pytest test/scenarios/test_memory.py --coordinator http://<host>:8080

The test sends 120 requests per phase and records routing distribution,
TTFT percentiles, and the live mw value read from /v1/workers.

Phases (you transition manually between them)
---------------------------------------------
baseline   No pressure applied — measure normal distribution
pressure   Apply load on worker-0 before running this phase
recovery   Remove load — measure routing recovery

To run all phases in sequence, re-run the test with --phase flag:
   pytest test/scenarios/test_memory.py --coordinator ... -k baseline
   # <apply load>
   pytest test/scenarios/test_memory.py --coordinator ... -k pressure
   # <remove load>
   pytest test/scenarios/test_memory.py --coordinator ... -k recovery
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from framework.client import RelayClient
from framework.cluster import ClusterClient
from framework.metrics import (
    all_phase_stats,
    save_records_csv,
    save_records_json,
    worker_share,
)
from framework.report import (
    plot_latency_boxes_phases,
    plot_worker_distribution_phases,
)
from framework.workload import send_batch

SCENARIO = "memory"
REQUESTS_PER_PHASE = 120
CONCURRENCY = 8


@pytest.mark.asyncio
async def test_memory_baseline(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Measure baseline routing distribution with no pressure applied."""
    workers = await cluster.wait_for_workers(min_count=2)
    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=1.0, jitter=0.0, thermal=0.0)

    records = await send_batch(
        relay_client,
        short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO,
        phase="baseline",
        concurrency=CONCURRENCY,
    )

    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "baseline")


@pytest.mark.asyncio
async def test_memory_under_pressure(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Measure routing distribution while worker-0 is under real memory pressure.

    Apply VRAM or RAM load on the worker-0 machine before running this test.
    The live mw value from /v1/workers is logged so you can confirm pressure
    is actually registered.
    """
    workers = await cluster.wait_for_workers(min_count=2)
    target = workers[0]

    # Log live telemetry so you can see the real mw value
    all_workers = await cluster.get_workers()
    for w in all_workers:
        t = w.get("telemetry", {})
        print(f"  [{SCENARIO}] {w['node_id']}  mw={t.get('mw', '?'):.3f}  "
              f"theta_w={t.get('theta_w', '?'):.3f}")

    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=1.0, jitter=0.0, thermal=0.0)

    records = await send_batch(
        relay_client,
        short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO,
        phase="pressure",
        concurrency=CONCURRENCY,
    )

    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "pressure")

    share = worker_share(records, target["node_id"])
    print(f"\n  target worker ({target['node_id']}) share under pressure: {share:.1%}")
    print("  (compare to baseline — should be lower if mw is elevated)")


@pytest.mark.asyncio
async def test_memory_recovery(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Measure routing after pressure is removed — should recover toward baseline."""
    workers = await cluster.wait_for_workers(min_count=2)
    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=1.0, jitter=0.0, thermal=0.0)
    await cluster.wait_telemetry_propagation(seconds=3.0)

    records = await send_batch(
        relay_client,
        short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO,
        phase="recovery",
        concurrency=CONCURRENCY,
    )

    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "recovery")


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
