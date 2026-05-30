"""Jitter routing scenario.

Observes how the scheduler routes requests as real network jitter rises on
one worker.  You introduce actual network delay on the target machine using
Linux traffic control, and this test records routing distribution and the live
jw value from the coordinator's jitter probe.

How to introduce real jitter (Linux, run on worker-0's machine)
---------------------------------------------------------------
# Add 200ms delay + 80ms variation to the loopback or LAN interface:
sudo tc qdisc add dev lo root netem delay 200ms 80ms

# Or for a LAN interface (replace eth0 with your interface):
sudo tc qdisc add dev eth0 root netem delay 200ms 80ms

# Verify it's active:
tc qdisc show dev lo

# Remove when done:
sudo tc qdisc del dev lo root

How the jitter probe works
--------------------------
The coordinator pings each worker's /health endpoint every 2 seconds.
RTT deviation is tracked via an EMA (alpha=0.15). After adding delay, wait
~12 seconds (≈6 probe cycles) for the EMA to reflect the new jitter level.

How to run
----------
1. Start cluster.
2. On worker-0's machine, run the tc command above.
3. Wait ~12 seconds.
4. Run: pytest test/scenarios/test_jitter.py --coordinator http://<host>:8080

The live jw (jitter EMA ms) from the coordinator is printed before each phase.
"""

from __future__ import annotations

import asyncio
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

SCENARIO = "jitter"
REQUESTS_PER_PHASE = 120
CONCURRENCY = 8
# How long to wait after the test prompts jitter probe EMA to converge
EMA_SETTLE_SECONDS = 12.0


@pytest.mark.asyncio
async def test_jitter_baseline(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Baseline — no artificial jitter on any worker."""
    workers = await cluster.wait_for_workers(min_count=2)
    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=0.0, jitter=1.0, thermal=0.0)
    _log_jitter(await cluster.get_workers())

    records = await send_batch(
        relay_client, short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO, phase="baseline", concurrency=CONCURRENCY,
    )
    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "baseline")


@pytest.mark.asyncio
async def test_jitter_under_network_delay(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Routing while worker-0 has artificial network delay via tc netem.

    Apply tc qdisc before running this test.  The jw EMA should be visibly
    elevated in the log below before requests are dispatched.
    """
    workers = await cluster.wait_for_workers(min_count=2)
    target = workers[0]

    print(f"\n  [{SCENARIO}] Waiting {EMA_SETTLE_SECONDS:.0f}s for jitter EMA to settle…")
    await asyncio.sleep(EMA_SETTLE_SECONDS)
    _log_jitter(await cluster.get_workers())

    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=0.0, jitter=1.0, thermal=0.0)

    records = await send_batch(
        relay_client, short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO, phase="high_jitter", concurrency=CONCURRENCY,
    )
    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "high_jitter")

    share = worker_share(records, target["node_id"])
    print(f"\n  target ({target['node_id']}) share with high jitter: {share:.1%}")
    print("  (should be lower than baseline if jw is elevated)")


@pytest.mark.asyncio
async def test_jitter_recovery(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Routing after tc qdisc is removed — jitter EMA decays over ~30 probe cycles.

    Remove tc qdisc first:  sudo tc qdisc del dev lo root
    Then wait ~30 seconds before running.
    """
    workers = await cluster.wait_for_workers(min_count=2)
    await cluster.reset_scheduler()
    await cluster.set_weights(queue=0.0, prefix_miss=0.0, memory=0.0, jitter=1.0, thermal=0.0)

    print(f"\n  [{SCENARIO}] Waiting 30s for jitter EMA to decay…")
    await asyncio.sleep(30.0)
    _log_jitter(await cluster.get_workers())

    records = await send_batch(
        relay_client, short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO, phase="recovery", concurrency=CONCURRENCY,
    )
    _save_and_plot(records, run_dir)
    _print_distribution(records, workers, "recovery")


def _log_jitter(workers: list[Any]) -> None:
    for w in workers:
        t = w.get("telemetry", {})
        print(f"  [{SCENARIO}] {w['node_id']}  jw={t.get('jw', '?'):.2f}ms")


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
