"""Queue depth routing scenario.

Verifies that when a worker's queue (``qw``) fills up with in-flight requests,
new incoming requests are routed to other workers.

Queue saturation is achieved by sending ``n_parallel`` long-output requests
to the target worker simultaneously so they occupy all available inference
slots.  While saturated, short fast requests are dispatched through the
coordinator and their routing distribution is captured.

Phases
------
baseline    no load on any worker
saturated   target worker queue filled with slow long requests
recovery    slow requests complete, queue drains back to 0
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
from framework.workload import send_batch, slow_prompts

SCENARIO = "queue"
REQUESTS_PER_PHASE = 120
CONCURRENCY = 8
# Number of slow parallel requests to flood the target worker's queue
SATURATION_REQUESTS = 12
# How long each slow request should run (large max_tokens to hold the slot)
SLOW_MAX_TOKENS = 256


@pytest.mark.asyncio
async def test_queue_routing_deviates_when_saturated(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Queue saturation on one worker shifts fast requests to other workers."""
    workers = await cluster.wait_for_workers(min_count=2)
    target = workers[0]
    target_id = target["node_id"]
    target_addr = target["address"]

    await cluster.reset_scheduler()
    await cluster.set_weights(queue=1.0, prefix_miss=0.0, memory=0.0, jitter=0.0, thermal=0.0)

    # --- Baseline phase ---
    baseline_records = await send_batch(
        relay_client,
        short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO,
        phase="baseline",
        concurrency=CONCURRENCY,
    )
    baseline_share = worker_share(baseline_records, target_id)
    print(f"  [{SCENARIO}] phase=baseline     target_share={baseline_share:.1%}")

    # --- Saturated phase ---
    # Fire slow requests at the target worker directly (bypass coordinator to
    # guarantee they land on the target). They run in the background while we
    # measure routing of fast requests through the coordinator.
    saturation_client = RelayClient(target_addr)
    saturation_prompts = slow_prompts(SATURATION_REQUESTS)

    saturation_tasks = [
        asyncio.create_task(
            saturation_client.chat_completion(
                p,
                scenario=SCENARIO,
                phase="saturation_filler",
                max_tokens=SLOW_MAX_TOKENS,
            )
        )
        for p in saturation_prompts
    ]
    # Give queue time to fill before measuring routing
    await asyncio.sleep(1.5)

    saturated_records = await send_batch(
        relay_client,
        short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO,
        phase="saturated",
        concurrency=CONCURRENCY,
    )
    saturated_share = worker_share(saturated_records, target_id)
    print(f"  [{SCENARIO}] phase=saturated    target_share={saturated_share:.1%}")

    # Wait for slow requests to finish
    await asyncio.gather(*saturation_tasks, return_exceptions=True)
    await asyncio.sleep(2.0)

    # --- Recovery phase ---
    recovery_records = await send_batch(
        relay_client,
        short_prompts[:REQUESTS_PER_PHASE],
        scenario=SCENARIO,
        phase="recovery",
        concurrency=CONCURRENCY,
    )
    recovery_share = worker_share(recovery_records, target_id)
    print(f"  [{SCENARIO}] phase=recovery     target_share={recovery_share:.1%}")

    all_records = baseline_records + saturated_records + recovery_records

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plot_worker_distribution_phases(all_records, SCENARIO, plots_dir)
    plot_latency_boxes_phases(all_records, SCENARIO, plots_dir)

    assert saturated_share < baseline_share * 0.65, (
        f"Expected queue saturation to reduce routing to target worker. "
        f"Baseline: {baseline_share:.1%}, Saturated: {saturated_share:.1%}"
    )
    assert recovery_share >= baseline_share * 0.55, (
        f"Expected routing to recover after queue drains. "
        f"Baseline: {baseline_share:.1%}, Recovery: {recovery_share:.1%}"
    )

    print(
        f"\n[{SCENARIO}] PASS — baseline={baseline_share:.1%}  "
        f"saturated={saturated_share:.1%}  recovery={recovery_share:.1%}"
    )
