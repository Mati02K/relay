"""Quality routing (RouteLLM / nu term) scenario.

Verifies that when ``nu > 0`` and workers have different ``model_quality``
scores, the coordinator routes high-complexity prompts to the stronger worker
more often than low-complexity prompts.

Setup
-----
Two workers are required.  The test boosts one worker's preference to act as
the "strong" worker (high quality) and penalises the other as "weak".  With
``nu = 5.0``, the quality term ``nu * complexity * (1 - model_quality)``
should dominate for high-complexity prompts.

Because the model_quality comes from the model chart (coordinator-side
lookup), we approximate "strong worker" bias by using the per-worker weight
override API instead of changing the chart — giving the strong worker a
large positive preference and the weak worker a large negative one.

Complexity tiers
----------------
low     Simple factual questions (complexity ≈ 0.1–0.2)
medium  Standard explanations (complexity ≈ 0.3–0.5)
high    Expert-level math/coding/reasoning (complexity ≈ 0.6–0.9)
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
    plot_quality_routing_bins,
    plot_quality_routing_scatter,
)
from framework.workload import complexity_tier_prompts, send_batch

SCENARIO = "quality_routing"
CONCURRENCY = 4


@pytest.mark.asyncio
async def test_quality_routing_sends_hard_prompts_to_strong_worker(
    cluster: ClusterClient,
    relay_client: RelayClient,
    run_dir: Path,
) -> None:
    """High-complexity prompts preferentially route to the designated strong worker."""
    workers = await cluster.wait_for_workers(min_count=2)
    sorted_workers = sorted(workers, key=lambda w: w["node_id"])
    strong = sorted_workers[0]
    weak = sorted_workers[1]
    strong_id = strong["node_id"]

    # Boost strong worker preference substantially and penalise weak worker
    # so the quality term has a clear target.
    await cluster.reset_scheduler()
    await cluster.set_weights(nu=5.0, queue=0.0, prefix_miss=0.0, memory=0.0, jitter=0.0, thermal=0.0)
    await cluster.set_worker_weight_override(strong_id, 0.8)
    await cluster.set_worker_weight_override(weak["node_id"], -0.8)
    await cluster.wait_telemetry_propagation()

    tiers = complexity_tier_prompts()
    all_records = []

    for tier_name, prompts in tiers.items():
        records = await send_batch(
            relay_client,
            prompts,
            scenario=SCENARIO,
            phase=tier_name,
            concurrency=CONCURRENCY,
            max_tokens=128,
        )
        all_records.extend(records)
        share = worker_share(records, strong_id)
        print(
            f"  [{SCENARIO}] tier={tier_name:8s}  n={len(records)}  "
            f"strong_worker_share={share:.1%}"
        )

    # Cleanup
    await cluster.reset_scheduler()
    await cluster.clear_all_worker_weight_overrides()

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plot_quality_routing_scatter(all_records, plots_dir)
    plot_quality_routing_bins(all_records, strong_id, plots_dir)

    high_records = [r for r in all_records if r.phase == "high"]
    low_records = [r for r in all_records if r.phase == "low"]

    high_strong_share = worker_share(high_records, strong_id)
    low_strong_share = worker_share(low_records, strong_id)

    assert high_strong_share > low_strong_share + 0.15, (
        f"Expected high-complexity prompts to route to strong worker more than low-complexity. "
        f"High: {high_strong_share:.1%}, Low: {low_strong_share:.1%}"
    )

    print(
        f"\n[{SCENARIO}] PASS — low_tier_strong={low_strong_share:.1%}  "
        f"high_tier_strong={high_strong_share:.1%}  "
        f"delta={high_strong_share - low_strong_share:.1%}"
    )
