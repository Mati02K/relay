"""Quality routing (RouteLLM nu term): round-robin baseline vs the nu signal.

Sends three complexity tiers (low / medium / high) twice — once under blind
round-robin, once with only ``nu=5`` — and compares how much of each tier lands
on the stronger-model worker. Round-robin is complexity-blind (~even split per
tier); the nu term ``nu * complexity * (1 - quality)`` should push **high**-
complexity prompts toward the higher-quality worker more than low-complexity
ones.

Requires two workers running **different-quality models** (the chart supplies
each model's quality). The "strong" worker is identified empirically as the one
the high tier favors under nu — no worker_weight override is used.

Assert: under nu, high-tier strong-share exceeds low-tier strong-share by ≥15pts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from framework.baseline import BASELINE_PHASE, run_round_robin_vs_signal
from framework.client import RelayClient, RoutingRecord
from framework.cluster import ClusterClient
from framework.metrics import save_records_csv, save_records_json, worker_share
from framework.report import plot_quality_routing_bins, plot_quality_routing_scatter
from framework.workload import complexity_tier_prompts, send_batch

SCENARIO = "quality_routing"
SIGNAL_PHASE = "nu_on"
CONCURRENCY = 4
MAX_TOKENS = 128
TIERS = ("low", "medium", "high")
SIGNAL_WEIGHTS = {
    "queue": 0.0, "prefix_miss": 0.0, "memory": 0.0, "jitter": 0.0, "thermal": 0.0, "nu": 5.0,
}


@pytest.mark.asyncio
async def test_quality_routing_vs_round_robin(
    cluster: ClusterClient,
    relay_client: RelayClient,
    run_dir: Path,
) -> None:
    """Under nu, high-complexity prompts favor the strong worker more than low-complexity."""
    await cluster.wait_for_workers(min_count=2)
    tiers = complexity_tier_prompts()

    async def run_workload(phase: str) -> list[RoutingRecord]:
        records: list[RoutingRecord] = []
        for tier_name in TIERS:
            records += await send_batch(
                relay_client, tiers[tier_name], scenario=SCENARIO, phase=f"{phase}:{tier_name}",
                concurrency=CONCURRENCY, max_tokens=MAX_TOKENS,
            )
        return records

    baseline, signal = await run_round_robin_vs_signal(
        cluster, run_workload, signal_phase=SIGNAL_PHASE, signal_weights=SIGNAL_WEIGHTS,
    )

    def tier_recs(records: list[RoutingRecord], phase: str, tier: str) -> list[RoutingRecord]:
        return [r for r in records if r.phase == f"{phase}:{tier}"]

    # Strong worker = the one the high tier favors under nu.
    worker_ids = {r.worker for r in signal if r.worker}
    nu_high = tier_recs(signal, SIGNAL_PHASE, "high")
    strong_id = max(worker_ids, key=lambda w: worker_share(nu_high, w)) if worker_ids else ""

    print(f"\n[{SCENARIO}] strong worker (high-tier favorite under nu) = {strong_id}")
    print(f"  {'tier':8s}  round_robin→strong   nu→strong")
    for tier_name in TIERS:
        rr = worker_share(tier_recs(baseline, BASELINE_PHASE, tier_name), strong_id)
        nu = worker_share(tier_recs(signal, SIGNAL_PHASE, tier_name), strong_id)
        print(f"  {tier_name:8s}  {rr:>16.1%}   {nu:>9.1%}")

    nu_low_strong = worker_share(tier_recs(signal, SIGNAL_PHASE, "low"), strong_id)
    nu_high_strong = worker_share(nu_high, strong_id)

    all_records = baseline + signal
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_quality_routing_scatter(signal, plots_dir)
    plot_quality_routing_bins(signal, strong_id, plots_dir)

    assert nu_high_strong > nu_low_strong + 0.15, (
        f"Expected high-complexity prompts to favor the strong worker more than low. "
        f"nu high→strong={nu_high_strong:.1%}, nu low→strong={nu_low_strong:.1%}"
    )
    print(
        f"\n[{SCENARIO}] PASS — nu low→strong={nu_low_strong:.1%}  "
        f"nu high→strong={nu_high_strong:.1%}  delta={nu_high_strong - nu_low_strong:+.1%}"
    )
