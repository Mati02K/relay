"""Quality routing by prompt category: does each kind of prompt reach the right model?

Sends four prompt categories (trivial / chat / reasoning / coding) twice — once
under blind round-robin, once under the quality signal (``nu`` + ``queue``) —
and checks that routing specializes by category:

* coding prompts concentrate on one worker (the coding model, via the skill
  filter that ``nu > 0`` turns on);
* reasoning prompts favor their strong worker far more than trivial prompts do
  (the ``nu * complexity`` escalation, with ``queue`` making the slow strong
  models "expensive" so trivial prompts fall to the cheap/fast worker).

``nu`` alone always picks the highest-quality model regardless of complexity —
the ``queue`` term is what lets simple prompts go to the cheap worker, so the
signal phase enables both. Produces a category × worker routing heatmap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from framework.baseline import run_round_robin_vs_signal
from framework.client import RelayClient, RoutingRecord
from framework.cluster import ClusterClient
from framework.metrics import save_records_csv, save_records_json, worker_share
from framework.report import plot_routing_by_category
from framework.workload import category_prompts, send_batch

SCENARIO = "quality_routing"
SIGNAL_PHASE = "quality_on"
CONCURRENCY = 4
MAX_TOKENS = 64
CATEGORIES = ("trivial", "chat", "reasoning", "coding")
SIGNAL_WEIGHTS = {
    "queue": 1.0,
    "prefix_miss": 0.0,
    "memory": 0.0,
    "jitter": 0.0,
    "thermal": 0.0,
    "nu": 5.0,
}


@pytest.mark.asyncio
async def test_quality_routing_by_category(
    cluster: ClusterClient,
    relay_client: RelayClient,
    run_dir: Path,
) -> None:
    """Coding concentrates on one worker; reasoning escalates to the strong worker."""
    await cluster.wait_for_workers(min_count=2)
    cats = category_prompts()

    async def run_workload(phase: str) -> list[RoutingRecord]:
        records: list[RoutingRecord] = []
        for category in CATEGORIES:
            records += await send_batch(
                relay_client,
                cats[category],
                scenario=SCENARIO,
                phase=phase,
                concurrency=CONCURRENCY,
                max_tokens=MAX_TOKENS,
                category=category,
            )
        return records

    baseline, signal = await run_round_robin_vs_signal(
        cluster,
        run_workload,
        signal_phase=SIGNAL_PHASE,
        signal_weights=SIGNAL_WEIGHTS,
    )

    def cat_recs(records: list[RoutingRecord], category: str) -> list[RoutingRecord]:
        return [r for r in records if r.category == category and not r.error]

    def favorite(records: list[RoutingRecord]) -> str:
        ids = {r.worker for r in records if r.worker}
        return max(ids, key=lambda w: worker_share(records, w)) if ids else ""

    coding_recs = cat_recs(signal, "coding")
    reasoning_recs = cat_recs(signal, "reasoning")
    coder_id = favorite(coding_recs)
    reason_id = favorite(reasoning_recs)

    coder_share = worker_share(coding_recs, coder_id)
    reason_strong = worker_share(reasoning_recs, reason_id)
    trivial_to_reason = worker_share(cat_recs(signal, "trivial"), reason_id)

    print(f"\n[{SCENARIO}] coding favorite={coder_id}  reasoning favorite={reason_id}")
    print(f"  {'category':10s}  round_robin spread          quality_on spread")
    workers = sorted({r.worker for r in signal if r.worker})
    for category in CATEGORIES:
        rr = {w: f"{worker_share(cat_recs(baseline, category), w):.0%}" for w in workers}
        nu = {w: f"{worker_share(cat_recs(signal, category), w):.0%}" for w in workers}
        print(f"  {category:10s}  {rr}   {nu}")

    all_records = baseline + signal
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_routing_by_category(signal, plots_dir)

    assert coder_share >= 0.7, (
        f"Expected coding prompts to concentrate on one worker (the coding model). "
        f"coding→{coder_id}={coder_share:.1%}"
    )
    assert reason_strong > trivial_to_reason + 0.15, (
        f"Expected reasoning prompts to favor the strong worker more than trivial prompts. "
        f"reasoning→{reason_id}={reason_strong:.1%}, trivial→{reason_id}={trivial_to_reason:.1%}"
    )
    print(
        f"\n[{SCENARIO}] PASS — coding→{coder_id}={coder_share:.1%}  "
        f"reasoning→{reason_id}={reason_strong:.1%}  trivial→{reason_id}={trivial_to_reason:.1%}"
    )
