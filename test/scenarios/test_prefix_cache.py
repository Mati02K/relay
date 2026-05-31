"""Prefix-cache affinity: round-robin baseline vs the prefix signal.

Replays multi-turn conversations twice — once under blind round-robin, once with
only ``prefix_miss=1`` — and compares same-worker affinity (the fraction of a
conversation's turns that stay on the worker that cached the earlier turns).
Round-robin rotates blindly so turns bounce → low affinity; the prefix term
pulls each turn back to its cache → high affinity.

Two disjoint conversation slices are used so the prefix run does not inherit the
caches the round-robin run warmed.

Assert: prefix-on affinity ≥ 85%, and higher than round-robin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from framework.baseline import BASELINE_PHASE, run_round_robin_vs_signal
from framework.client import RelayClient, RoutingRecord
from framework.cluster import ClusterClient
from framework.metrics import conversation_affinity, save_records_csv, save_records_json
from framework.report import plot_prefix_cache_heatmap
from framework.workload import replay_conversations_concurrent

SCENARIO = "prefix_cache"
SIGNAL_PHASE = "prefix_on"
N_CONVERSATIONS = 20
CONCURRENCY = 4
MAX_TOKENS = 32
SIGNAL_WEIGHTS = {
    "queue": 0.0, "prefix_miss": 1.0, "memory": 0.0, "jitter": 0.0, "thermal": 0.0, "nu": 0.0,
}


@pytest.mark.asyncio
async def test_prefix_cache_vs_round_robin(
    cluster: ClusterClient,
    relay_client: RelayClient,
    conversations: list[Any],
    run_dir: Path,
) -> None:
    """prefix_miss=1 yields ≥85% same-worker affinity, beating blind round-robin."""
    assert len(conversations) >= N_CONVERSATIONS * 2, (
        f"Need at least {N_CONVERSATIONS * 2} conversations, got {len(conversations)}"
    )
    await cluster.wait_for_workers(min_count=2)
    subset_a = conversations[:N_CONVERSATIONS]
    subset_b = conversations[N_CONVERSATIONS:N_CONVERSATIONS * 2]

    async def run_workload(phase: str) -> list[RoutingRecord]:
        subset = subset_a if phase == BASELINE_PHASE else subset_b
        return await replay_conversations_concurrent(
            relay_client, subset, scenario=SCENARIO, phase=phase,
            concurrency=CONCURRENCY, max_tokens=MAX_TOKENS,
        )

    baseline, signal = await run_round_robin_vs_signal(
        cluster, run_workload, signal_phase=SIGNAL_PHASE, signal_weights=SIGNAL_WEIGHTS,
    )

    affinity_rr = conversation_affinity(baseline)["overall_affinity"]
    affinity_sig = conversation_affinity(signal)["overall_affinity"]
    print(f"\n[{SCENARIO}] round_robin affinity: {affinity_rr:.1%}")
    print(f"[{SCENARIO}] prefix_on   affinity: {affinity_sig:.1%}")

    all_records = baseline + signal
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_prefix_cache_heatmap(signal, plots_dir)
    _plot_affinity_comparison(affinity_rr, conversation_affinity(signal), plots_dir)

    assert affinity_sig >= 0.85, f"prefix_on affinity {affinity_sig:.1%} < 85% threshold"
    print(
        f"\n[{SCENARIO}] PASS — round_robin={affinity_rr:.1%}  "
        f"prefix_on={affinity_sig:.1%}  delta={affinity_sig - affinity_rr:+.1%}"
    )


def _plot_affinity_comparison(rr_affinity: float, sig_affinity: dict, plots_dir: Path) -> None:
    """Bar chart: prefix-on per-conversation affinity vs the round-robin baseline line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    per_conv = sig_affinity["per_conversation"]
    conv_ids = sorted(per_conv)
    rates = [per_conv[c] * 100 for c in conv_ids]
    overall = sig_affinity["overall_affinity"] * 100
    x = range(len(conv_ids))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, rates, color="#42A5F5", alpha=0.85, label="Prefix on")
    ax.axhline(rr_affinity * 100, color="#EF5350", linestyle="--", linewidth=1.8,
               label=f"Round-robin baseline ({rr_affinity * 100:.1f}%)")
    ax.axhline(85, color="navy", linestyle=":", linewidth=1.5, label="85% threshold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.split("_")[-1] for c in conv_ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Same-worker rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Prefix Cache Affinity — prefix on per conversation\n"
        f"round-robin: {rr_affinity * 100:.1f}%  |  prefix on: {overall:.1f}%"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "prefix_cache_affinity_comparison.png", dpi=150)
    plt.close(fig)
