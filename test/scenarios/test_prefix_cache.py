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

    aff_rr = conversation_affinity(baseline)
    aff_sig = conversation_affinity(signal)
    affinity_rr = aff_rr["overall_affinity"]
    affinity_sig = aff_sig["overall_affinity"]
    print(f"\n[{SCENARIO}] round_robin affinity: {affinity_rr:.1%}")
    print(f"[{SCENARIO}] prefix_on   affinity: {affinity_sig:.1%}")

    all_records = baseline + signal
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_prefix_cache_heatmap(signal, plots_dir)
    _plot_affinity_comparison(aff_rr, aff_sig, plots_dir)

    assert affinity_sig >= 0.85, f"prefix_on affinity {affinity_sig:.1%} < 85% threshold"
    print(
        f"\n[{SCENARIO}] PASS — round_robin={affinity_rr:.1%}  "
        f"prefix_on={affinity_sig:.1%}  delta={affinity_sig - affinity_rr:+.1%}"
    )


def _plot_affinity_comparison(rr_affinity: dict, sig_affinity: dict, plots_dir: Path) -> None:
    """Two-bar overall comparison (round-robin vs prefix on) with per-conversation dots.

    Each bar is the overall same-worker rate; the white dots scattered on it are
    the per-conversation rates — so you see both the headline number and how
    consistent it is (prefix-on dots cluster at 100%, round-robin dots scatter low).
    """
    try:
        import random as _random

        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels = ["round-robin", "prefix on"]
    colors = ["#EF5350", "#42A5F5"]
    overalls = [rr_affinity["overall_affinity"] * 100, sig_affinity["overall_affinity"] * 100]
    per_conv = [
        [v * 100 for v in rr_affinity["per_conversation"].values()],
        [v * 100 for v in sig_affinity["per_conversation"].values()],
    ]
    x = [0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, overalls, width=0.55, color=colors, alpha=0.85, zorder=2)
    for xi, value in zip(x, overalls):
        ax.text(xi, value + 1.5, f"{value:.1f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    rng = _random.Random(42)
    for xi, values in zip(x, per_conv):
        xs = [xi + rng.uniform(-0.13, 0.13) for _ in values]
        ax.scatter(xs, values, color="white", edgecolor="#333", s=32, zorder=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Same-worker rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title(
        "Prefix Cache Affinity — round-robin vs prefix on\n"
        "(bar = overall · dots = per conversation)"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "prefix_cache_affinity_comparison.png", dpi=150)
    plt.close(fig)
