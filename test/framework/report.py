"""Plot generation and JSON summary writer.

All plots are saved as 150-dpi PNGs.  The functions are intentionally
independent — each takes only the data it needs — so callers can invoke them
selectively without running the full suite.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from framework.client import RoutingRecord
from framework.metrics import (
    _percentile,
    conversation_affinity,
    latency_cdf,
    throughput_rps,
    worker_share,
)

_DPI = 150
_FIG_W = 10
_FIG_H = 6


def _plt():  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": _DPI, "font.size": 11})
    return plt


def _sns():  # type: ignore[no-untyped-def]
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return sns


# ── Scheduler signal tests ────────────────────────────────────────────────────

def plot_worker_distribution_phases(
    records: list[RoutingRecord],
    scenario: str,
    output_dir: Path,
) -> Path:
    """Grouped bar chart: successful routed-request worker share per phase."""
    plt = _plt()
    phases = list(dict.fromkeys(r.phase for r in records))
    if not phases:
        return output_dir / f"{scenario}_worker_distribution.png"

    shares_by_phase: dict[str, dict[str, float]] = {}
    for phase in phases:
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            if record.phase == phase and record.worker:
                counts[record.worker] += 1
        total = sum(counts.values())
        shares_by_phase[phase] = (
            {worker: count / total for worker, count in counts.items()}
            if total
            else {}
        )

    all_workers = sorted({w for shares in shares_by_phase.values() for w in shares})
    x = range(len(phases))
    width = 0.8 / max(len(all_workers), 1)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    for i, worker in enumerate(all_workers):
        shares = [shares_by_phase[phase].get(worker, 0) * 100 for phase in phases]
        offset = (i - len(all_workers) / 2 + 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            shares,
            width,
            label=worker,
            color=colors[i % len(colors)],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(phases, rotation=15, ha="right")
    ax.set_ylabel("Routed request share (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"{scenario.replace('_', ' ').title()} — Worker Distribution per Phase")
    if all_workers:
        ax.legend(title="Worker", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    out = output_dir / f"{scenario}_worker_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_latency_boxes_phases(
    records: list[RoutingRecord],
    scenario: str,
    output_dir: Path,
) -> Path:
    """Box plot of TTFT per phase."""
    plt = _plt()
    phases = list(dict.fromkeys(r.phase for r in records))
    data = [[r.ttft_ms for r in records if r.phase == p] for p in phases]
    data = [d for d in data if d]
    phases_filtered = [p for p, d in zip(phases, data) if d]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.boxplot(data, tick_labels=phases_filtered, patch_artist=True)
    ax.set_xlabel("Phase")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"{scenario.replace('_', ' ').title()} — TTFT Distribution per Phase")
    fig.tight_layout()

    out = output_dir / f"{scenario}_latency_boxes.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_pressure_vs_routing(
    phase_names: list[str],
    pressure_values: list[float | None],
    target_shares: list[float],
    scenario: str,
    pressure_label: str,
    output_dir: Path,
) -> Path:
    """Line chart: pressure value vs target-worker routing share."""
    plt = _plt()
    fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))

    valid_pressures = [p if p is not None else 0.0 for p in pressure_values]
    ax1.plot(
        phase_names,
        [s * 100 for s in target_shares],
        "o-b",
        linewidth=2,
        label="Worker share %",
    )
    ax1.set_ylabel("Target worker routing share (%)", color="blue")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(phase_names, valid_pressures, "s--r", linewidth=2, label=pressure_label)
    ax2.set_ylabel(pressure_label, color="red")
    ax2.set_ylim(-0.05, 1.1)
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_title(f"{scenario.replace('_', ' ').title()} — {pressure_label} vs Routing Share")
    ax1.set_xlabel("Phase")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()

    out = output_dir / f"{scenario}_pressure_vs_routing.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Prefix cache ──────────────────────────────────────────────────────────────

def plot_prefix_cache_heatmap(
    records: list[RoutingRecord],
    output_dir: Path,
) -> Path:
    """Heatmap: conversation × turn → worker index."""
    sns = _sns()
    plt = _plt()

    by_conv: dict[str, dict[int, str]] = defaultdict(dict)
    for r in records:
        if r.conversation_id:
            by_conv[r.conversation_id][r.turn] = r.worker

    if not by_conv:
        return output_dir / "prefix_cache_heatmap.png"

    conv_ids = sorted(by_conv.keys())
    all_workers = sorted({w for turns in by_conv.values() for w in turns.values()})
    worker_idx = {w: i for i, w in enumerate(all_workers)}
    max_turn = max(t for turns in by_conv.values() for t in turns)

    import numpy as np  # type: ignore[import-untyped]

    matrix = np.full((len(conv_ids), max_turn), np.nan)
    for ci, conv_id in enumerate(conv_ids):
        for turn, worker in by_conv[conv_id].items():
            if 1 <= turn <= max_turn:
                matrix[ci, turn - 1] = worker_idx.get(worker, -1)

    fig, ax = plt.subplots(figsize=(min(max_turn * 0.6 + 2, 18), min(len(conv_ids) * 0.35 + 2, 16)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Set2",
        cbar_kws={"label": "Worker index"},
        linewidths=0.3,
        linecolor="lightgray",
    )
    ax.set_xlabel("Turn")
    ax.set_ylabel("Conversation")
    ax.set_title(
        "Prefix Cache Affinity — Worker per Turn per Conversation\n"
        "(same color = same worker)"
    )

    out = output_dir / "prefix_cache_heatmap.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_affinity_bars(
    records: list[RoutingRecord],
    output_dir: Path,
) -> Path:
    """Horizontal bar chart: per-conversation affinity rate."""
    plt = _plt()
    result = conversation_affinity(records)
    per_conv = result["per_conversation"]
    if not per_conv:
        return output_dir / "prefix_cache_affinity.png"

    conv_ids = sorted(per_conv.keys())
    affinities = [per_conv[c] * 100 for c in conv_ids]

    fig, ax = plt.subplots(figsize=(_FIG_W, max(4, len(conv_ids) * 0.25 + 2)))
    colors = ["green" if a >= 90 else "orange" if a >= 70 else "red" for a in affinities]
    ax.barh(conv_ids, affinities, color=colors)
    ax.axvline(90, color="navy", linestyle="--", linewidth=1.5, label="90% threshold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Same-worker rate (%)")
    ax.set_title(
        f"Prefix Cache Affinity per Conversation\n"
        f"Overall: {result['overall_affinity'] * 100:.1f}%"
    )
    ax.legend()
    fig.tight_layout()

    out = output_dir / "prefix_cache_affinity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Quality routing ───────────────────────────────────────────────────────────

def plot_quality_routing_scatter(
    records: list[RoutingRecord],
    output_dir: Path,
) -> Path:
    """Scatter plot: prompt complexity vs worker."""
    plt = _plt()

    workers = sorted({r.worker for r in records if r.worker})
    colors = plt.cm.Set1.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    rng = __import__("random").Random(42)
    for i, worker in enumerate(workers):
        subset = [r for r in records if r.worker == worker]
        xs = [r.complexity for r in subset]
        jitter = [rng.uniform(-0.05, 0.05) for _ in subset]
        ys = [i + j for j in jitter]
        ax.scatter(xs, ys, alpha=0.6, color=colors[i % len(colors)], label=worker, s=40)

    ax.set_yticks(range(len(workers)))
    ax.set_yticklabels(workers)
    ax.set_xlabel("Request complexity (0 = simple, 1 = expert)")
    ax.set_title("Quality Routing — Complexity vs Worker")
    ax.legend(title="Worker", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    out = output_dir / "quality_routing_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_quality_routing_bins(
    records: list[RoutingRecord],
    strong_worker: str,
    output_dir: Path,
) -> Path:
    """Bar chart: strong-worker routing share per complexity bucket."""
    plt = _plt()
    buckets = [(0.0, 0.25, "Low\n(0–0.25)"), (0.25, 0.5, "Med-Low\n(0.25–0.5)"),
               (0.5, 0.75, "Med-High\n(0.5–0.75)"), (0.75, 1.01, "High\n(0.75–1.0)")]

    shares = []
    labels = []
    for lo, hi, label in buckets:
        bucket_records = [r for r in records if lo <= r.complexity < hi]
        share = worker_share(bucket_records, strong_worker) * 100 if bucket_records else 0.0
        shares.append(share)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = ["#4caf50" if s >= 60 else "#ff9800" if s >= 40 else "#f44336" for s in shares]
    ax.bar(labels, shares, color=colors, edgecolor="white")
    ax.axhline(50, color="navy", linestyle="--", linewidth=1.5, label="Baseline (50%)")
    ax.set_ylabel(f"Routing share to strong worker ({strong_worker}) (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Quality Routing — Strong-Worker Share per Complexity Bucket")
    ax.legend()
    fig.tight_layout()

    out = output_dir / "quality_routing_bins.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Baseline comparison ───────────────────────────────────────────────────────

def plot_latency_cdf_comparison(
    records_by_config: dict[str, list[RoutingRecord]],
    output_dir: Path,
) -> Path:
    """CDF of TTFT for multiple configurations on one chart."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    styles = ["-", "--", "-."]
    colors = ["#1976D2", "#D32F2F", "#388E3C"]

    for i, (config, records) in enumerate(records_by_config.items()):
        if not records:
            continue
        cdf_data = latency_cdf(records)
        ax.plot(
            cdf_data["ttft_ms"],
            cdf_data["cdf"],
            linestyle=styles[i % len(styles)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=config,
        )

    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("TTFT CDF — Relay vs Baselines")
    ax.legend(title="Scheduler mode")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    out = output_dir / "latency_cdf_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_throughput_comparison(
    records_by_config: dict[str, list[RoutingRecord]],
    output_dir: Path,
) -> Path:
    """Bar chart comparing throughput and P-percentile latency across configs."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    configs = list(records_by_config.keys())
    colors = ["#1976D2", "#D32F2F", "#388E3C", "#7B1FA2"]

    # RPS
    rps_vals = [throughput_rps(records_by_config[c]) for c in configs]
    axes[0].bar(configs, rps_vals, color=[colors[i % len(colors)] for i in range(len(configs))])
    axes[0].set_ylabel("Requests per second")
    axes[0].set_title("Throughput Comparison")
    for i, v in enumerate(rps_vals):
        axes[0].text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # P50/P90/P99 latency grouped bar
    x = range(len(configs))
    w = 0.25
    for pi, (pct, label) in enumerate([(50, "P50"), (90, "P90"), (99, "P99")]):
        vals = [_percentile([r.ttft_ms for r in records_by_config[c]], pct) for c in configs]
        offset = (pi - 1) * w
        axes[1].bar([xi + offset for xi in x], vals, w, label=label,
                    color=colors[pi])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(configs)
    axes[1].set_ylabel("TTFT (ms)")
    axes[1].set_title("Latency Percentiles Comparison")
    axes[1].legend()

    fig.tight_layout()
    out = output_dir / "throughput_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_worker_heatmap_time(
    records: list[RoutingRecord],
    config: str,
    output_dir: Path,
    window_seconds: float = 10.0,
) -> Path:
    """Heat map of worker selection density over the run timeline."""
    sns = _sns()
    plt = _plt()
    import numpy as np  # type: ignore[import-untyped]

    if not records:
        return output_dir / f"{config}_worker_heatmap.png"

    workers = sorted({r.worker for r in records if r.worker})
    t_min = min(r.timestamp for r in records)
    t_max = max(r.timestamp for r in records)
    duration = max(t_max - t_min, 1.0)
    n_bins = max(int(duration / window_seconds), 2)

    matrix = np.zeros((len(workers), n_bins))
    for r in records:
        if not r.worker:
            continue
        wi = workers.index(r.worker) if r.worker in workers else -1
        bi = min(int((r.timestamp - t_min) / duration * n_bins), n_bins - 1)
        if wi >= 0:
            matrix[wi, bi] += 1

    fig, ax = plt.subplots(figsize=(min(n_bins * 0.4 + 2, 18), max(3, len(workers) * 1.2)))
    bin_labels = [f"{i * window_seconds:.0f}s" for i in range(n_bins)]
    sns.heatmap(matrix, ax=ax, xticklabels=bin_labels, yticklabels=workers,
                cmap="Blues", cbar_kws={"label": "Requests"})
    ax.set_xlabel(f"Time window ({window_seconds:.0f}s bins)")
    ax.set_title(f"Worker Selection Over Time — {config}")
    fig.tight_layout()

    out = output_dir / f"{config}_worker_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Failure comparison ────────────────────────────────────────────────────────

def plot_failure_counts(
    counts: dict[str, int],
    output_dir: Path,
    scenario: str,
    total: int | None = None,
) -> Path:
    """Bar chart comparing failed-request counts (503/timeout) across runs."""
    plt = _plt()
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = ["#f44336" if v > 0 else "#4caf50" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, value in zip(bars, values):
        annotation = f"{value}" + (f" / {total}" if total else "")
        ax.text(bar.get_x() + bar.get_width() / 2, value, annotation,
                ha="center", va="bottom", fontsize=12)

    ax.set_ylabel("Failed requests (503 / timeout)")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.set_title(f"{scenario.replace('_', ' ').title()} — Failed Requests per Run")
    fig.tight_layout()

    out = output_dir / f"{scenario}_failures.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Summary JSON ──────────────────────────────────────────────────────────────

def save_summary(
    run_dir: Path,
    scenario_results: dict[str, Any],
    git_sha: str = "unknown",
    coordinator_url: str = "",
) -> Path:
    """Write ``summary.json`` to ``run_dir``."""
    import datetime

    summary = {
        "run_id": run_dir.name,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_sha": git_sha,
        "coordinator": coordinator_url,
        "scenarios": scenario_results,
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return out
