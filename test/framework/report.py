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


def plot_signal_outcome_bars(
    values: dict[str, float],
    noise: float,
    output_dir: Path,
    *,
    filename: str,
    title: str,
    ylabel: str,
) -> Path:
    """Sorted bar chart of a per-signal outcome metric with a noise-floor line.

    Bars above ``noise`` are green (a real effect beyond run-to-run wobble); bars
    below it are grey (indistinguishable from noise). Used for the standalone-vs-
    round-robin latency improvement, where higher = the signal helped more.
    """
    plt = _plt()
    out = output_dir / filename
    if not values:
        return out

    items = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = ["#4caf50" if v > noise else "#bdbdbd" for v in vals]
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(
        noise,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"noise floor (±{noise:.0f} ms)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:+.0f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cumulative_metric(
    labels: list[str],
    values: list[float],
    output_dir: Path,
    *,
    filename: str,
    title: str,
    ylabel: str,
) -> Path:
    """Line chart of one metric as signals are switched on cumulatively over round-robin.

    Used for both P90 TTFT (should staircase down) and throughput (should step up)
    as each signal is added — a flat segment means that signal added nothing on top
    of the earlier ones.
    """
    plt = _plt()
    out = output_dir / filename
    if not labels:
        return out

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(labels, values, marker="o", linewidth=2, color="#1f77b4")
    for x, y in zip(labels, values, strict=True):
        ax.text(
            x, y, f"{y:.0f}" if abs(y) >= 10 else f"{y:.2f}", ha="center", va="bottom", fontsize=9
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Signals enabled (cumulative)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_pareto_signals(
    impacts: dict[str, float],
    output_dir: Path,
    *,
    filename: str = "signal_pareto.png",
) -> Path:
    """Pareto chart: each signal's 0–1 impact, sorted high→low, with a cumulative line.

    Answers "which of the cost-function signals carries the routing?" — the 80/20
    view. ``impacts`` maps signal name → score in [0, 1] (e.g. the leave-one-out
    routing shift from ``test_signal_importance``).
    """
    plt = _plt()
    out = output_dir / filename
    if not impacts:
        return out

    items = sorted(impacts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values) or 1.0
    cumulative = []
    running = 0.0
    for v in values:
        running += v
        cumulative.append(running / total * 100)

    fig, ax1 = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    bars = ax1.bar(names, values, color=[colors[i % len(colors)] for i in range(len(names))])
    ax1.set_ylabel("Routing impact (0–1, measured in each signal's scenario)")
    ax1.set_ylim(0, 1.05)
    for bar, v in zip(bars, values, strict=True):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(names, cumulative, color="#c0392b", marker="o", linewidth=2, label="cumulative %")
    ax2.set_ylabel("Cumulative share of total impact (%)", color="#c0392b")
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    for x, c in zip(names, cumulative, strict=True):
        ax2.text(x, c + 2, f"{c:.0f}%", ha="center", color="#c0392b", fontsize=8)

    ax1.set_title("Which signal matters? — per-signal routing impact (Pareto)")
    fig.text(
        0.5,
        0.01,
        "Conditional on the pressure each signal targets being present when its scenario ran.",
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


_SIGNAL_LABELS = {
    "mw": "Memory pressure  m_w",
    "theta_w": "Thermal pressure  θ_w",
    "jw": "Jitter  j_w (ms)",
    "qw": "Queue depth  q_w",
}


def plot_signal_over_time(
    samples: list[tuple[float, dict[str, float]]],
    field: str,
    scenario: str,
    output_dir: Path,
) -> Path:
    """Line chart of one telemetry field per worker over the whole test duration.

    ``samples`` is ``[(elapsed_s, {node_id: value}), ...]`` as collected by
    :class:`framework.baseline.TelemetrySampler`. Shows that the pressured node's
    signal stayed elevated for the run, next to the routing-share result.
    """
    plt = _plt()
    out = output_dir / f"{scenario}_{field}_over_time.png"
    if not samples:
        return out

    nodes = sorted({n for _, row in samples for n in row})
    times = [t for t, _ in samples]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]
    for i, node in enumerate(nodes):
        ys = [row.get(node, 0.0) for _, row in samples]
        ax.plot(times, ys, label=node, linewidth=2, color=colors[i % len(colors)])

    ax.set_xlabel("Time since test start (s)")
    ax.set_ylabel(_SIGNAL_LABELS.get(field, field))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"{scenario.replace('_', ' ').title()} — {_SIGNAL_LABELS.get(field, field)} per worker"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Worker", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


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
            {worker: count / total for worker, count in counts.items()} if total else {}
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
        "Prefix Cache Affinity — Worker per Turn per Conversation\n(same color = same worker)"
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
        f"Prefix Cache Affinity per Conversation\nOverall: {result['overall_affinity'] * 100:.1f}%"
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
    buckets = [
        (0.0, 0.25, "Low\n(0–0.25)"),
        (0.25, 0.5, "Med-Low\n(0.25–0.5)"),
        (0.5, 0.75, "Med-High\n(0.5–0.75)"),
        (0.75, 1.01, "High\n(0.75–1.0)"),
    ]

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


def plot_routing_by_category(
    records: list[RoutingRecord],
    output_dir: Path,
) -> Path:
    """Heatmap of routing share per (prompt category x worker).

    Rows are prompt categories, columns are workers; each cell is the fraction
    of that category's successful requests that landed on the worker. A correct
    router lights up one cell per row (each category → its intended model).
    """
    import numpy as np

    plt = _plt()
    ok = [r for r in records if not r.error and r.category and r.worker]

    preferred_order = ("coding", "reasoning", "chat", "trivial")
    present = {r.category for r in ok}
    categories = [c for c in preferred_order if c in present]
    categories += sorted(c for c in present if c not in preferred_order)
    workers = sorted({r.worker for r in ok})

    fig, ax = plt.subplots(
        figsize=(max(_FIG_W, len(workers) * 1.6), max(4.0, len(categories) * 0.9))
    )
    if categories and workers:
        matrix = np.zeros((len(categories), len(workers)))
        for i, cat in enumerate(categories):
            subset = [r for r in ok if r.category == cat]
            for j, worker in enumerate(workers):
                matrix[i, j] = worker_share(subset, worker)

        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(workers)))
        ax.set_xticklabels(workers, rotation=20, ha="right")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        for i in range(len(categories)):
            for j in range(len(workers)):
                share = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{share * 100:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if share > 0.55 else "black",
                    fontsize=10,
                )
        fig.colorbar(im, ax=ax, label="share of category's requests")
    else:
        ax.text(0.5, 0.5, "no routed requests", ha="center", va="center")

    ax.set_title("Routing share by prompt category")
    fig.tight_layout()

    out = output_dir / "routing_by_category.png"
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
        axes[1].bar([xi + offset for xi in x], vals, w, label=label, color=colors[pi])
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
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=bin_labels,
        yticklabels=workers,
        cmap="Blues",
        cbar_kws={"label": "Requests"},
    )
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
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            annotation,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_ylabel("Failed requests (503 / timeout)")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.set_title(f"{scenario.replace('_', ' ').title()} — Failed Requests per Run")
    fig.tight_layout()

    out = output_dir / f"{scenario}_failures.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ── TTFT percentiles ──────────────────────────────────────────────────────────


def plot_latency_percentiles_comparison(
    records_by_label: dict[str, list[RoutingRecord]],
    output_dir: Path,
    scenario: str,
) -> Path:
    """Grouped bar chart of P50/P90/P99 TTFT for each labelled run (lower = better).

    Only successful requests count toward the percentiles (errored requests have
    no real TTFT). This is where a routing signal usually shows up — the tail
    (P99) shrinks when traffic is steered off slow/loaded workers.
    """
    plt = _plt()
    labels = list(records_by_label.keys())
    pcts = [(50, "P50"), (90, "P90"), (99, "P99")]
    colors = ["#1976D2", "#D32F2F", "#388E3C", "#7B1FA2"]

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    x = range(len(pcts))
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        ttfts = [r.ttft_ms for r in records_by_label[label] if r.error is None]
        values = [_percentile(ttfts, pct) for pct, _ in pcts]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x], values, width, label=label, color=colors[i % len(colors)]
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([name for _, name in pcts])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"{scenario.replace('_', ' ').title()} — TTFT percentiles (lower is better)")
    ax.legend(title="Run")
    fig.tight_layout()

    out = output_dir / f"{scenario}_ttft_percentiles.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ── Chaos / worker churn ──────────────────────────────────────────────────────

_MODE_LABELS = {"cost": "Relay (cost)", "round_robin": "Round-robin"}
_MODE_COLORS = {"cost": "#1976D2", "round_robin": "#C62828"}


def _chaos_overlay(
    series_by_mode: dict[str, tuple[list[float], list[float]]],
    events: list[tuple[float, str]],
    out: Path,
    *,
    title: str,
    ylabel: str,
    step: bool = False,
    ymax: float | None = None,
) -> Path:
    """Overlay one metric per scheduler mode over run time, with kill/restore lines."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    for mode, (xs, ys) in series_by_mode.items():
        color = _MODE_COLORS.get(mode, None)
        label = _MODE_LABELS.get(mode, mode)
        if step:
            ax.step(xs, ys, where="post", label=label, linewidth=2, color=color)
        else:
            ax.plot(xs, ys, label=label, linewidth=2, color=color)

    for t, action in events:
        ax.axvline(t, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(
            t,
            0.98,
            action,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="dimgray",
        )

    ax.set_xlabel("Time since run start (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Scheduler")
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_chaos_throughput(
    series_by_mode: dict[str, tuple[list[float], list[float]]],
    events: list[tuple[float, str]],
    output_dir: Path,
) -> Path:
    """Throughput (successful req/s) over time, Relay vs round-robin overlaid."""
    return _chaos_overlay(
        series_by_mode,
        events,
        output_dir / "chaos_throughput.png",
        title="Worker churn — throughput over time",
        ylabel="Successful requests/s",
    )


def plot_chaos_failure_rate(
    series_by_mode: dict[str, tuple[list[float], list[float]]],
    events: list[tuple[float, str]],
    output_dir: Path,
) -> Path:
    """Failure rate (errored/total per bucket) over time, both modes overlaid."""
    return _chaos_overlay(
        series_by_mode,
        events,
        output_dir / "chaos_failure_rate.png",
        title="Worker churn — failure rate over time",
        ylabel="Failure rate",
        ymax=1.0,
    )


def plot_chaos_worker_count(
    series_by_mode: dict[str, tuple[list[float], list[float]]],
    events: list[tuple[float, str]],
    output_dir: Path,
) -> Path:
    """Healthy worker count over time, both modes overlaid (step chart)."""
    return _chaos_overlay(
        series_by_mode,
        events,
        output_dir / "chaos_worker_count.png",
        title="Worker churn — healthy workers over time",
        ylabel="Healthy workers",
        step=True,
    )


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
