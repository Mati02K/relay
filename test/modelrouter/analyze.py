"""Turn the ablation CSV into a per-config summary table + matplotlib plots.

Reads ``bench/results/ablation.csv`` (or whatever was passed in), computes
TTFT and total-latency percentiles per ``run_label``, prints a Markdown-ready
table to stdout, and writes:

  * ``ttft_p50_p90_p99.png``     — bar chart of p50/p90/p99 TTFT by config
  * ``worker_distribution.png``  — stacked bar of requests per worker by config
  * ``rejection_rate.png``       — bar chart of % rejected (for the SLO config)
  * ``summary.md``               — the same table the report can paste verbatim

We use only matplotlib + the stdlib so this script runs without pandas.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ORDER = [
    "rr", "no_cache", "no_jitter", "no_thermal",
    "full", "full_slo", "full_slo_tight", "routellm_nu",
    "full_gemma", "routellm_nu_gemma", "routellm_nu_gemma_strong",
]


def _pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile that gracefully handles tiny lists."""
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _load(path: Path) -> dict[str, list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_label[row["run_label"]].append(row)
    return by_label


def _summary_row(label: str, rows: list[dict]) -> dict:
    """Compute the per-config metrics the report needs."""
    ok = [r for r in rows if r["status"] == "200" and r["rejected"] != "True"]
    rejected = sum(1 for r in rows if r["rejected"] == "True")
    failed = sum(1 for r in rows if r["status"] != "200" and r["rejected"] != "True")
    ttfts = [float(r["ttft_ms"]) for r in ok]
    totals = [float(r["total_ms"]) for r in ok]
    decode_speeds = []
    for r in ok:
        try:
            ct = int(r["completion_tokens"])
            dec_ms = float(r["decode_ms"])
            if ct > 0 and dec_ms > 0 and not math.isnan(dec_ms):
                decode_speeds.append(ct / (dec_ms / 1000.0))
        except (ValueError, ZeroDivisionError):
            pass
    workers: dict[str, int] = defaultdict(int)
    for r in ok:
        workers[r["worker"] or "?"] += 1
    return {
        "label": label,
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_rejected": rejected,
        "n_failed": failed,
        "ttft_p50": _pct(ttfts, 50),
        "ttft_p90": _pct(ttfts, 90),
        "ttft_p99": _pct(ttfts, 99),
        "total_p50": _pct(totals, 50),
        "total_p90": _pct(totals, 90),
        "decode_toks_s_mean": statistics.fmean(decode_speeds) if decode_speeds else float("nan"),
        "workers": dict(workers),
    }


def _ordered_labels(seen: list[str]) -> list[str]:
    """Place known configs in the canonical order, then anything new alphabetically."""
    known = [c for c in CONFIG_ORDER if c in seen]
    extras = sorted([c for c in seen if c not in CONFIG_ORDER])
    return known + extras


def _print_table(summaries: list[dict]) -> str:
    """Build a Markdown table the report can copy directly."""
    lines = []
    lines.append(
        "| config | n_ok | rejected | ttft p50 | ttft p90 | ttft p99 | total p50 | decode tok/s | worker mix |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in summaries:
        mix = ", ".join(f"{k}:{v}" for k, v in sorted(s["workers"].items()))
        lines.append(
            f"| {s['label']} | {s['n_ok']} | {s['n_rejected']} | "
            f"{s['ttft_p50']:.0f} | {s['ttft_p90']:.0f} | {s['ttft_p99']:.0f} | "
            f"{s['total_p50']:.0f} | {s['decode_toks_s_mean']:.1f} | {mix} |"
        )
    text = "\n".join(lines)
    print(text)
    return text


def _plot_ttft(summaries: list[dict], out: Path) -> None:
    labels = [s["label"] for s in summaries]
    p50 = [s["ttft_p50"] for s in summaries]
    p90 = [s["ttft_p90"] for s in summaries]
    p99 = [s["ttft_p99"] for s in summaries]
    x = range(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width for i in x], p50, width, label="p50", color="#4c78a8")
    ax.bar(list(x), p90, width, label="p90", color="#f58518")
    ax.bar([i + width for i in x], p99, width, label="p99", color="#e45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT distribution by scheduler configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_workers(summaries: list[dict], out: Path) -> None:
    worker_ids = sorted({w for s in summaries for w in s["workers"]})
    labels = [s["label"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = [0.0] * len(summaries)
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#9d755d"]
    for i, w in enumerate(worker_ids):
        values = [s["workers"].get(w, 0) for s in summaries]
        ax.bar(labels, values, bottom=bottom, label=w, color=palette[i % len(palette)])
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_ylabel("Requests routed")
    ax.set_title("Per-worker request distribution by configuration")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_rejection(summaries: list[dict], out: Path) -> None:
    labels = [s["label"] for s in summaries]
    rates = [
        (s["n_rejected"] / s["n_total"] * 100.0) if s["n_total"] else 0.0 for s in summaries
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, rates, color="#e45756")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Rejected (%)")
    ax.set_title("SLO-driven admission rejection by configuration")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("bench/results/ablation.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("bench/results"))
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"no CSV at {args.csv}")
        return 2

    by_label = _load(args.csv)
    labels = _ordered_labels(list(by_label.keys()))
    summaries = [_summary_row(l, by_label[l]) for l in labels]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = _print_table(summaries)
    (args.out_dir / "summary.md").write_text(md + "\n")

    _plot_ttft(summaries, args.out_dir / "ttft_p50_p90_p99.png")
    _plot_workers(summaries, args.out_dir / "worker_distribution.png")
    _plot_rejection(summaries, args.out_dir / "rejection_rate.png")
    print(f"\nplots written to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
