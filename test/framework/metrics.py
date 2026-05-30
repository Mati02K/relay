"""Routing record aggregation and statistics."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from framework.client import RoutingRecord


# ── Per-phase aggregation ─────────────────────────────────────────────────────

def phase_stats(records: list[RoutingRecord], phase: str) -> dict[str, Any]:
    """Compute summary statistics for all records in a given phase.

    Returns a dict with latency percentiles, worker distribution, error count,
    and the dominant worker (highest share).
    """
    subset = [r for r in records if r.phase == phase]
    if not subset:
        return {"phase": phase, "count": 0}

    ttfts = [r.ttft_ms for r in subset]
    workers = [r.worker for r in subset]
    worker_counts: dict[str, int] = defaultdict(int)
    for w in workers:
        worker_counts[w] += 1
    total = len(subset)
    worker_shares = {w: c / total for w, c in worker_counts.items()}
    errors = sum(1 for r in subset if r.error)

    return {
        "phase": phase,
        "count": total,
        "errors": errors,
        "ttft_p50_ms": _percentile(ttfts, 50),
        "ttft_p90_ms": _percentile(ttfts, 90),
        "ttft_p99_ms": _percentile(ttfts, 99),
        "ttft_mean_ms": statistics.mean(ttfts),
        "worker_counts": dict(worker_counts),
        "worker_shares": worker_shares,
        "dominant_worker": max(worker_counts, key=lambda w: worker_counts[w]),
    }


def all_phase_stats(records: list[RoutingRecord]) -> list[dict[str, Any]]:
    """Return phase stats for every distinct phase present in ``records``."""
    phases = list(dict.fromkeys(r.phase for r in records))
    return [phase_stats(records, p) for p in phases]


def worker_share(records: list[RoutingRecord], node_id: str) -> float:
    """Return the fraction of ``records`` routed to ``node_id``."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.worker == node_id) / len(records)


# ── Prefix-cache affinity ─────────────────────────────────────────────────────

def conversation_affinity(records: list[RoutingRecord]) -> dict[str, Any]:
    """Compute per-conversation and overall prefix-cache affinity rates.

    Affinity for a conversation is the fraction of turns (excluding turn 1,
    the cold start) that land on the same worker as turn 1.
    """
    by_conv: dict[str, list[RoutingRecord]] = defaultdict(list)
    for r in records:
        if r.conversation_id:
            by_conv[r.conversation_id].append(r)

    per_conv: dict[str, float] = {}
    for conv_id, conv_records in by_conv.items():
        sorted_turns = sorted(conv_records, key=lambda r: r.turn)
        if len(sorted_turns) < 2:
            per_conv[conv_id] = 1.0
            continue
        anchor = sorted_turns[0].worker
        warm_turns = sorted_turns[1:]
        per_conv[conv_id] = sum(1 for r in warm_turns if r.worker == anchor) / len(warm_turns)

    overall = statistics.mean(per_conv.values()) if per_conv else 0.0
    return {
        "overall_affinity": overall,
        "per_conversation": per_conv,
        "n_conversations": len(per_conv),
        "n_records": len(records),
    }


# ── Throughput ────────────────────────────────────────────────────────────────

def throughput_rps(records: list[RoutingRecord]) -> float:
    """Estimate requests per second from wall-clock timestamps."""
    if len(records) < 2:
        return 0.0
    t_min = min(r.timestamp for r in records)
    t_max = max(r.timestamp + r.total_ms / 1000.0 for r in records)
    duration = t_max - t_min
    return len(records) / duration if duration > 0 else 0.0


def latency_cdf(records: list[RoutingRecord]) -> dict[str, list[float]]:
    """Return sorted TTFT values and CDF fractions for plotting."""
    ttfts = sorted(r.ttft_ms for r in records)
    n = len(ttfts)
    cdf = [(i + 1) / n for i in range(n)]
    return {"ttft_ms": ttfts, "cdf": cdf}


# ── CSV / JSON output ─────────────────────────────────────────────────────────

def save_records_csv(records: list[RoutingRecord], path: Path) -> None:
    """Write all records to a CSV file."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].as_dict().keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.as_dict())


def save_records_json(records: list[RoutingRecord], path: Path) -> None:
    """Write all records to a JSON file."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.as_dict() for r in records], indent=2))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
