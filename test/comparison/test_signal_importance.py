"""Signal importance under normal mixed load — round-robin baseline, two passes.

No injected pressure. Every machine just does its normal work; the cluster's own
heterogeneity (fast GPU node vs slow CPU node, LAN vs Tailscale hops) makes queue
depth, memory, thermal and jitter vary across workers organically. We send ONE
realistic mixed workload — multi-turn conversations (KV/prefix cache) + long prompts
(memory) + normal short prompts (general queue load) — and compare scheduler configs
against blind round-robin.

Why round-robin as the baseline: comparing a signal against *uniform* round-robin is
a much bigger contrast than removing it from an all-on set, so effects are far easier
to see above run-to-run noise.

Pass 1 — standalone (each signal's own steering power)
    round_robin, then queue-only, prefix-only, memory-only, jitter-only, thermal-only.
    importance(s) = how much s ALONE moves routing away from round-robin (TVD) and how
    much it lowers P90 TTFT vs round-robin.

Pass 2 — cumulative (marginal value of adding each signal)
    round_robin → queue → +prefix → +memory → +jitter → +thermal (all five).
    Each step vs the previous = what that signal adds on top of what's already on.
    (queue-only is shared with Pass 1, so it is not re-run.)

Outcome metric is P90 TTFT (P99 is too tail-unstable at these sample sizes). Two
round-robin runs establish the routing + latency noise floors; only effects beyond
the floor count.

No coordinator/worker/scheduler changes: this only flips scheduler mode/weights via
the existing control API.

Tuning (env)
------------
RELAY_TEST_IMPORTANCE_CONVS     conversations per run (KV/prefix)   default 8
RELAY_TEST_IMPORTANCE_LONG      long memory-heavy prompts per run   default 12
RELAY_TEST_IMPORTANCE_NORMAL    normal short prompts per run        default 20
RELAY_TEST_IMPORTANCE_CONC      single-turn concurrency             default 10
RELAY_TEST_IMPORTANCE_CONV_CONC conversation concurrency            default 4
RELAY_TEST_IMPORTANCE_MAX_TOKENS  tokens per request                default 64
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from framework.client import RelayClient, RoutingRecord
from framework.cluster import ClusterClient
from framework.metrics import (
    _percentile,
    routing_shift,
    save_records_csv,
    save_records_json,
    throughput_rps,
)
from framework.report import (
    plot_cumulative_metric,
    plot_pareto_signals,
    plot_signal_outcome_bars,
)
from framework.workload import replay_conversations_concurrent, send_batch

N_CONVS = int(os.getenv("RELAY_TEST_IMPORTANCE_CONVS", "8"))
N_LONG = int(os.getenv("RELAY_TEST_IMPORTANCE_LONG", "12"))
N_NORMAL = int(os.getenv("RELAY_TEST_IMPORTANCE_NORMAL", "20"))
CONCURRENCY = int(os.getenv("RELAY_TEST_IMPORTANCE_CONC", "10"))
CONV_CONCURRENCY = int(os.getenv("RELAY_TEST_IMPORTANCE_CONV_CONC", "4"))
MAX_TOKENS = int(os.getenv("RELAY_TEST_IMPORTANCE_MAX_TOKENS", "64"))

# Order signals are turned on (standalone) and added (cumulative).
SIGNALS = ["queue", "prefix_miss", "memory", "jitter", "thermal"]


def _only(signal: str) -> dict[str, float]:
    """Cost weights with a single signal at 1.0, everything else off."""
    return {s: (1.0 if s == signal else 0.0) for s in SIGNALS} | {"nu": 0.0}


def _first_k(k: int) -> dict[str, float]:
    """Cost weights with the first ``k`` signals (in SIGNALS order) at 1.0."""
    return {s: (1.0 if i < k else 0.0) for i, s in enumerate(SIGNALS)} | {"nu": 0.0}


def _prompt_len(prompt: list[dict[str, str]]) -> int:
    """Total character length of a single-turn prompt (proxy for context size)."""
    return sum(len(m.get("content", "")) for m in prompt)


@pytest.mark.asyncio
async def test_signal_importance_under_normal_load(
    cluster: ClusterClient,
    relay_client: RelayClient,
    mixed_prompts: list[Any],
    conversations: list[Any],
    run_dir: Path,
) -> None:
    """Round-robin baseline; rank signals standalone and cumulatively under mixed load."""
    await cluster.wait_for_workers(min_count=2)

    # Runs: rr_a, rr_b, 5 standalone, 4 cumulative steps (queue-only shared) = 11.
    n_runs = 2 + len(SIGNALS) + (len(SIGNALS) - 1)
    assert len(conversations) >= n_runs * N_CONVS, (
        f"Need {n_runs * N_CONVS} conversations, got {len(conversations)}. Lower "
        f"RELAY_TEST_IMPORTANCE_CONVS (max {len(conversations) // n_runs})."
    )
    need_single = n_runs * (N_LONG + N_NORMAL)
    assert len(mixed_prompts) >= need_single, (
        f"Need {need_single} single-turn prompts, got {len(mixed_prompts)}. Lower "
        f"RELAY_TEST_IMPORTANCE_LONG / _NORMAL."
    )

    # Longest mixed prompts → memory-heavy pool; the rest → normal pool.
    by_len = sorted(mixed_prompts, key=_prompt_len, reverse=True)
    long_pool = by_len[: n_runs * N_LONG]
    normal_pool = by_len[n_runs * N_LONG : n_runs * N_LONG + n_runs * N_NORMAL]
    conv_pool = conversations[: n_runs * N_CONVS]
    run_idx = 0

    async def run(label: str, mode: str, weights: dict[str, float] | None) -> list[RoutingRecord]:
        nonlocal run_idx
        i = run_idx
        run_idx += 1
        convs = conv_pool[i * N_CONVS : (i + 1) * N_CONVS]
        singles = (
            long_pool[i * N_LONG : (i + 1) * N_LONG]
            + normal_pool[i * N_NORMAL : (i + 1) * N_NORMAL]
        )
        await cluster.full_reset()
        if mode == "round_robin":
            await cluster.set_mode("round_robin")
        else:
            await cluster.set_mode("cost")
            await cluster.set_weights(**(weights or {}))
        await cluster.wait_telemetry_propagation()
        print(f"[importance] run '{label}'  mode={mode}  weights={weights or '-'}")
        conv_recs, single_recs = await asyncio.gather(
            replay_conversations_concurrent(
                relay_client, convs, scenario="importance", phase=label,
                concurrency=CONV_CONCURRENCY, max_tokens=MAX_TOKENS,
            ),
            send_batch(
                relay_client, singles, scenario="importance", phase=label,
                concurrency=CONCURRENCY, max_tokens=MAX_TOKENS,
            ),
        )
        return conv_recs + single_recs

    def p90(records: list[RoutingRecord]) -> float:
        return _percentile([r.ttft_ms for r in records if r.error is None], 90)

    def p99(records: list[RoutingRecord]) -> float:
        return _percentile([r.ttft_ms for r in records if r.error is None], 99)

    # ── Baseline: two round-robin runs → reference + noise floors ─────────────
    rr_a = await run("round_robin_a", "round_robin", None)
    rr_b = await run("round_robin_b", "round_robin", None)
    tvd_noise = routing_shift(rr_a, rr_b)
    p90_rr = (p90(rr_a) + p90(rr_b)) / 2
    p90_noise = abs(p90(rr_a) - p90(rr_b))
    rps_rr = (throughput_rps(rr_a) + throughput_rps(rr_b)) / 2
    rps_noise = abs(throughput_rps(rr_a) - throughput_rps(rr_b))

    # ── Pass 1: each signal alone vs round-robin ──────────────────────────────
    standalone_recs: dict[str, list[RoutingRecord]] = {}
    standalone_routing: dict[str, float] = {}
    standalone_p90_gain: dict[str, float] = {}
    standalone_rps_gain: dict[str, float] = {}
    standalone_p99: dict[str, float] = {}
    for s in SIGNALS:
        recs = await run(f"only_{s}", "cost", _only(s))
        standalone_recs[s] = recs
        standalone_routing[s] = max(0.0, routing_shift(rr_a, recs) - tvd_noise)
        standalone_p90_gain[s] = p90_rr - p90(recs)  # +ve = faster than round-robin
        standalone_rps_gain[s] = throughput_rps(recs) - rps_rr  # +ve = more throughput
        standalone_p99[s] = p99(recs)

    # ── Pass 2: cumulative build-up (queue-only reused from Pass 1) ────────────
    cumulative_labels = ["round_robin"]
    cumulative_p90 = [p90_rr]
    cumulative_rps = [rps_rr]
    cumulative_marginal: list[dict[str, Any]] = []
    prev_recs = rr_a
    prev_p90 = p90_rr
    prev_rps = rps_rr
    for k in range(1, len(SIGNALS) + 1):
        added = SIGNALS[k - 1]
        if k == 1:
            recs = standalone_recs[added]  # queue-only already run in Pass 1
        else:
            recs = await run(f"cum_{k}_{added}", "cost", _first_k(k))
        step_p90, step_rps = p90(recs), throughput_rps(recs)
        cumulative_labels.append(f"+{added}")
        cumulative_p90.append(step_p90)
        cumulative_rps.append(step_rps)
        cumulative_marginal.append({
            "added": added,
            "routing_shift_vs_prev": round(routing_shift(prev_recs, recs), 4),
            "p90_gain_vs_prev_ms": round(prev_p90 - step_p90, 1),
            "rps_gain_vs_prev": round(step_rps - prev_rps, 2),
            "p90_ms": round(step_p90, 1),
            "p99_ms": round(p99(recs), 1),
            "rps": round(step_rps, 2),
        })
        prev_recs, prev_p90, prev_rps = recs, step_p90, step_rps

    await cluster.full_reset()

    # ── Persist + plot ────────────────────────────────────────────────────────
    all_records = rr_a + rr_b + [r for recs in standalone_recs.values() for r in recs]
    save_records_csv(all_records, run_dir / "importance_records.csv")
    save_records_json(all_records, run_dir / "importance_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_pareto_signals(standalone_routing, plots_dir, filename="standalone_routing_pareto.png")
    plot_signal_outcome_bars(
        standalone_p90_gain, p90_noise, plots_dir,
        filename="standalone_latency_gain.png",
        title="Each signal alone vs round-robin — P90 latency reduction (mixed load)",
        ylabel="P90 TTFT reduction vs round-robin (ms) — higher = faster",
    )
    plot_signal_outcome_bars(
        standalone_rps_gain, rps_noise, plots_dir,
        filename="standalone_throughput_gain.png",
        title="Each signal alone vs round-robin — throughput gain (mixed load)",
        ylabel="Requests/sec gain vs round-robin — higher = more throughput",
    )
    plot_cumulative_metric(
        cumulative_labels, cumulative_p90, plots_dir,
        filename="cumulative_latency.png",
        title="Cumulative effect: P90 latency as each signal is added over round-robin",
        ylabel="P90 TTFT (ms) — lower is better",
    )
    plot_cumulative_metric(
        cumulative_labels, cumulative_rps, plots_dir,
        filename="cumulative_throughput.png",
        title="Cumulative effect: throughput as each signal is added over round-robin",
        ylabel="Requests/sec — higher is better",
    )

    def verdict(s: str) -> str:
        steers = standalone_routing[s] > 0
        helps = standalone_p90_gain[s] > p90_noise or standalone_rps_gain[s] > rps_noise
        if steers and helps:
            return "NEEDED"
        if steers:
            return "steers, no outcome gain"
        return "no effect"

    result = {
        "baseline": {"p90_rr_ms": round(p90_rr, 1), "rps_rr": round(rps_rr, 2)},
        "noise": {
            "routing_tvd": round(tvd_noise, 4),
            "p90_ms": round(p90_noise, 1),
            "rps": round(rps_noise, 2),
        },
        "standalone": {
            s: {
                "routing_vs_rr": round(standalone_routing[s], 4),
                "p90_gain_vs_rr_ms": round(standalone_p90_gain[s], 1),
                "rps_gain_vs_rr": round(standalone_rps_gain[s], 2),
                "p99_ms": round(standalone_p99[s], 1),
                "verdict": verdict(s),
            }
            for s in SIGNALS
        },
        "cumulative": cumulative_marginal,
        "workload_per_run": {"conversations": N_CONVS, "long": N_LONG, "normal": N_NORMAL},
    }
    (run_dir / "signal_importance.json").write_text(json.dumps(result, indent=2))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n[importance] baseline round-robin  P90={p90_rr:.0f}ms  RPS={rps_rr:.1f}  "
          f"noise: routing={tvd_noise:.3f}  p90=±{p90_noise:.0f}ms  rps=±{rps_noise:.1f}")
    print("\n[importance] PASS 1 — each signal alone vs round-robin:")
    print("    signal         routing   ΔP90(ms)   ΔRPS    P99(ms)   verdict")
    for s in sorted(SIGNALS, key=lambda s: standalone_routing[s], reverse=True):
        print(f"    {s:14} {standalone_routing[s]:6.3f}  {standalone_p90_gain[s]:+8.0f}  "
              f"{standalone_rps_gain[s]:+6.1f}  {standalone_p99[s]:8.0f}   {verdict(s)}")
    print("\n[importance] PASS 2 — cumulative (marginal value of adding each):")
    for step in cumulative_marginal:
        print(f"    +{step['added']:12} routing+{step['routing_shift_vs_prev']:.3f}   "
              f"ΔP90 {step['p90_gain_vs_prev_ms']:+.0f}ms   ΔRPS {step['rps_gain_vs_prev']:+.1f}   "
              f"(P90={step['p90_ms']:.0f} RPS={step['rps']:.1f})")
    winners = [s for s in SIGNALS if verdict(s) == "NEEDED"]
    print(f"\n[importance] earns-its-place (standalone): {winners or 'none beat the noise floor'}")

    # At least one signal must move routing past the noise floor, else the workload
    # produced no organic variation for any term to act on.
    assert max(standalone_routing.values()) > 0.0, (
        "No signal moved routing past round-robin beyond the noise floor — the mixed "
        "workload produced no organic queue/prefix/memory/jitter/thermal variation. "
        "Increase concurrency or the conversation/long-prompt mix and retry."
    )
