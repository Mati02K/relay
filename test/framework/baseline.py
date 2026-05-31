"""Shared round-robin-vs-signal A/B harness for scenario tests.

Every signal scenario uses the same shape: run a workload once under the blind
round-robin scheduler (baseline), then once with a single cost-function weight
enabled (the signal under test), and compare. This module owns the mode/weight
switching and full state reset so each scenario only supplies its workload.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from framework.client import RoutingRecord
from framework.cluster import ClusterClient

BASELINE_PHASE = "round_robin"

WorkloadFn = Callable[[str], Awaitable[list[RoutingRecord]]]


async def run_round_robin_vs_signal(
    cluster: ClusterClient,
    run_workload: WorkloadFn,
    *,
    signal_phase: str,
    signal_weights: dict[str, float],
) -> tuple[list[RoutingRecord], list[RoutingRecord]]:
    """Run ``run_workload`` under round-robin, then under one enabled signal.

    ``run_workload(phase)`` sends the scenario's traffic and returns the records;
    it is called once with :data:`BASELINE_PHASE` and once with ``signal_phase``.
    ``signal_weights`` is the full cost-weight set for the signal run (the term
    under test at 1.0, everything else 0). Returns ``(baseline, signal)``.

    A full reset (weights, mode, and per-worker overrides) runs before each phase
    and at the end, so leftover state from a prior test can never leak in.
    """
    # ── Baseline: blind round-robin ───────────────────────────────────────────
    await cluster.full_reset()
    await cluster.set_mode("round_robin")
    await cluster.wait_telemetry_propagation()
    baseline = await run_workload(BASELINE_PHASE)

    # ── Signal on: cost mode with only this term's weight ─────────────────────
    await cluster.full_reset()
    await cluster.set_mode("cost")
    await cluster.set_weights(**signal_weights)
    await cluster.wait_telemetry_propagation()
    signal = await run_workload(signal_phase)

    await cluster.full_reset()
    return baseline, signal
