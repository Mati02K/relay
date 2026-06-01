"""Shared round-robin-vs-signal A/B harness for scenario tests.

Every signal scenario uses the same shape: run a workload once under the blind
round-robin scheduler (baseline), then once with a single cost-function weight
enabled (the signal under test), and compare. This module owns the mode/weight
switching and full state reset so each scenario only supplies its workload.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from framework.client import RoutingRecord
from framework.cluster import ClusterClient

BASELINE_PHASE = "round_robin"

WorkloadFn = Callable[[str], Awaitable[list[RoutingRecord]]]


class TelemetrySampler:
    """Background sampler of one telemetry field per worker for a test's duration.

    Polls ``GET /v1/workers`` every ``interval`` seconds while active and records
    ``(elapsed_seconds, {node_id: value})`` for ``field`` (``mw``/``theta_w``/
    ``jw``). Use as an async context manager around the workload; afterwards
    :attr:`samples` is the time series to plot — so the run can show that the
    pressured node's signal really was elevated the whole time, not just trusted.
    """

    def __init__(self, cluster: ClusterClient, field: str, interval: float = 1.5) -> None:
        self.cluster = cluster
        self.field = field
        self.interval = interval
        self.samples: list[tuple[float, dict[str, float]]] = []
        self._task: asyncio.Task[None] | None = None
        self._start = 0.0

    async def __aenter__(self) -> TelemetrySampler:
        self._start = time.monotonic()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                workers = await self.cluster.get_workers()
                elapsed = time.monotonic() - self._start
                row = {
                    w["node_id"]: float(w.get("telemetry", {}).get(self.field) or 0.0)
                    for w in workers
                }
                self.samples.append((elapsed, row))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(self.interval)


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
