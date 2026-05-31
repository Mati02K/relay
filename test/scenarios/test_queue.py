"""Queue routing under overload: round-robin baseline vs the queue signal.

Sends the same heavy burst through the coordinator twice — once under blind
round-robin, once with only ``queue=1`` — and compares how many requests fail
(503 / client timeout). Round-robin keeps shovelling work onto an already-busy
worker; queue-aware routing steers to whichever worker has free slots, so it
should drop no more requests (we expect fewer). A short per-request timeout
turns "overloaded" into "failed fast" so the test stays bounded.

Tuning (env)
------------
RELAY_TEST_OVERLOAD_REQUESTS     burst size per run (default 60; max 500 — needs
                                 2× from the 1000-prompt mixed pool)
RELAY_TEST_OVERLOAD_CONCURRENCY  in-flight requests (default 30)
RELAY_TEST_OVERLOAD_MAX_TOKENS   tokens per request (default 128)
RELAY_TEST_OVERLOAD_TIMEOUT      per-request timeout seconds (default 30)
RELAY_TEST_OVERLOAD_SETTLE       max seconds to wait for queues to drain (default 60)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from framework.baseline import BASELINE_PHASE, run_round_robin_vs_signal
from framework.client import RelayClient, RoutingRecord
from framework.cluster import ClusterClient
from framework.metrics import _percentile, save_records_csv, save_records_json, worker_share
from framework.report import plot_failure_counts, plot_worker_distribution_phases
from framework.workload import send_batch

SCENARIO = "queue"
SIGNAL_PHASE = "queue_on"
OVERLOAD_REQUESTS = int(os.getenv("RELAY_TEST_OVERLOAD_REQUESTS", "60"))
OVERLOAD_CONCURRENCY = int(os.getenv("RELAY_TEST_OVERLOAD_CONCURRENCY", "30"))
OVERLOAD_MAX_TOKENS = int(os.getenv("RELAY_TEST_OVERLOAD_MAX_TOKENS", "128"))
OVERLOAD_TIMEOUT = float(os.getenv("RELAY_TEST_OVERLOAD_TIMEOUT", "30"))
SETTLE_SECONDS = float(os.getenv("RELAY_TEST_OVERLOAD_SETTLE", "60"))
SIGNAL_WEIGHTS = {
    "queue": 1.0, "prefix_miss": 0.0, "memory": 0.0, "jitter": 0.0, "thermal": 0.0, "nu": 0.0,
}


@pytest.mark.asyncio
async def test_queue_routing_vs_round_robin(
    cluster: ClusterClient,
    coordinator_url: str,
    mixed_prompts: list[Any],
    run_dir: Path,
) -> None:
    """queue=1 drops no more requests under overload than blind round-robin."""
    workers = await cluster.wait_for_workers(min_count=2)

    # Disjoint prompt slices per run so neither rides the other's warm prefix
    # cache. Drawn from the 1000-prompt mixed pool so large bursts still fit.
    need = OVERLOAD_REQUESTS * 2
    assert len(mixed_prompts) >= need, (
        f"Need {need} prompts for two disjoint slices, got {len(mixed_prompts)} "
        f"(max RELAY_TEST_OVERLOAD_REQUESTS = {len(mixed_prompts) // 2})."
    )
    subset_a = mixed_prompts[:OVERLOAD_REQUESTS]
    subset_b = mixed_prompts[OVERLOAD_REQUESTS:need]

    # Short timeout so queued-too-long requests fail fast and are counted,
    # instead of the suite hanging on the default 120s.
    client = RelayClient(coordinator_url, timeout=OVERLOAD_TIMEOUT)
    try:
        async def run_workload(phase: str) -> list[RoutingRecord]:
            prompts = subset_a if phase == BASELINE_PHASE else subset_b
            await _wait_idle(cluster)
            records = await send_batch(
                client, prompts, scenario=SCENARIO, phase=phase,
                concurrency=OVERLOAD_CONCURRENCY, max_tokens=OVERLOAD_MAX_TOKENS,
            )
            _report(phase, records, workers, _failures(records))
            await _wait_idle(cluster)
            return records

        baseline, signal = await run_round_robin_vs_signal(
            cluster, run_workload, signal_phase=SIGNAL_PHASE, signal_weights=SIGNAL_WEIGHTS,
        )
    finally:
        await client.aclose()

    fails_rr = _failures(baseline)
    fails_sig = _failures(signal)

    all_records = baseline + signal
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_worker_distribution_phases(all_records, SCENARIO, plots_dir)
    plot_failure_counts(
        {"round_robin": fails_rr, "queue_on (queue=1)": fails_sig},
        plots_dir, SCENARIO, total=OVERLOAD_REQUESTS,
    )

    if fails_rr == 0:
        print(
            f"\n[{SCENARIO}] NOTE: round-robin produced 0 failures — the burst was not "
            f"heavy enough to overload a worker. Raise RELAY_TEST_OVERLOAD_REQUESTS / "
            f"_CONCURRENCY / _MAX_TOKENS or lower _TIMEOUT."
        )
    assert fails_sig <= fails_rr, (
        f"Queue-aware routing failed more requests than blind round-robin. "
        f"round_robin={fails_rr}, queue_on={fails_sig} (expected queue_on <= round_robin)"
    )
    print(
        f"\n[{SCENARIO}] PASS — failed requests: round_robin={fails_rr}  queue_on={fails_sig}  "
        f"(reduced by {fails_rr - fails_sig})"
    )


def _failures(records: list[RoutingRecord]) -> int:
    """Count requests that errored (HTTP 503, client timeout, transport failure)."""
    return sum(1 for r in records if r.error)


async def _wait_idle(cluster: ClusterClient, timeout: float = SETTLE_SECONDS) -> None:
    """Wait until every worker's queue depth drains to zero, or until ``timeout``."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        workers = await cluster.get_workers()
        max_qw = max((w.get("telemetry", {}).get("qw", 0) for w in workers), default=0)
        if max_qw == 0 or asyncio.get_event_loop().time() >= deadline:
            return
        await asyncio.sleep(1.0)


def _report(phase: str, records: list[RoutingRecord], workers: list[Any], failures: int) -> None:
    """Print failures, per-worker routing share, and P99 TTFT for one run."""
    ttfts = [r.ttft_ms for r in records if r.error is None]
    p99 = _percentile(ttfts, 99) if ttfts else 0.0
    print(f"\n  [{SCENARIO}] phase={phase}  n={len(records)}  failures={failures}  P99={p99:.0f}ms")
    for w in workers:
        print(f"    {w['node_id']}: {worker_share(records, w['node_id']):.1%}")
