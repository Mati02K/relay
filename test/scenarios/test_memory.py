"""Memory-pressure routing: round-robin baseline vs the memory signal.

Apply **real** RAM/VRAM pressure to the target worker (workers[0]) and keep it
applied for the whole run. The test sends the same short-prompt workload twice —
once under blind round-robin, once with only ``memory=1`` — and reports how much
traffic each policy sends to the pressured worker. Memory-aware routing should
send noticeably **less** to it.

Apply pressure on the target worker's machine, e.g.:
    stress-ng --vm 1 --vm-bytes 80% --timeout 300s     # RAM
    # or fill VRAM with a GPU workload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from framework.baseline import TelemetrySampler, run_round_robin_vs_signal
from framework.client import RelayClient
from framework.cluster import ClusterClient
from framework.metrics import save_records_csv, save_records_json, worker_share
from framework.report import plot_signal_over_time, plot_worker_distribution_phases
from framework.workload import send_batch

SCENARIO = "memory"
SIGNAL_PHASE = "memory_on"
SIGNAL_FIELD = "mw"
REQUESTS = int(os.getenv("RELAY_TEST_REQUESTS_PER_PHASE", "60"))
CONCURRENCY = 8
MAX_TOKENS = int(os.getenv("RELAY_TEST_MAX_TOKENS", "64"))
SIGNAL_WEIGHTS = {
    "queue": 0.0, "prefix_miss": 0.0, "memory": 1.0, "jitter": 0.0, "thermal": 0.0, "nu": 0.0,
}


@pytest.mark.asyncio
async def test_memory_routing_vs_round_robin(
    cluster: ClusterClient,
    relay_client: RelayClient,
    short_prompts: list[Any],
    run_dir: Path,
) -> None:
    """memory=1 routes less traffic to a pressured worker than blind round-robin."""
    workers = await cluster.wait_for_workers(min_count=2)

    async def run_workload(phase: str) -> Any:
        return await send_batch(
            relay_client, short_prompts[:REQUESTS], scenario=SCENARIO, phase=phase,
            concurrency=CONCURRENCY, max_tokens=MAX_TOKENS,
        )

    async with TelemetrySampler(cluster, SIGNAL_FIELD) as sampler:
        baseline, signal = await run_round_robin_vs_signal(
            cluster, run_workload, signal_phase=SIGNAL_PHASE, signal_weights=SIGNAL_WEIGHTS,
        )

    print(f"\n[{SCENARIO}] worker share  round_robin → {SIGNAL_PHASE}  "
          f"(memory-aware routing sends less to the loaded node):")
    for w in workers:
        nid = w["node_id"]
        rr = worker_share(baseline, nid)
        sig = worker_share(signal, nid)
        print(f"    {nid:16} {rr:6.1%} → {sig:6.1%}   ({sig - rr:+.1%})")

    all_records = baseline + signal
    save_records_csv(all_records, run_dir / f"{SCENARIO}_records.csv")
    save_records_json(all_records, run_dir / f"{SCENARIO}_records.json")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_worker_distribution_phases(all_records, SCENARIO, plots_dir)
    plot_signal_over_time(sampler.samples, SIGNAL_FIELD, SCENARIO, plots_dir)
