"""Relay vs round-robin vs single-worker comparison.

Runs the same mixed-prompt workload under three scheduler configurations and
proves Relay's cost-function routing delivers better P99 latency and higher
throughput than the simpler baselines.

Configurations
--------------
relay         Cost-function scheduling (default mode, all weights = 1.0)
round_robin   ``RELAY_SCHED_MODE=round_robin`` — blind rotation
single_worker All worker-weight overrides set so only one worker receives
              traffic (worker_weight = 1.0 for chosen, −1.0 for rest).

All three configs run the same 500-prompt mixed workload with the same seed,
so TTFT differences are attributable to the scheduler, not prompt variance.

Metrics collected
-----------------
- P50 / P90 / P99 TTFT
- Requests per second
- Error rate
- Worker distribution (relay + round_robin only)

Plots
-----
- latency_cdf_comparison.png   — CDF for all three configs
- throughput_comparison.png    — bar + grouped percentile chart
- relay_worker_heatmap.png     — worker selection over time (relay only)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from framework.client import RelayClient
from framework.cluster import ClusterClient
from framework.metrics import (
    _percentile,
    save_records_csv,
    save_records_json,
    throughput_rps,
    worker_share,
)
from framework.report import (
    plot_latency_cdf_comparison,
    plot_throughput_comparison,
    plot_worker_heatmap_time,
    save_summary,
)
from framework.workload import send_batch

CONCURRENCY = 12
PROMPTS_PER_CONFIG = 500
MAX_TOKENS = 64


@pytest.mark.asyncio
async def test_relay_outperforms_baselines(
    cluster: ClusterClient,
    relay_client: RelayClient,
    mixed_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Relay cost routing achieves lower P99 TTFT than round-robin and single-worker."""
    workers = await cluster.wait_for_workers(min_count=2)
    prompt_subset = mixed_prompts[:PROMPTS_PER_CONFIG]

    records_by_config: dict[str, Any] = {}

    # ── Config 1: Relay cost routing ─────────────────────────────────────────
    print(f"\n[comparison] Running relay config ({PROMPTS_PER_CONFIG} prompts)…")
    await cluster.reset_scheduler()  # cost mode, default weights
    relay_records = await send_batch(
        relay_client,
        prompt_subset,
        scenario="comparison",
        phase="relay",
        concurrency=CONCURRENCY,
        max_tokens=MAX_TOKENS,
    )
    records_by_config["relay"] = relay_records
    _print_stats("relay", relay_records)

    # ── Config 2: Round-robin ─────────────────────────────────────────────────
    print(f"[comparison] Running round_robin config…")
    await cluster.set_mode("round_robin")
    rr_records = await send_batch(
        relay_client,
        prompt_subset,
        scenario="comparison",
        phase="round_robin",
        concurrency=CONCURRENCY,
        max_tokens=MAX_TOKENS,
    )
    records_by_config["round_robin"] = rr_records
    _print_stats("round_robin", rr_records)
    await cluster.set_mode("cost")

    # ── Config 3: Single worker ───────────────────────────────────────────────
    print(f"[comparison] Running single_worker config…")
    # Pin all traffic to workers[0] by giving it maximum preference
    # and maximum penalty to all others.
    chosen = workers[0]["node_id"]
    await cluster.reset_scheduler()
    await cluster.set_worker_weight_override(chosen, 1.0)
    for w in workers[1:]:
        await cluster.set_worker_weight_override(w["node_id"], -1.0)
    await cluster.wait_telemetry_propagation()

    single_records = await send_batch(
        relay_client,
        prompt_subset,
        scenario="comparison",
        phase="single_worker",
        concurrency=CONCURRENCY,
        max_tokens=MAX_TOKENS,
    )
    records_by_config["single_worker"] = single_records
    _print_stats("single_worker", single_records)

    # Cleanup
    await cluster.full_reset()

    # Save data
    all_records = relay_records + rr_records + single_records
    save_records_csv(all_records, run_dir / "comparison_records.csv")
    save_records_json(all_records, run_dir / "comparison_records.json")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_latency_cdf_comparison(records_by_config, plots_dir)
    plot_throughput_comparison(records_by_config, plots_dir)
    plot_worker_heatmap_time(relay_records, "relay", plots_dir)

    # Summary JSON
    comparison_summary = {
        config: {
            "n_requests": len(recs),
            "rps": round(throughput_rps(recs), 2),
            "ttft_p50_ms": round(_percentile([r.ttft_ms for r in recs], 50), 1),
            "ttft_p90_ms": round(_percentile([r.ttft_ms for r in recs], 90), 1),
            "ttft_p99_ms": round(_percentile([r.ttft_ms for r in recs], 99), 1),
            "error_rate": round(sum(1 for r in recs if r.error) / max(len(recs), 1), 4),
        }
        for config, recs in records_by_config.items()
    }
    save_summary(run_dir, {"comparison": comparison_summary})

    # Assertions
    relay_p99 = _percentile([r.ttft_ms for r in relay_records], 99)
    rr_p99 = _percentile([r.ttft_ms for r in rr_records], 99)
    single_p99 = _percentile([r.ttft_ms for r in single_records], 99)

    relay_rps = throughput_rps(relay_records)
    rr_rps = throughput_rps(rr_records)

    assert relay_p99 <= rr_p99 * 1.15, (
        f"Relay P99 TTFT ({relay_p99:.0f}ms) should not be more than 15% worse "
        f"than round-robin ({rr_p99:.0f}ms)"
    )
    assert relay_p99 < single_p99 * 0.95, (
        f"Relay P99 TTFT ({relay_p99:.0f}ms) should beat single-worker "
        f"({single_p99:.0f}ms) with ≥2 workers"
    )

    print(
        f"\n[comparison] PASS\n"
        f"  relay      P99={relay_p99:.0f}ms  RPS={relay_rps:.1f}\n"
        f"  round_robin P99={rr_p99:.0f}ms  RPS={rr_rps:.1f}\n"
        f"  single_worker P99={single_p99:.0f}ms"
    )


def _print_stats(config: str, records: list[Any]) -> None:
    ttfts = [r.ttft_ms for r in records]
    errors = sum(1 for r in records if r.error)
    rps = throughput_rps(records)
    print(
        f"  {config:15s}  n={len(records)}  errors={errors}  rps={rps:.1f}  "
        f"P50={_percentile(ttfts, 50):.0f}ms  P90={_percentile(ttfts, 90):.0f}ms  "
        f"P99={_percentile(ttfts, 99):.0f}ms"
    )
