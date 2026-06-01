"""Worker churn under constant load: Relay cost routing vs round-robin.

Holds a constant open-loop request rate while the operator kills and restarts
workers by hand, then compares how each scheduler reacts. The test does not kill
anything itself — it prints a scripted prompt timeline (``>>> KILL a worker
now`` / ``>>> RESTORE``) at fixed offsets so the same churn happens at the same
moments in both the cost run and the round-robin run, making the throughput
overlay apples-to-apples. It samples healthy-worker count at 1 Hz and buckets
completed requests, then emits three overlaid charts.

Charts (run_dir/plots)
----------------------
- chaos_throughput.png    — successful req/s over time, cost vs round-robin
- chaos_failure_rate.png  — errored/total per bucket over time, both modes
- chaos_worker_count.png  — healthy worker count over time, both modes

Tuning (env)
------------
RELAY_CHAOS_RPS       constant arrival rate, requests/sec (default 5)
RELAY_CHAOS_DURATION  seconds per mode run (default 300)
RELAY_CHAOS_BUCKET    chart bucket width in seconds (default 2)

Because kills are manual, this is observational: it asserts that data was
collected and that churn actually occurred in each run (the operator did their
part), and prints the failure-rate / throughput comparison as the result.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from framework.client import RelayClient
from framework.cluster import ClusterClient
from framework.metrics import bucket_series, save_records_csv, save_records_json
from framework.report import (
    plot_chaos_failure_rate,
    plot_chaos_throughput,
    plot_chaos_worker_count,
    save_summary,
)
from framework.workload import send_constant_rate

SCENARIO = "chaos"
MAX_TOKENS = 64
MODES = ["cost", "round_robin"]


def _rps() -> float:
    return float(os.environ.get("RELAY_CHAOS_RPS", "5"))


def _duration() -> float:
    return float(os.environ.get("RELAY_CHAOS_DURATION", "300"))


def _bucket() -> float:
    return float(os.environ.get("RELAY_CHAOS_BUCKET", "2"))


def _schedule(duration: float) -> list[tuple[float, str]]:
    """Scripted operator timeline: (offset_seconds, short_label) for vlines/prompts."""
    return [
        (round(duration * 0.20), "KILL"),
        (round(duration * 0.40), "KILL"),
        (round(duration * 0.60), "RESTORE"),
        (round(duration * 0.80), "RESTORE"),
    ]


class WorkerCountSampler:
    """Polls healthy-worker count at ``interval`` Hz against a shared clock.

    Records ``(elapsed_seconds, healthy_count)`` using ``run_start`` (a
    ``time.perf_counter()`` reference shared with the load generator) so the
    samples align with the bucketed request series on one time axis.
    """

    def __init__(self, cluster: ClusterClient, run_start: float, interval: float = 1.0) -> None:
        self.cluster = cluster
        self.run_start = run_start
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> WorkerCountSampler:
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
                workers = await self.cluster.get_healthy_workers()
                self.samples.append((time.perf_counter() - self.run_start, len(workers)))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(self.interval)


async def _operator_script(schedule: list[tuple[float, str]], run_start: float) -> None:
    """Print the scripted kill/restore prompts at their scheduled offsets."""
    for offset, action in schedule:
        delay = run_start + offset - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        verb = "kill ONE worker now" if action == "KILL" else "RESTORE one worker now"
        print(f"\n>>> [t={offset:>4}s] {verb}\n", flush=True)


async def _run_mode(
    cluster: ClusterClient,
    client: RelayClient,
    prompts: list[Any],
    *,
    mode: str,
    rps: float,
    duration: float,
    schedule: list[tuple[float, str]],
) -> tuple[list[Any], list[tuple[float, int]], float]:
    """Run one constant-load + sampling pass under ``mode``; return records, samples, run_start."""
    await cluster.full_reset()
    await cluster.set_mode(mode)
    await cluster.wait_telemetry_propagation()

    print(f"\n===== CHAOS RUN: {mode}  ({duration:.0f}s @ {rps:g} rps) =====", flush=True)
    run_start = time.perf_counter()
    sampler = WorkerCountSampler(cluster, run_start)
    async with sampler:
        op = asyncio.create_task(_operator_script(schedule, run_start))
        records = await send_constant_rate(
            client, prompts,
            scenario=SCENARIO, phase=mode,
            rps=rps, duration=duration, run_start=run_start,
            max_tokens=MAX_TOKENS,
        )
        op.cancel()
        try:
            await op
        except asyncio.CancelledError:
            pass

    await cluster.full_reset()
    return records, sampler.samples, run_start


@pytest.mark.asyncio
async def test_worker_churn_resilience(
    cluster: ClusterClient,
    relay_client: RelayClient,
    mixed_prompts: list[Any],
    run_dir: Path,
) -> None:
    """Constant load through worker kills/restores; compare cost vs round-robin."""
    rps, duration, bucket = _rps(), _duration(), _bucket()
    schedule = _schedule(duration)
    await cluster.wait_for_workers(min_count=2)

    print(
        f"\n[chaos] Two {duration:.0f}s runs at {rps:g} rps. Follow the on-screen "
        f"prompts to kill/restore workers at the SAME moments in both runs.",
        flush=True,
    )

    throughput_series: dict[str, tuple[list[float], list[float]]] = {}
    failure_series: dict[str, tuple[list[float], list[float]]] = {}
    worker_series: dict[str, tuple[list[float], list[float]]] = {}
    all_records: list[Any] = []
    summary: dict[str, Any] = {}

    for mode in MODES:
        records, samples, run_start = await _run_mode(
            cluster, relay_client, mixed_prompts,
            mode=mode, rps=rps, duration=duration, schedule=schedule,
        )
        all_records.extend(records)

        times, throughput, failure = bucket_series(records, run_start, duration, bucket)
        throughput_series[mode] = (times, throughput)
        failure_series[mode] = (times, failure)
        worker_series[mode] = ([t for t, _ in samples], [float(c) for _, c in samples])

        errors = sum(1 for r in records if r.error)
        counts = [c for _, c in samples]
        summary[mode] = {
            "n_requests": len(records),
            "errors": errors,
            "failure_rate": round(errors / max(len(records), 1), 4),
            "successful_rps": round((len(records) - errors) / duration, 2),
            "worker_count_start": counts[0] if counts else 0,
            "worker_count_min": min(counts) if counts else 0,
            "worker_count_max": max(counts) if counts else 0,
        }
        print(
            f"  {mode:12s}  n={len(records)}  errors={errors}  "
            f"fail_rate={summary[mode]['failure_rate']:.2%}  "
            f"workers {summary[mode]['worker_count_min']}→{summary[mode]['worker_count_max']}",
            flush=True,
        )

    # ── Save data + plots ─────────────────────────────────────────────────────
    save_records_csv(all_records, run_dir / "chaos_records.csv")
    save_records_json(all_records, run_dir / "chaos_records.json")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    events = [(float(t), label) for t, label in schedule]
    plot_chaos_throughput(throughput_series, events, plots_dir)
    plot_chaos_failure_rate(failure_series, events, plots_dir)
    plot_chaos_worker_count(worker_series, events, plots_dir)

    save_summary(run_dir, {SCENARIO: summary})

    cost = summary.get("cost", {})
    rr = summary.get("round_robin", {})
    print(
        f"\n[chaos] RESULT\n"
        f"  cost        fail_rate={cost.get('failure_rate', 0):.2%}  "
        f"rps={cost.get('successful_rps', 0)}\n"
        f"  round_robin fail_rate={rr.get('failure_rate', 0):.2%}  "
        f"rps={rr.get('successful_rps', 0)}",
        flush=True,
    )

    # ── Observational asserts: data collected and churn actually happened ──────
    for mode in MODES:
        assert summary[mode]["n_requests"] > 0, f"{mode}: no requests recorded"
        assert summary[mode]["worker_count_min"] < summary[mode]["worker_count_max"], (
            f"{mode}: worker count never changed — no kill/restore was observed, so "
            f"the run is not a valid chaos test (follow the on-screen prompts)"
        )
