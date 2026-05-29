"""Closed-loop load generator for the Relay coordinator.

For each row in the prompts file we fire one streaming POST to ``/v1/chat`` and
record TTFT, total latency, decode-only latency, completion tokens, the worker
id (from the ``X-Relay-Worker`` response header), and whether the request was
rejected by SLO admission control. Results are appended to a CSV that
``bench/analyze.py`` can post-process for the ablation report.

The script intentionally keeps a small, bounded number of concurrent requests
in flight (``--concurrency``) instead of pre-scheduling them all, so the
coordinator sees a realistic queue depth at any given moment. With
``--concurrency 1`` it becomes a strict sequential replay, which is the
cleanest setting for TTFT ablation comparisons.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields as _dc_fields
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class RequestResult:
    """Per-request observation written to the results CSV."""

    request_id: str
    prompt_id: str
    prompt_chars: int
    worker: str
    status: int
    ttft_ms: float
    total_ms: float
    decode_ms: float
    completion_tokens: int
    rejected: bool = False
    error: str = ""


def _load_prompts(path: Path) -> list[dict]:
    """Load prompts from JSONL. Each line must have ``prompt_id`` and ``content`` fields."""
    prompts: list[dict] = []
    with path.open() as f:
        for i, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            prompts.append({
                "prompt_id": str(obj.get("prompt_id") or f"p{i}"),
                "content": str(obj.get("content", "")),
            })
    return prompts


def _extract_completion_tokens(payload: str) -> Optional[int]:
    """Pull ``usage.completion_tokens`` from a final llama-server chunk if present."""
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    usage = obj.get("usage")
    if isinstance(usage, dict):
        ct = usage.get("completion_tokens")
        if isinstance(ct, int):
            return ct
    return None


async def _run_one(
    client: httpx.AsyncClient,
    coordinator_url: str,
    prompt: dict,
    max_tokens: int,
    timeout_s: float,
) -> RequestResult:
    """Send one streaming chat request and time it end-to-end."""
    body = {
        "messages": [{"role": "user", "content": prompt["content"]}],
        "max_tokens": max_tokens,
    }
    req_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    first_byte_t: float | None = None
    completion_tokens = 0
    worker = ""
    status = 0
    rejected = False
    error_str = ""
    try:
        async with client.stream(
            "POST",
            f"{coordinator_url}/v1/chat",
            json=body,
            headers={"Accept": "text/event-stream"},
            timeout=timeout_s,
        ) as resp:
            status = resp.status_code
            worker = resp.headers.get("X-Relay-Worker", "")
            if status == 429:
                rejected = True
                await resp.aread()
            elif status >= 400:
                error_str = (await resp.aread()).decode(errors="replace")[:200]
            else:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if first_byte_t is None:
                        first_byte_t = time.perf_counter()
                    if not line.startswith("data: "):
                        continue
                    payload = line.removeprefix("data: ").strip()
                    if not payload or payload == "[DONE]":
                        continue
                    ct = _extract_completion_tokens(payload)
                    if ct is not None:
                        completion_tokens = ct
    except (httpx.RequestError, asyncio.TimeoutError) as e:
        error_str = f"transport:{type(e).__name__}:{e}"

    total = (time.perf_counter() - t0) * 1000.0
    ttft = (first_byte_t - t0) * 1000.0 if first_byte_t else float("nan")
    decode = (total - ttft) if first_byte_t else float("nan")
    return RequestResult(
        request_id=req_id,
        prompt_id=prompt["prompt_id"],
        prompt_chars=len(prompt["content"]),
        worker=worker,
        status=status,
        ttft_ms=ttft,
        total_ms=total,
        decode_ms=decode,
        completion_tokens=completion_tokens,
        rejected=rejected,
        error=error_str,
    )


async def _run_workload(args: argparse.Namespace, prompts: list[dict]) -> list[RequestResult]:
    """Drive the workload with bounded concurrency."""
    sem = asyncio.Semaphore(args.concurrency)
    results: list[RequestResult] = []
    finished = 0
    total = len(prompts)

    async with httpx.AsyncClient() as client:
        async def worker_coro(p: dict) -> None:
            nonlocal finished
            async with sem:
                r = await _run_one(client, args.coordinator, p, args.max_tokens, args.timeout)
                results.append(r)
                finished += 1
                if finished % max(1, total // 10) == 0 or finished == total:
                    print(
                        f"  [{finished}/{total}] last: worker={r.worker or '-':9} "
                        f"ttft={r.ttft_ms:7.1f}ms total={r.total_ms:7.1f}ms "
                        f"tokens={r.completion_tokens} rejected={r.rejected}",
                        flush=True,
                    )

        if args.warmup > 0:
            warmup_prompt = {"prompt_id": "warmup", "content": "Say hi."}
            print(f"Warmup: {args.warmup} requests...", flush=True)
            for _ in range(args.warmup):
                await _run_one(client, args.coordinator, warmup_prompt, 8, args.timeout)

        print(f"Replaying {total} prompts at concurrency={args.concurrency}...", flush=True)
        await asyncio.gather(*[worker_coro(p) for p in prompts])

    return results


def _write_csv(results: list[RequestResult], out_path: Path, run_label: str) -> None:
    """Append results to a CSV with a `run_label` column for cross-config analysis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    fields = ["run_label"] + [f.name for f in _dc_fields(RequestResult)]
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in results:
            row = asdict(r)
            row["run_label"] = run_label
            w.writerow(row)


def _print_summary(results: list[RequestResult], run_label: str) -> None:
    """Console summary so each run gives an at-a-glance verdict without opening the CSV."""
    ok = [r for r in results if r.status == 200 and not r.rejected]
    if not ok:
        print(f"[{run_label}] no successful requests")
        return
    ttfts = sorted(r.ttft_ms for r in ok if r.ttft_ms == r.ttft_ms)  # filter NaN
    totals = sorted(r.total_ms for r in ok)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return float("nan")
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    by_worker: dict[str, int] = {}
    for r in ok:
        by_worker[r.worker] = by_worker.get(r.worker, 0) + 1

    rejected_n = sum(1 for r in results if r.rejected)
    failed_n = sum(1 for r in results if r.status >= 400 and not r.rejected)
    print(f"\n[{run_label}] n={len(results)} ok={len(ok)} rejected={rejected_n} failed={failed_n}")
    print(
        f"  ttft_ms:  p50={pct(ttfts,50):7.1f}  p90={pct(ttfts,90):7.1f}  "
        f"p99={pct(ttfts,99):7.1f}  mean={statistics.fmean(ttfts):7.1f}"
    )
    print(
        f"  total_ms: p50={pct(totals,50):7.1f}  p90={pct(totals,90):7.1f}  "
        f"p99={pct(totals,99):7.1f}  mean={statistics.fmean(totals):7.1f}"
    )
    print(f"  per-worker request counts: {by_worker}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop replay against the Relay coordinator")
    parser.add_argument("--coordinator", default=os.getenv("RELAY_COORD", "http://127.0.0.1:8080"))
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="results CSV (appended)")
    parser.add_argument("--run-label", required=True, help="label distinguishing this ablation config")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 = all prompts")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts)
    if args.limit > 0:
        prompts = prompts[: args.limit]
    if not prompts:
        print("no prompts loaded", file=sys.stderr)
        return 2

    results = asyncio.run(_run_workload(args, prompts))
    _write_csv(results, args.out, args.run_label)
    _print_summary(results, args.run_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
