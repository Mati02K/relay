"""Worker warm-up: prime every worker before the suite measures anything.

Workers load their model lazily on the first request, so the first measured
phase of a cold cluster is skewed by model-load latency and empty telemetry.
Before any test runs, we send a small fixed set of synthetic prompts directly
to each worker (bypassing the scheduler) so every worker has its engine loaded
and its queue/speed telemetry settled by the time measurement starts.

The prompts are deliberately unique sentinel strings that do not appear in the
ShareGPT working set, so warming never pollutes the prefix-cache or routing
state the real scenarios depend on. Warming hits every worker symmetrically,
so it introduces no routing bias.
"""

from __future__ import annotations

import asyncio
from typing import Any

from framework.client import RelayClient
from framework.cluster import ClusterClient

# Synthetic, dataset-free prompts. The "[relay-warmup]" sentinel guarantees they
# never collide with a ShareGPT prompt or share a cache prefix with test prompts.
WARMUP_PROMPTS: list[str] = [
    "[relay-warmup] Reply with the single word: ready.",
    "[relay-warmup] Count from one to three in words.",
    "[relay-warmup] Name one primary color and stop.",
    "[relay-warmup] What is 6 plus 9? Answer with only the number.",
    "[relay-warmup] Echo this token exactly: stabilize.",
]

WARMUP_RUNS = 5
WARMUP_MAX_TOKENS = 16
_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0


async def warm_up_workers(
    cluster: ClusterClient,
    *,
    model: str | None,
    runs: int = WARMUP_RUNS,
    settle_seconds: float = 3.0,
) -> None:
    """Send warm-up prompts directly to every healthy worker, then settle.

    Each worker receives ``runs`` prompts (cycling :data:`WARMUP_PROMPTS`) so its
    engine is loaded and its queue/speed telemetry has real samples. Requests go
    to the worker address directly so warming is guaranteed to cover every
    worker regardless of how the scheduler would route. Failures are retried and
    then tolerated — warming is best-effort and must never block the suite.
    """
    try:
        workers = await cluster.wait_for_workers(min_count=1, timeout_seconds=90.0)
    except TimeoutError:
        print("[warmup] no healthy workers appeared in time; skipping warm-up")
        return

    print(f"[warmup] priming {len(workers)} worker(s) with {runs} run(s) each")
    await asyncio.gather(*(_warm_one_worker(w, model, runs) for w in workers))

    # Let queue depth drain back to zero and prefill-speed telemetry propagate
    # before the first phase is measured.
    await cluster.wait_telemetry_propagation(settle_seconds)
    print("[warmup] all workers primed and settled")


async def _warm_one_worker(worker: dict[str, Any], model: str | None, runs: int) -> None:
    """Warm a single worker by sending it ``runs`` prompts in sequence."""
    node_id = str(worker.get("node_id", "?"))
    address = worker.get("address")
    if not isinstance(address, str) or not address:
        print(f"[warmup] {node_id} has no address; skipping")
        return

    client = RelayClient(address)
    try:
        for i in range(runs):
            prompt = WARMUP_PROMPTS[i % len(WARMUP_PROMPTS)]
            await _warm_request(client, prompt, model, node_id, i)
    finally:
        await client.aclose()


async def _warm_request(
    client: RelayClient,
    prompt: str,
    model: str | None,
    node_id: str,
    index: int,
) -> None:
    """Send one warm-up request, retrying while the engine is still loading."""
    messages = [{"role": "user", "content": prompt}]
    record = None
    for attempt in range(1, _RETRIES + 1):
        record = await client.chat_completion(
            messages,
            model=model,
            max_tokens=WARMUP_MAX_TOKENS,
            scenario="warmup",
            phase="warmup",
        )
        if record.error is None:
            return
        if attempt < _RETRIES:
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
    last_error = record.error if record is not None else "no response"
    print(f"[warmup] {node_id} run {index} did not complete (last error: {last_error})")
