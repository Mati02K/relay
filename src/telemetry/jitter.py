"""Worker jitter probe — provides the ``j_w`` term of the scheduler cost.

Unlike the other modules in :mod:`telemetry` (engine telemetry, thermal,
prefix cache, request metrics) which are *worker-published* — workers measure
them locally and publish to etcd — jitter is **coordinator-measured**. The
coordinator periodically pings each known worker's ``/health`` endpoint,
records the round-trip time, and maintains two exponential moving averages
per worker: a smoothed mean RTT and a smoothed deviation. The deviation EMA
becomes ``j_w`` and is stamped onto :class:`WorkerSnapshot.telemetry` just
before :func:`choose_worker` runs.

It lives in :mod:`telemetry` (not :mod:`coordinator`) because the scheduler
formula treats it as one input among many — sibling to ``q_w``, ``m_w``,
``theta_w`` — and grouping them together makes the formula's data sources
self-evident. The "who measures it" detail is an implementation choice that
only matters when extending this collector.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import httpx
from loguru import logger

from membership.base import MembershipLayer

WORKER_METADATA_KEY_PREFIX = "/relay/workers"
DEFAULT_PROBE_INTERVAL_SECONDS = float(os.getenv("RELAY_JITTER_PROBE_INTERVAL_SECONDS", "2.0"))
DEFAULT_PROBE_TIMEOUT_SECONDS = float(os.getenv("RELAY_JITTER_PROBE_TIMEOUT_SECONDS", "1.0"))
DEFAULT_EMA_ALPHA = float(os.getenv("RELAY_JITTER_EMA_ALPHA", "0.15"))


@dataclass
class _WorkerJitterState:
    """Per-worker rolling RTT and jitter EMAs in milliseconds."""

    rtt_ema_ms: float = 0.0
    jitter_ema_ms: float = 0.0
    samples: int = field(default=0)


class JitterProbe:
    """Background task that measures network jitter to each known worker."""

    def __init__(
        self,
        membership: MembershipLayer,
        *,
        interval_seconds: float = DEFAULT_PROBE_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
        alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        self._membership = membership
        self._interval_seconds = max(0.1, interval_seconds)
        self._timeout_seconds = max(0.05, timeout_seconds)
        self._alpha = min(1.0, max(0.0, alpha))
        self._state: dict[str, _WorkerJitterState] = {}
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background probing loop."""
        if self._task is not None:
            return
        self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        self._task = asyncio.create_task(self._probe_loop())
        logger.info(
            "Jitter probe started | intervalSeconds={} alpha={}",
            self._interval_seconds,
            self._alpha,
        )

    async def stop(self) -> None:
        """Cancel the probing loop and close the HTTP client."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("Jitter probe stopped")

    def get_jitter_ms(self, node_id: str) -> float | None:
        """Return the latest jitter EMA in ms for ``node_id``, or None if untested."""
        state = self._state.get(node_id)
        if state is None or state.samples == 0:
            return None
        return state.jitter_ema_ms

    def observe(self, node_id: str, rtt_ms: float) -> None:
        """Record a single RTT sample and update the EMAs.

        Exposed for tests and for paths that already have an RTT in hand.
        """
        state = self._state.setdefault(node_id, _WorkerJitterState())
        if state.samples == 0:
            state.rtt_ema_ms = rtt_ms
            state.jitter_ema_ms = 0.0
        else:
            deviation = abs(rtt_ms - state.rtt_ema_ms)
            state.rtt_ema_ms = self._alpha * rtt_ms + (1.0 - self._alpha) * state.rtt_ema_ms
            state.jitter_ema_ms = (
                self._alpha * deviation + (1.0 - self._alpha) * state.jitter_ema_ms
            )
        state.samples += 1

    async def _probe_loop(self) -> None:
        while True:
            try:
                targets = await self._discover_targets()
                self._drop_missing(targets)
                if targets:
                    await asyncio.gather(
                        *(self._probe_one(node_id, addr) for node_id, addr in targets.items()),
                        return_exceptions=True,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Jitter probe loop error")
            await asyncio.sleep(self._interval_seconds)

    async def _discover_targets(self) -> dict[str, str]:
        raw = await self._membership.getByPrefix(f"{WORKER_METADATA_KEY_PREFIX}/")
        targets: dict[str, str] = {}
        prefix = f"{WORKER_METADATA_KEY_PREFIX}/"
        for key, value in raw.items():
            if not key.startswith(prefix) or not key.endswith("/metadata"):
                continue
            node_id = key.removeprefix(prefix).removesuffix("/metadata")
            if not node_id or "/" in node_id:
                continue
            try:
                metadata = json.loads(value)
            except json.JSONDecodeError:
                continue
            address = metadata.get("address") if isinstance(metadata, dict) else None
            if isinstance(address, str) and address:
                targets[node_id] = address.rstrip("/")
        return targets

    def _drop_missing(self, targets: dict[str, str]) -> None:
        for node_id in list(self._state.keys()):
            if node_id not in targets:
                del self._state[node_id]

    async def _probe_one(self, node_id: str, address: str) -> None:
        assert self._client is not None
        start = time.perf_counter()
        try:
            response = await self._client.get(f"{address}/health")
        except httpx.RequestError as e:
            logger.debug("Jitter probe failed | nodeId={} error={}", node_id, e)
            return
        if response.status_code != 200:
            return
        rtt_ms = (time.perf_counter() - start) * 1000.0
        self.observe(node_id, rtt_ms)
