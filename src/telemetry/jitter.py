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

How the EMAs are computed
-------------------------
Two exponential moving averages run side-by-side per worker, with smoothing
factor ``alpha = 0.15`` (see :data:`DEFAULT_EMA_ALPHA`). On each probe::

    deviation       = abs(rtt - rtt_ema_prev)
    rtt_ema_new     = alpha * rtt       + (1 - alpha) * rtt_ema_prev
    jitter_ema_new  = alpha * deviation + (1 - alpha) * jitter_ema_prev

``rtt_ema_ms`` smooths the RTT itself — the "center" each sample is measured
against. ``jitter_ema_ms`` smooths the absolute deviation from that center,
which is the standard definition of jitter (variability, not magnitude). Only
``jitter_ema_ms`` is exposed to the scheduler; ``rtt_ema_ms`` exists solely
as the reference point ``deviation`` is measured against.

State persists until the worker leaves membership. Past flakiness decays
exponentially but never fully resets — this is deliberate, since correlated
flakiness (thermal, congestion, WiFi) makes recent history mildly predictive.
A genuine "fresh start" only happens via :meth:`_drop_missing` when the
worker disappears from etcd and re-registers.

Probe failures
--------------
A failed ``/health`` probe (connect error, timeout, non-200) does **not**
mean the worker is dead — its etcd lease is renewed via a separate gRPC
path, and its telemetry keeps flowing. It only means the coordinator can't
reach this worker right now. To express that as a routing penalty without
evicting the worker from membership, failures feed a synthetic deviation
into ``jitter_ema_ms`` via :meth:`observe_failure`.

The penalty **ramps** with consecutive failures so a single transient
hiccup doesn't dominate routing::

    penalty_ms = min(consecutive_failures * FAILURE_PENALTY_BASE_MS,
                     FAILURE_PENALTY_MAX_MS)
    jitter_ema_new = alpha * penalty_ms + (1 - alpha) * jitter_ema_prev

``consecutive_failures`` resets to 0 on the next successful probe, so
alternating success/failure stays gentle while sustained failures escalate.
``rtt_ema_ms`` is intentionally **not** updated on failure — failure isn't
a real RTT measurement, and polluting the center would slow recovery once
the worker comes back. Tunable via ``RELAY_JITTER_FAILURE_BASE_MS`` and
``RELAY_JITTER_FAILURE_MAX_MS``.

Recovery is gradual: the counter resets immediately on first success, but
``jitter_ema_ms`` decays exponentially over ~15-30 subsequent probes
(~30-60 seconds at the default 2s interval). This matches the routing
intent — stop *adding* penalty the moment the worker recovers, but stay
suspicious until new data confirms the recovery is stable.
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
FAILURE_PENALTY_BASE_MS = float(os.getenv("RELAY_JITTER_FAILURE_BASE_MS", "100"))
FAILURE_PENALTY_MAX_MS = float(os.getenv("RELAY_JITTER_FAILURE_MAX_MS", "1000"))


@dataclass
class _WorkerJitterState:
    """Per-worker rolling RTT and jitter EMAs in milliseconds."""

    rtt_ema_ms: float = 0.0
    jitter_ema_ms: float = 0.0
    # True once any update has happened (success OR failure). Gates whether
    # get_jitter_ms returns a value or None — the scheduler must see failure
    # penalties even before any successful probe, so failures flip this too.
    observed: bool = False
    # True once a real RTT measurement has anchored rtt_ema_ms. Gates the
    # "seed vs blend" branch in observe(); failures do NOT flip this because
    # they never produce a real RTT.
    has_rtt: bool = False
    consecutive_failures: int = field(default=0)


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
        if state is None or not state.observed:
            return None
        return state.jitter_ema_ms

    def observe(self, node_id: str, rtt_ms: float) -> None:
        """Record a single RTT sample and update the per-worker EMAs.

        Computation per sample, with ``a = self._alpha`` (default 0.15)::

            deviation       = abs(rtt - rtt_ema_prev)
            rtt_ema_new     = a * rtt       + (1 - a) * rtt_ema_prev
            jitter_ema_new  = a * deviation + (1 - a) * jitter_ema_prev

        The first sample seeds ``rtt_ema_ms = rtt`` and ``jitter_ema_ms = 0``
        so a single observation does not pretend to be a deviation from
        itself. ``deviation`` is computed against ``rtt_ema_prev`` *before*
        the RTT EMA is updated, so each sample is measured against the
        center that existed at the moment it arrived.

        Only ``jitter_ema_ms`` escapes via :meth:`get_jitter_ms` and becomes
        the scheduler's ``j_w``. Exposed publicly for tests and for paths
        that already have an RTT in hand.
        """
        state = self._state.setdefault(node_id, _WorkerJitterState())
        state.consecutive_failures = 0
        if not state.has_rtt:
            # First real RTT — seed the center. Any prior jitter_ema from
            # failure-only samples is preserved so the worker still wears its
            # earned penalty; the EMA will decay through subsequent successes.
            state.rtt_ema_ms = rtt_ms
        else:
            # Measure deviation against the pre-update center, then advance both EMAs.
            deviation = abs(rtt_ms - state.rtt_ema_ms)
            state.rtt_ema_ms = self._alpha * rtt_ms + (1.0 - self._alpha) * state.rtt_ema_ms
            state.jitter_ema_ms = (
                self._alpha * deviation + (1.0 - self._alpha) * state.jitter_ema_ms
            )
        state.has_rtt = True
        state.observed = True

    def observe_failure(self, node_id: str) -> None:
        """Record a probe failure with a ramped penalty into ``jitter_ema_ms``.

        Penalty per failure, with ``base = FAILURE_PENALTY_BASE_MS`` and
        ``cap = FAILURE_PENALTY_MAX_MS``::

            consecutive_failures += 1
            penalty_ms      = min(consecutive_failures * base, cap)
            jitter_ema_new  = alpha * penalty_ms + (1 - alpha) * jitter_ema_prev

        ``rtt_ema_ms`` is left untouched because the probe never produced an
        RTT measurement, and polluting the center would slow recovery once
        the worker comes back. The counter resets to 0 on the next successful
        probe (in :meth:`observe`), so alternating success/failure stays
        gentle while sustained failures escalate.
        """
        state = self._state.setdefault(node_id, _WorkerJitterState())
        state.consecutive_failures += 1
        penalty_ms = min(
            state.consecutive_failures * FAILURE_PENALTY_BASE_MS,
            FAILURE_PENALTY_MAX_MS,
        )
        if not state.observed:
            state.jitter_ema_ms = penalty_ms
        else:
            state.jitter_ema_ms = (
                self._alpha * penalty_ms + (1.0 - self._alpha) * state.jitter_ema_ms
            )
        state.observed = True

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
            self.observe_failure(node_id)
            logger.debug(
                "Jitter probe failed | nodeId={} error={} consecutiveFailures={}",
                node_id,
                e,
                self._state[node_id].consecutive_failures,
            )
            return
        if response.status_code != 200:
            self.observe_failure(node_id)
            logger.debug(
                "Jitter probe skipped | nodeId={} address={} status={} consecutiveFailures={}",
                node_id,
                address,
                response.status_code,
                self._state[node_id].consecutive_failures,
            )
            return
        rtt_ms = (time.perf_counter() - start) * 1000.0
        self.observe(node_id, rtt_ms)
        state = self._state.get(node_id)
        if state is not None:
            logger.debug(
                "Jitter probe sample | nodeId={} address={} rttMs={:.3f} "
                "rttEmaMs={:.3f} jitterMs={:.3f}",
                node_id,
                address,
                rtt_ms,
                state.rtt_ema_ms,
                state.jitter_ema_ms,
            )
