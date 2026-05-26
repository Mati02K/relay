"""Live worker telemetry registry maintained by the coordinator.

Polls every alive worker's ``/v1/telemetry`` endpoint on a fixed cadence,
measures coordinator-side RTT jitter (paper §3.2 ``j_w``), and exposes a
snapshot of the cluster state for the scheduler.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import httpx
from loguru import logger

from membership.base import MembershipLayer
from worker.inference.base import Telemetry, ema_update

_DEFAULT_POLL_INTERVAL_MS: int = int(os.getenv("RELAY_POLL_INTERVAL_MS", "200"))
_DEFAULT_POLL_TIMEOUT_MS: int = int(os.getenv("RELAY_POLL_TIMEOUT_MS", "500"))
_DEFAULT_STALE_THRESHOLD_MS: int = int(os.getenv("RELAY_STALE_THRESHOLD_MS", "2000"))


@dataclass
class WorkerState:
    """One worker's most recent observed state, as seen by the coordinator."""

    node_id: str
    url: str
    telemetry: Telemetry
    rtt_ms_ema: float = 0.0
    jitter_ms_ema: float = 0.0
    last_seen_ms: float = 0.0
    online: bool = False
    consecutive_failures: int = 0
    metadata: dict = field(default_factory=dict)


class WorkerRegistry:
    """Background poller that keeps fresh telemetry for every worker in the cluster."""

    def __init__(
        self,
        membership: MembershipLayer,
        *,
        poll_interval_ms: int = _DEFAULT_POLL_INTERVAL_MS,
        poll_timeout_ms: int = _DEFAULT_POLL_TIMEOUT_MS,
        stale_threshold_ms: int = _DEFAULT_STALE_THRESHOLD_MS,
    ) -> None:
        self._membership = membership
        self._poll_interval_s: float = poll_interval_ms / 1000.0
        self._stale_threshold_ms = stale_threshold_ms
        self._states: dict[str, WorkerState] = {}
        self._task: asyncio.Task | None = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(poll_timeout_ms / 1000.0))

    def start(self) -> None:
        """Begin polling workers in the background. Idempotent."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info(
                "WorkerRegistry started | interval={}ms timeout={}ms",
                int(self._poll_interval_s * 1000),
                int(self._http.timeout.read * 1000) if self._http.timeout.read else -1,
            )

    async def stop(self) -> None:
        """Stop the background task and close the HTTP client."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._http.aclose()
        logger.info("WorkerRegistry stopped")

    def snapshot(self) -> dict[str, WorkerState]:
        """Return a shallow copy of the current per-worker state map."""
        return dict(self._states)

    def online_workers(self) -> dict[str, WorkerState]:
        """Return only workers that are currently online."""
        return {nid: s for nid, s in self._states.items() if s.online}

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception:
                logger.exception("Worker registry tick failed")
            await asyncio.sleep(self._poll_interval_s)

    async def _tick(self) -> None:
        members = await self._membership.getAliveMembers()
        workers = [m for m in members if m.metadata.get("role") == "worker"]

        # Poll all workers concurrently so one slow node doesn't block the others.
        await asyncio.gather(
            *[self._poll_worker(m.id, m.metadata) for m in workers],
            return_exceptions=True,
        )

        # Mark workers we haven't heard from recently as offline.
        now_ms = time.monotonic() * 1000.0
        for state in self._states.values():
            if state.online and (now_ms - state.last_seen_ms) > self._stale_threshold_ms:
                state.online = False
                logger.warning(
                    "Worker marked offline | nodeId={} lastSeenMs={:.0f} ago",
                    state.node_id,
                    now_ms - state.last_seen_ms,
                )

        # Drop entries for workers that have been deregistered entirely.
        known_ids = {m.id for m in workers}
        for stale_id in list(self._states.keys()):
            if stale_id not in known_ids:
                del self._states[stale_id]
                logger.info("Worker removed from registry | nodeId={}", stale_id)

    async def _poll_worker(self, node_id: str, metadata: dict) -> None:
        url = metadata.get("url")
        if not url:
            return

        t0 = time.monotonic()
        try:
            resp = await self._http.get(f"{url}/v1/telemetry")
        except httpx.RequestError as exc:
            self._record_failure(node_id, url, metadata, reason=str(exc))
            return

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code != 200:
            self._record_failure(
                node_id, url, metadata, reason=f"HTTP {resp.status_code}"
            )
            return

        try:
            telemetry = Telemetry(**resp.json())
        except Exception as exc:
            self._record_failure(node_id, url, metadata, reason=f"parse: {exc}")
            return

        prev = self._states.get(node_id)
        if prev is None:
            rtt_ema = elapsed_ms
            jitter_ema = 0.0
        else:
            rtt_ema = ema_update(prev.rtt_ms_ema, elapsed_ms)
            instantaneous_jitter = abs(elapsed_ms - prev.rtt_ms_ema)
            jitter_ema = ema_update(prev.jitter_ms_ema, instantaneous_jitter)

        # Override engine-reported jw with the coordinator-measured jitter EMA;
        # the engine has no visibility into network conditions, only the coordinator does.
        telemetry_with_jitter = telemetry.model_copy(update={"jw": jitter_ema})

        self._states[node_id] = WorkerState(
            node_id=node_id,
            url=url,
            telemetry=telemetry_with_jitter,
            rtt_ms_ema=rtt_ema,
            jitter_ms_ema=jitter_ema,
            last_seen_ms=time.monotonic() * 1000.0,
            online=True,
            consecutive_failures=0,
            metadata=metadata,
        )

    def _record_failure(self, node_id: str, url: str, metadata: dict, *, reason: str) -> None:
        prev = self._states.get(node_id)
        failures = (prev.consecutive_failures + 1) if prev else 1
        logger.debug(
            "Telemetry poll failed | nodeId={} reason={} consecutiveFailures={}",
            node_id,
            reason,
            failures,
        )
        if prev is not None:
            prev.consecutive_failures = failures
            if failures >= 3:
                prev.online = False
        else:
            self._states[node_id] = WorkerState(
                node_id=node_id,
                url=url,
                telemetry=Telemetry(),
                last_seen_ms=0.0,
                online=False,
                consecutive_failures=failures,
                metadata=metadata,
            )
