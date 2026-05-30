"""Cluster control helpers — weight management, fake telemetry injection, worker discovery."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class ClusterClient:
    """HTTP client targeting the Relay coordinator for cluster-level control.

    Wraps the coordinator's scheduler weight, mode, and worker-weight APIs,
    plus the per-worker test override endpoint added for this test suite.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "queue": 1.0,
        "prefix_miss": 1.0,
        "memory": 1.0,
        "jitter": 1.0,
        "thermal": 1.0,
        "nu": 0.0,
    }

    def __init__(self, coordinator_url: str, timeout: float = 10.0) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.timeout = timeout

    # ── Worker discovery ──────────────────────────────────────────────────────

    async def get_workers(self) -> list[dict[str, Any]]:
        """Return the live worker list from ``GET /v1/workers``."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.coordinator_url}/v1/workers")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def get_healthy_workers(self) -> list[dict[str, Any]]:
        """Return only workers where ``healthy == True``."""
        return [w for w in await self.get_workers() if w.get("healthy")]

    async def wait_for_workers(
        self, min_count: int = 2, timeout_seconds: float = 60.0
    ) -> list[dict[str, Any]]:
        """Poll until at least ``min_count`` healthy workers are registered.

        Raises :class:`TimeoutError` if the deadline passes.
        """
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while True:
            workers = await self.get_healthy_workers()
            if len(workers) >= min_count:
                return workers
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Only {len(workers)} healthy workers after {timeout_seconds}s "
                    f"(need {min_count})"
                )
            await asyncio.sleep(2.0)

    # ── Scheduler control ─────────────────────────────────────────────────────

    async def set_weights(self, **weights: float) -> dict[str, float]:
        """Partially update scheduler weights via ``POST /v1/scheduler/weights``."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.coordinator_url}/v1/scheduler/weights", json=weights
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def reset_weights(self) -> dict[str, float]:
        """Restore all weights to their defaults (all 1.0 except nu=0)."""
        return await self.set_weights(**self.DEFAULT_WEIGHTS)

    async def set_mode(self, mode: str) -> dict[str, str]:
        """Switch scheduler mode: ``"cost"`` or ``"round_robin"``."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.coordinator_url}/v1/scheduler/mode", json={"mode": mode}
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def get_weights(self) -> dict[str, float]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.coordinator_url}/v1/scheduler/weights")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def reset_scheduler(self) -> None:
        """Reset weights to defaults and switch to cost mode."""
        await self.reset_weights()
        await self.set_mode("cost")

    async def set_worker_weight_override(self, node_id: str, weight: float | None) -> None:
        """Set or clear a per-worker preference override."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.coordinator_url}/v1/scheduler/worker_weights",
                json={node_id: weight},
            )
            response.raise_for_status()

    async def clear_all_worker_weight_overrides(self) -> None:
        """Clear every per-worker override currently active on the coordinator."""
        workers = await self.get_workers()
        if not workers:
            return
        payload = {w["node_id"]: None for w in workers}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.coordinator_url}/v1/scheduler/worker_weights", json=payload
            )
            response.raise_for_status()

    async def wait_telemetry_propagation(self, seconds: float = 1.5) -> None:
        """Wait for worker telemetry to propagate to etcd and be visible to the coordinator."""
        await asyncio.sleep(seconds)

    # ── Full reset ────────────────────────────────────────────────────────────

    async def full_reset(self) -> None:
        """Reset scheduler weights, mode, and per-worker overrides."""
        await asyncio.gather(
            self.reset_scheduler(),
            self.clear_all_worker_weight_overrides(),
        )
