"""Worker daemon that owns inference, telemetry, membership, and request execution."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from membership.base import MembershipLayer
from membership.etcd import EtcdMembership
from network.tailscale import TailscaleNetwork
from telemetry.request_metrics import extract_usage_token_counts
from telemetry.schemas import SystemTelemetry, ThermalSourceTelemetry, ThermalTelemetry
from telemetry.state import WorkerTelemetryState
from telemetry.thermal import ThermalAggregator, detect_thermal_collectors
from worker.inference.base import InferenceEngine
from worker.inference.llamacpp import LlamaCppEngine

WORKER_METADATA_KEY_PREFIX = "/relay/workers"
TELEMETRY_INTERVAL_SECONDS = 0.2
THERMAL_INTERVAL_SECONDS = float(os.getenv("RELAY_THERMAL_INTERVAL_SECONDS", "5.0"))
WORKER_LEASE_TTL_SECONDS = int(os.getenv("RELAY_WORKER_LEASE_TTL_SECONDS", "10"))


class WorkerDaemon:
    """Long-lived worker process state.

    A worker owns one inference engine and one mutable telemetry state. The
    coordinator sends scheduled requests here; this daemon runs inference and
    continuously publishes telemetry for the scheduler.
    """

    def __init__(
        self,
        *,
        node_id: str,
        address: str,
        membership: MembershipLayer,
        inference_engine: InferenceEngine,
        telemetry: WorkerTelemetryState | None = None,
        thermal: ThermalAggregator | None = None,
        telemetry_interval_seconds: float = TELEMETRY_INTERVAL_SECONDS,
        thermal_interval_seconds: float = THERMAL_INTERVAL_SECONDS,
    ) -> None:
        self.node_id = node_id
        self.address = address
        self.membership = membership
        self.inference_engine = inference_engine
        self.telemetry = telemetry or WorkerTelemetryState()
        self.thermal = thermal or ThermalAggregator(
            detect_thermal_collectors(uses_gpu=inference_engine.uses_gpu),
            primary_device="gpu" if inference_engine.uses_gpu else "cpu",
        )
        self.telemetry_interval_seconds = telemetry_interval_seconds
        self.thermal_interval_seconds = max(0.5, thermal_interval_seconds)
        self._tasks: list[asyncio.Task[None]] = []

    @classmethod
    def from_env(cls) -> WorkerDaemon:
        """Build the worker daemon from environment configuration."""
        node_id = os.getenv("NODE_ID", "worker")
        port = int(os.getenv("WORKER_PORT", "9090"))
        host = os.getenv("WORKER_HOST")
        if not host:
            host = TailscaleNetwork().getMyAddress()
        address = f"http://{host}:{port}"

        membership = EtcdMembership(
            host=os.getenv("MEMBERSHIP_HOST", "localhost"),
            port=int(os.getenv("MEMBERSHIP_PORT", "50051")),
        )

        # Currently only support for llamacpp
        inference_engine = LlamaCppEngine()

        return cls(
            node_id=node_id,
            address=address,
            membership=membership,
            inference_engine=inference_engine,
        )

    async def start(self) -> None:
        """Register the worker under a lease and start background publishers.

        Metadata and telemetry are attached to an etcd lease so the keys vanish
        automatically if this process crashes or loses the network. See
        :meth:`MembershipLayer.holdLease` for the contract.
        """
        models = _worker_models()
        if models:
            await self.ensure_ready()

        telemetry = await self.telemetry.snapshot()
        metadata = {
            "role": "worker",
            "node_id": self.node_id,
            "address": self.address,
            "engine": "llama.cpp",
            "engines": _worker_engines(),
            "models": models,
            "prefix_cache": telemetry.prefix_cache.model_dump(),
        }
        await self.membership.holdLease(WORKER_LEASE_TTL_SECONDS)
        await self.membership.register(self.node_id, metadata)
        await self.membership.putWithLease(self._metadata_key(), json.dumps(metadata))
        self._tasks.append(asyncio.create_task(self._publish_telemetry_loop()))
        self._tasks.append(asyncio.create_task(self._collect_thermal_loop()))
        logger.info(
            "Worker daemon started | nodeId={} addr={} thermalCollectors={}",
            self.node_id,
            self.address,
            self.thermal.collector_names,
        )

    async def stop(self) -> None:
        """Stop background tasks and deregister the worker."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.thermal.close()
        await self.inference_engine.stop()
        await self.membership.deregister(self.node_id)
        await self.membership.close()
        logger.info("Worker daemon stopped | nodeId={}", self.node_id)

    async def health(self) -> dict[str, Any]:
        """Return worker and engine health."""
        engine_health = await self.inference_engine.health()
        return {
            "status": "ok" if engine_health.status else "unhealthy",
            "nodeId": self.node_id,
            "address": self.address,
            "engine": engine_health.model_dump(),
        }

    async def ensure_ready(self) -> None:
        """Start the inference engine and fail before this worker advertises readiness."""
        await self.inference_engine.start()

    async def telemetry_snapshot(self) -> dict[str, Any]:
        """Return the latest local telemetry snapshot."""
        await self._refresh_engine_telemetry()
        snapshot = await self.telemetry.snapshot()
        return {
            "nodeId": self.node_id,
            "address": self.address,
            "telemetry": snapshot.model_dump(),
        }

    async def generate(self, request: Mapping[str, object]) -> AsyncIterator[str]:
        """Run one scheduled chat completion request and update request telemetry."""
        body = dict(request)
        prompt_context = self.telemetry.context_from_request(body)

        t0 = time.perf_counter()
        first_token_t: float | None = None
        last_usage_prompt: int | None = None
        last_usage_completion: int | None = None

        async for line in self.inference_engine.generate(body):
            if first_token_t is None and _line_has_content_delta(line):
                first_token_t = time.perf_counter()
            pt, ct = _usage_from_sse_line(line)
            if pt is not None:
                last_usage_prompt = pt
            if ct is not None:
                last_usage_completion = ct
            yield line

        elapsed = time.perf_counter() - t0
        prompt_tokens = (
            last_usage_prompt if last_usage_prompt is not None else prompt_context.estimated_tokens
        )
        completion_tokens = last_usage_completion if last_usage_completion is not None else 0
        first_token_seconds = first_token_t - t0 if first_token_t is not None else None
        await self.telemetry.observe_request_completion(
            prompt_context=prompt_context,
            elapsed_seconds=elapsed,
            first_token_seconds=first_token_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def _publish_telemetry_loop(self) -> None:
        while True:
            try:
                await self._refresh_engine_telemetry()
                snapshot = await self.telemetry.snapshot()
                await self.membership.putWithLease(
                    self._telemetry_key(),
                    snapshot.model_dump_json(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Telemetry publication failed | nodeId={}", self.node_id)
            await asyncio.sleep(self.telemetry_interval_seconds)

    async def _collect_thermal_loop(self) -> None:
        while True:
            try:
                snapshot = await self.thermal.sample()
                await self.telemetry.update_system(
                    SystemTelemetry(
                        jw=0.0,
                        theta_w=snapshot.theta_w,
                        thermal=_thermal_snapshot_to_schema(snapshot),
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Thermal collection failed | nodeId={}", self.node_id)
            await asyncio.sleep(self.thermal_interval_seconds)

    async def _refresh_engine_telemetry(self) -> None:
        engine_telemetry = await self.inference_engine.get_engine_telemetry()
        await self.telemetry.update_engine(engine_telemetry)

    def _metadata_key(self) -> str:
        return f"{WORKER_METADATA_KEY_PREFIX}/{self.node_id}/metadata"

    def _telemetry_key(self) -> str:
        return f"{WORKER_METADATA_KEY_PREFIX}/{self.node_id}/telemetry"


def _usage_from_sse_line(line: str) -> tuple[int | None, int | None]:
    if not line.startswith("data: "):
        return None, None
    payload = line.removeprefix("data: ").strip()
    if not payload or payload == "[DONE]":
        return None, None
    return extract_usage_token_counts(payload)


def _line_has_content_delta(line: str) -> bool:
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        return False
    payload = line.removeprefix("data: ").strip()
    if not payload or payload == "[DONE]":
        return False
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return False
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    delta = first.get("delta") or {}
    return isinstance(delta, dict) and bool(delta.get("content") or delta.get("reasoning_content"))


def _worker_engines() -> list[str]:
    """Return inference engines advertised by this worker."""
    raw = os.getenv("RELAY_MODEL_INVENTORY_JSON")
    if raw:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = {}
        engines = obj.get("engines") if isinstance(obj, dict) else None
        if isinstance(engines, list):
            return [item for item in engines if isinstance(item, str)]
    return [os.getenv("RELAY_ENGINE", "llama.cpp")]


def _worker_models() -> list[dict[str, Any]]:
    """Return model inventory advertised by this worker."""
    raw = os.getenv("RELAY_MODEL_INVENTORY_JSON")
    if raw:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = {}
        models = obj.get("models") if isinstance(obj, dict) else None
        if isinstance(models, list):
            return [item for item in models if isinstance(item, dict)]

    model_path = os.getenv("LLAMA_MODEL_PATH", "")
    model_id = (
        os.getenv("RELAY_MODEL_ID")
        or os.getenv("LLAMA_MODEL_ID")
        or (Path(model_path).stem if model_path else "unknown")
    )
    if not model_path:
        return []
    return [
        {
            "id": model_id,
            "engine": "llama.cpp",
            "path": model_path,
            "loaded": True,
        }
    ]


def _thermal_snapshot_to_schema(snapshot: Any) -> ThermalTelemetry:
    """Convert thermal dataclasses into Pydantic telemetry schemas."""
    return ThermalTelemetry(
        pressure=snapshot.pressure,
        state=snapshot.state,
        cpu_pressure=snapshot.cpu_pressure,
        gpu_pressure=snapshot.gpu_pressure,
        confidence=snapshot.confidence,
        sources=[
            ThermalSourceTelemetry(
                source=source.source,
                device_type=source.device_type,
                pressure=source.pressure,
                state=source.state,
                confidence=source.confidence,
                temperature_c=source.temperature_c,
                limit_c=source.limit_c,
                throttle_active=source.throttle_active,
                details=source.details,
            )
            for source in snapshot.sources
        ],
    )
