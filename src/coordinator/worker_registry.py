"""Worker metadata and telemetry discovery for coordinator scheduling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from membership.base import MembershipLayer
from telemetry.schemas import PrefixCacheTelemetry, RequestComputedTelemetry, Telemetry

WORKER_METADATA_KEY_PREFIX = "/relay/workers"


@dataclass(frozen=True)
class WorkerSnapshot:
    """Coordinator-side view of one worker."""

    node_id: str
    address: str
    metadata: dict[str, Any]
    telemetry: Telemetry

    def supports_model(self, model_id: str | None) -> bool:
        """Return whether this worker advertises support for ``model_id``."""
        if not model_id:
            return True
        models = self.metadata.get("models")
        if not isinstance(models, list):
            return False
        for model in models:
            if not isinstance(model, dict):
                continue
            if not model.get("loaded", True):
                continue
            advertised_id = model.get("id")
            if isinstance(advertised_id, str) and advertised_id == model_id:
                return True
        return False


async def fetch_worker_snapshots(membership: MembershipLayer) -> list[WorkerSnapshot]:
    """Read all worker metadata and telemetry snapshots from membership storage."""
    raw = await membership.getByPrefix(f"{WORKER_METADATA_KEY_PREFIX}/")
    metadata_by_node: dict[str, dict[str, Any]] = {}
    telemetry_by_node: dict[str, Telemetry] = {}

    for key, value in raw.items():
        parsed = _parse_worker_key(key)
        if parsed is None:
            continue
        node_id, field = parsed
        if field == "metadata":
            metadata = _load_json_object(value)
            if metadata is not None:
                metadata_by_node[node_id] = metadata
        elif field == "telemetry":
            try:
                telemetry_by_node[node_id] = Telemetry.model_validate_json(value)
            except ValueError:
                logger.warning("Ignoring invalid worker telemetry | key={}", key)

    snapshots: list[WorkerSnapshot] = []
    for node_id, metadata in metadata_by_node.items():
        address = metadata.get("address")
        if not isinstance(address, str) or not address:
            logger.warning("Ignoring worker without address | nodeId={}", node_id)
            continue
        snapshots.append(
            WorkerSnapshot(
                node_id=node_id,
                address=address.rstrip("/"),
                metadata=metadata,
                telemetry=telemetry_by_node.get(node_id, _default_telemetry(metadata)),
            )
        )
    return snapshots


def _parse_worker_key(key: str) -> tuple[str, str] | None:
    prefix = f"{WORKER_METADATA_KEY_PREFIX}/"
    if not key.startswith(prefix):
        return None
    rest = key.removeprefix(prefix)
    parts = rest.split("/")
    if len(parts) != 2:
        return None
    node_id, field = parts
    if not node_id or field not in {"metadata", "telemetry"}:
        return None
    return node_id, field


def _load_json_object(value: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _default_telemetry(metadata: dict[str, Any]) -> Telemetry:
    prefix_cache = metadata.get("prefix_cache")
    if isinstance(prefix_cache, dict):
        try:
            return _telemetry_with_prefix(PrefixCacheTelemetry.model_validate(prefix_cache))
        except ValueError:
            logger.warning("Ignoring invalid worker prefix metadata")
    return Telemetry.from_parts()


def _telemetry_with_prefix(prefix_cache: PrefixCacheTelemetry) -> Telemetry:
    return Telemetry.from_parts(
        request=RequestComputedTelemetry(
            sw_by_bucket={},
            prefix_cache=prefix_cache,
            sprefill_tokens_per_sec=0.0,
        )
    )
