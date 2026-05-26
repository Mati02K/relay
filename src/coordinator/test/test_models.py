from __future__ import annotations

from coordinator.main import _collect_model_entries
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import PrefixHashConfig
from telemetry.schemas import PrefixCacheTelemetry, Telemetry


def test_empty_worker_list_returns_no_entries() -> None:
    assert _collect_model_entries([]) == []


def test_collects_every_loaded_model_in_openai_shape() -> None:
    workers = [
        _worker_with_models("a", [{"id": "qwen2.5-0.5b", "loaded": True}]),
        _worker_with_models("b", [{"id": "mistral-7b", "loaded": True}]),
    ]

    entries = _collect_model_entries(workers)

    assert entries == [
        {"id": "qwen2.5-0.5b", "object": "model", "owned_by": "relay"},
        {"id": "mistral-7b", "object": "model", "owned_by": "relay"},
    ]


def test_duplicate_model_across_workers_appears_once() -> None:
    workers = [
        _worker_with_models("a", [{"id": "qwen2.5-3b", "loaded": True}]),
        _worker_with_models("b", [{"id": "qwen2.5-3b", "loaded": True}]),
    ]

    entries = _collect_model_entries(workers)

    assert [entry["id"] for entry in entries] == ["qwen2.5-3b"]


def test_unloaded_models_are_excluded() -> None:
    workers = [
        _worker_with_models(
            "a",
            [
                {"id": "qwen2.5-3b", "loaded": True},
                {"id": "phi-3.5-mini", "loaded": False},
            ],
        ),
    ]

    entries = _collect_model_entries(workers)

    assert [entry["id"] for entry in entries] == ["qwen2.5-3b"]


def test_missing_or_invalid_models_field_is_ignored() -> None:
    workers = [
        _worker_with_metadata({"address": "http://a", "models": "not-a-list"}),
        _worker_with_metadata({"address": "http://b"}),  # no models key at all
        _worker_with_models("c", [{"id": "qwen2.5-3b", "loaded": True}]),
    ]

    entries = _collect_model_entries(workers)

    assert [entry["id"] for entry in entries] == ["qwen2.5-3b"]


def _worker_with_models(node_id: str, models: list[dict[str, object]]) -> WorkerSnapshot:
    return _worker_with_metadata({"address": f"http://{node_id}", "models": models}, node_id)


def _worker_with_metadata(metadata: dict[str, object], node_id: str = "node") -> WorkerSnapshot:
    telemetry = Telemetry(
        qw=0,
        sw_by_bucket={},
        mw=0.0,
        jw=0.0,
        theta_w=0,
        prefix_cache=PrefixCacheTelemetry.from_config(PrefixHashConfig(model_id="x"), []),
        sprefill_tokens_per_sec=0.0,
    )
    address = metadata.get("address", f"http://{node_id}")
    assert isinstance(address, str)
    return WorkerSnapshot(
        node_id=node_id,
        address=address,
        metadata=metadata,
        telemetry=telemetry,
    )
