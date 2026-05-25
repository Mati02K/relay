from __future__ import annotations

from pathlib import Path

import pytest

from relay.config import ModelConfig, RelayConfig, load_config
from relay.init import InitOptions, _resolve_catalog_model_selection, run_init


def test_numbered_catalog_model_selection() -> None:
    assert _resolve_catalog_model_selection("1") == "qwen2.5-0.5b"
    assert _resolve_catalog_model_selection("2") == "qwen2.5-1.5b"
    assert _resolve_catalog_model_selection("mistral-7b") == "mistral-7b"


def test_interactive_pull_accepts_numbered_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELAY_HOME", str(tmp_path))
    inputs = iter(["pull", "1"])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    def fake_pull(config: RelayConfig, model_id: str) -> tuple[RelayConfig, ModelConfig]:
        assert model_id == "qwen2.5-0.5b"
        model = ModelConfig(id=model_id, path="/tmp/qwen2.5-0.5b.gguf", source="test")
        return config.with_model(model), model

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("relay.init.pull_catalog_model", fake_pull)
    monkeypatch.setattr("relay.init.ensure_runtime_software", lambda config: None)

    run_init(
        InitOptions(
            role="all",
            network="lan",
            coordinator=None,
            node_id="numbered-model-node",
            host="127.0.0.1",
            model=None,
            model_path=None,
            skip_model=False,
            force=True,
        )
    )

    config = load_config(tmp_path / "config.json")
    assert config.models[0].id == "qwen2.5-0.5b"
