"""Tests for the quality-aware router."""

from __future__ import annotations

import pytest

from coordinator.router import (
    DEFAULT_MODEL_QUALITY,
    detect_request_modalities,
    estimate_complexity_score,
    quality_routing_term,
    worker_model_quality,
    worker_modalities,
    worker_supports_modalities,
)
from coordinator.scheduler import WEIGHTS, SchedulingError, choose_worker
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import PrefixHashConfig
from telemetry.schemas import PrefixCacheTelemetry, Telemetry


def test_complexity_zero_for_empty_prompt() -> None:
    assert estimate_complexity_score("") == 0.0


def test_complexity_rises_with_reasoning_keywords() -> None:
    trivial = estimate_complexity_score("hi")
    hard = estimate_complexity_score(
        "Explain step by step why a memory-pressured worker should "
        "be deprioritised and analyze the trade-offs."
    )
    assert hard > trivial
    assert 0.0 <= hard <= 1.0


def test_complexity_clamped_to_one_for_long_prompt() -> None:
    long_prompt = "Explain why " * 1000
    assert estimate_complexity_score(long_prompt) <= 1.0


def test_worker_model_quality_uses_default_when_missing() -> None:
    assert worker_model_quality({}) == DEFAULT_MODEL_QUALITY


def test_worker_model_quality_clamps_bad_values() -> None:
    assert worker_model_quality({"model_quality": "1.5"}) == 1.0
    assert worker_model_quality({"model_quality": -0.5}) == DEFAULT_MODEL_QUALITY
    assert worker_model_quality({"model_quality": "abc"}) == DEFAULT_MODEL_QUALITY


def test_quality_term_zero_when_nu_or_complexity_zero() -> None:
    assert quality_routing_term(nu=0.0, complexity=0.9, model_quality=0.1) == 0.0
    assert quality_routing_term(nu=10.0, complexity=0.0, model_quality=0.1) == 0.0


def test_quality_term_zero_for_top_quality_worker() -> None:
    assert quality_routing_term(nu=20.0, complexity=0.9, model_quality=1.0) == 0.0


def test_quality_term_penalises_weak_workers_for_complex_prompts() -> None:
    weak_cost = quality_routing_term(nu=20.0, complexity=0.9, model_quality=0.2)
    strong_cost = quality_routing_term(nu=20.0, complexity=0.9, model_quality=0.9)
    assert weak_cost > strong_cost > 0


def test_choose_worker_picks_strong_for_complex_prompt_when_nu_enabled() -> None:
    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=20.0)
    try:
        complex_prompt = (
            "Explain step by step why a memory-pressured worker should "
            "be deprioritised and analyze the trade-offs in detail."
        )
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": complex_prompt}],
        }
        weak = _worker("weak", model_quality=0.2)
        strong = _worker("strong", model_quality=1.0)

        choice = choose_worker(request, [weak, strong])

        assert choice.worker.node_id == "strong"
        assert choice.complexity > 0.0
        assert choice.quality_term == 0.0
    finally:
        WEIGHTS.update(nu=original_nu)


def test_choose_worker_ignores_quality_when_nu_zero() -> None:
    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=0.0)
    try:
        complex_prompt = "Explain step by step why " * 30
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": complex_prompt}],
        }
        weak = _worker("weak", model_quality=0.2)
        strong = _worker("strong", model_quality=1.0)

        choice = choose_worker(request, [weak, strong])

        assert choice.quality_term == 0.0
    finally:
        WEIGHTS.update(nu=original_nu)


def test_detect_modalities_plain_text() -> None:
    request = {"messages": [{"role": "user", "content": "hello"}]}
    assert detect_request_modalities(request) == {"text"}


def test_detect_modalities_image_request() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
    }
    assert detect_request_modalities(request) == {"text", "image"}


def test_detect_modalities_audio_request() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": "..."}}],
            }
        ]
    }
    assert detect_request_modalities(request) == {"text", "audio"}


def test_worker_modalities_defaults_to_text() -> None:
    assert worker_modalities({}) == {"text"}


def test_worker_modalities_reads_capabilities_too() -> None:
    assert worker_modalities({"capabilities": ["text", "image"]}) == {"text", "image"}


def test_worker_supports_modalities_filters_correctly() -> None:
    text_only = {"modalities": ["text"]}
    vision = {"modalities": ["text", "image"]}
    assert worker_supports_modalities(text_only, {"text"}) is True
    assert worker_supports_modalities(text_only, {"text", "image"}) is False
    assert worker_supports_modalities(vision, {"text", "image"}) is True


def test_choose_worker_filters_image_requests_to_vision_workers() -> None:
    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this picture"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    }
    text_only = _worker("text-only", model_quality=0.5, modalities=["text"])
    vision = _worker("vision", model_quality=0.5, modalities=["text", "image"])

    choice = choose_worker(request, [text_only, vision])

    assert choice.worker.node_id == "vision"


def test_choose_worker_rejects_when_no_modality_match() -> None:
    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    }
    text_only_a = _worker("a", model_quality=0.5, modalities=["text"])
    text_only_b = _worker("b", model_quality=0.5, modalities=["text"])

    with pytest.raises(SchedulingError, match="modalities"):
        choose_worker(request, [text_only_a, text_only_b])


def _worker(
    node_id: str,
    *,
    model_quality: float,
    modalities: list[str] | None = None,
) -> WorkerSnapshot:
    telemetry = Telemetry(
        qw=0,
        sw_by_bucket={"<=256": 10.0, "<=1024": 10.0},
        mw=0.0,
        jw=0.0,
        theta_w=0,
        prefix_cache=PrefixCacheTelemetry.from_config(
            PrefixHashConfig(model_id="test-model"), []
        ),
        sprefill_tokens_per_sec=0.0,
    )
    metadata: dict = {
        "model_quality": model_quality,
        "models": [{"id": "test-model", "loaded": True}],
    }
    if modalities is not None:
        metadata["modalities"] = modalities
    return WorkerSnapshot(
        node_id=node_id,
        address=f"http://{node_id}",
        metadata=metadata,
        telemetry=telemetry,
    )
