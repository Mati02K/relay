"""Tests for the chart-driven, skill-aware router."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from coordinator import scheduler as scheduler_module
from coordinator.router import (
    Classification,
    classify,
    detect_request_modalities,
    detect_skills,
    estimate_complexity_score,
    lookup,
    quality_routing_term,
    reload_chart,
)
from coordinator.scheduler import (
    WEIGHTS,
    SchedulingError,
    choose_worker,
    get_mode,
    set_mode,
)
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import PrefixHashConfig
from telemetry.schemas import PrefixCacheTelemetry, Telemetry


@pytest.fixture(autouse=True)
def _reset_chart() -> None:
    """Reload the model chart between tests so env-driven overrides do not leak."""
    reload_chart()


@pytest.fixture(autouse=True)
def _reset_mode() -> Iterator[None]:
    """Force ``cost`` mode + a fresh RR counter between tests."""
    set_mode("cost")
    scheduler_module._reset_round_robin()
    yield
    set_mode("cost")
    scheduler_module._reset_round_robin()


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


def test_quality_term_zero_when_nu_or_complexity_zero() -> None:
    assert quality_routing_term(nu=0.0, complexity=0.9, model_quality=0.1) == 0.0
    assert quality_routing_term(nu=10.0, complexity=0.0, model_quality=0.1) == 0.0


def test_quality_term_zero_for_top_quality_worker() -> None:
    assert quality_routing_term(nu=20.0, complexity=0.9, model_quality=1.0) == 0.0


def test_quality_term_penalises_weak_workers_for_complex_prompts() -> None:
    weak_cost = quality_routing_term(nu=20.0, complexity=0.9, model_quality=0.2)
    strong_cost = quality_routing_term(nu=20.0, complexity=0.9, model_quality=0.9)
    assert weak_cost > strong_cost > 0


def test_detect_skills_plain_prompt_is_instruct_only() -> None:
    assert detect_skills("what's the capital of France") == frozenset({"instruct"})


def test_detect_skills_picks_up_coding_from_fences() -> None:
    skills = detect_skills("Fix this:\n```python\nprint('hi')\n```")
    assert "coding" in skills
    assert "instruct" in skills


def test_detect_skills_picks_up_coding_from_keywords() -> None:
    skills = detect_skills("write a python function that reverses a string")
    assert "coding" in skills


def test_detect_skills_picks_up_reasoning() -> None:
    skills = detect_skills("Compare and contrast eventual consistency and linearizability.")
    assert "reasoning" in skills


def test_classify_returns_complexity_and_skills() -> None:
    request = {
        "messages": [
            {"role": "user", "content": "explain why and analyze the trade-offs of CAP"},
        ]
    }
    result = classify(request)
    assert isinstance(result, Classification)
    assert 0.0 < result.complexity <= 1.0
    assert "instruct" in result.skills_needed


def test_lookup_returns_chart_entry_for_known_model() -> None:
    entry = lookup("qwen2.5-coder-7b")
    assert "coding" in entry.skills
    assert entry.quality > 0.5


def test_lookup_derives_entry_for_unknown_model() -> None:
    entry = lookup("frobnicator-coder-3b-q4_0")
    assert "coding" in entry.skills
    assert entry.quant == "q4_0"
    assert entry.params_b == pytest.approx(3.0)


def test_choose_worker_routes_coding_prompt_to_coder() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Write a python function that prints fibonacci",
            }
        ],
    }
    coder = _worker("coder", model_id="qwen2.5-coder-7b")
    chat = _worker("chat", model_id="qwen2.5-1.5b")

    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=20.0)
    try:
        choice = choose_worker(request, [coder, chat])
    finally:
        WEIGHTS.update(nu=original_nu)

    assert choice.worker.node_id == "coder"
    assert "coding" in choice.skills_needed
    assert choice.skill_match is True


def test_choose_worker_falls_back_to_all_when_no_skill_match() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Write a python function that prints fibonacci",
            }
        ],
    }
    chat_a = _worker("chat-a", model_id="qwen2.5-1.5b")
    chat_b = _worker("chat-b", model_id="qwen2.5-0.5b")

    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=20.0)
    try:
        choice = choose_worker(request, [chat_a, chat_b])
    finally:
        WEIGHTS.update(nu=original_nu)

    assert choice.worker.node_id == "chat-a"
    assert choice.skill_match is False


def test_skill_filter_disabled_when_nu_zero() -> None:
    """nu=0 ablates the whole chart: no skill filter, no quality term."""
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Write a python function that prints fibonacci",
            }
        ],
    }
    coder = _worker("coder", model_id="qwen2.5-coder-1.5b")
    strong_chat = _worker("strong-chat", model_id="qwen2.5-7b")

    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=0.0)
    try:
        choice = choose_worker(request, [coder, strong_chat])
    finally:
        WEIGHTS.update(nu=original_nu)

    # With nu=0 the skill filter is off, so the coder no longer wins by
    # specialization. Both workers are scored on the base 5-term cost only.
    # Tie-breaker is node_id, so "coder" < "strong-chat" alphabetically wins
    # — the key signal here is that quality_term is exactly 0.
    assert choice.quality_term == 0.0


def test_choose_worker_picks_strong_for_complex_prompt_when_nu_enabled() -> None:
    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=20.0)
    try:
        complex_prompt = (
            "Explain step by step why a memory-pressured worker should "
            "be deprioritised and analyze the trade-offs in detail."
        )
        request = {
            "messages": [{"role": "user", "content": complex_prompt}],
        }
        weak = _worker("weak", model_id="qwen2.5-0.5b")
        strong = _worker("strong", model_id="qwen2.5-7b")

        choice = choose_worker(request, [weak, strong])

        assert choice.worker.node_id == "strong"
        assert choice.complexity > 0.0
    finally:
        WEIGHTS.update(nu=original_nu)


def test_choose_worker_ignores_quality_when_nu_zero() -> None:
    original_nu = WEIGHTS.nu
    WEIGHTS.update(nu=0.0)
    try:
        complex_prompt = "Explain step by step why " * 30
        request = {
            "messages": [{"role": "user", "content": complex_prompt}],
        }
        weak = _worker("weak", model_id="qwen2.5-0.5b")
        strong = _worker("strong", model_id="qwen2.5-7b")

        choice = choose_worker(request, [weak, strong])

        assert choice.quality_term == 0.0
    finally:
        WEIGHTS.update(nu=original_nu)


def test_default_mode_is_cost() -> None:
    assert get_mode() == "cost"


def test_set_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        set_mode("hyperscale")


def test_round_robin_rotates_across_eligible_workers() -> None:
    request = {
        "messages": [{"role": "user", "content": "anything"}],
    }
    a = _worker("pi-a", model_id="qwen2.5-1.5b")
    b = _worker("pi-b", model_id="qwen2.5-7b")
    c = _worker("pi-c", model_id="qwen2.5-coder-7b")

    set_mode("round_robin")
    picks = [choose_worker(request, [a, b, c]).worker.node_id for _ in range(6)]

    assert picks == ["pi-a", "pi-b", "pi-c", "pi-a", "pi-b", "pi-c"]


def test_round_robin_returns_zeroed_cost_diagnostics() -> None:
    request = {
        "messages": [{"role": "user", "content": "write a python function"}],
    }
    coder = _worker("a-coder", model_id="qwen2.5-coder-7b")

    set_mode("round_robin")
    choice = choose_worker(request, [coder])

    # All scoring fields are intentionally zero in RR — the dashboard reads
    # these to know the cost path didn't run.
    assert choice.cost == 0.0
    assert choice.complexity == 0.0
    assert choice.quality_term == 0.0
    assert choice.skill_match is False


def test_round_robin_still_honours_input_modality_filter() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    }
    text_only_a = _worker("a", model_id="qwen2.5-7b")
    text_only_b = _worker("b", model_id="qwen2.5-1.5b")

    set_mode("round_robin")
    with pytest.raises(SchedulingError, match="modalities"):
        choose_worker(request, [text_only_a, text_only_b])


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


def test_choose_worker_rejects_image_when_no_vision_worker() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ],
    }
    text_only_a = _worker("a", model_id="qwen2.5-7b")
    text_only_b = _worker("b", model_id="qwen2.5-1.5b")

    with pytest.raises(SchedulingError, match="modalities"):
        choose_worker(request, [text_only_a, text_only_b])


def _worker(node_id: str, *, model_id: str) -> WorkerSnapshot:
    telemetry = Telemetry(
        qw=0,
        mw=0.0,
        jw=0.0,
        theta_w=0,
        prefix_cache=PrefixCacheTelemetry.from_config(PrefixHashConfig(model_id="test-model"), []),
        sprefill_tokens_per_sec=0.0,
    )
    metadata: dict[str, Any] = {
        "models": [{"id": model_id, "loaded": True, "quant": "q4_k_m"}],
    }
    return WorkerSnapshot(
        node_id=node_id,
        address=f"http://{node_id}",
        metadata=metadata,
        telemetry=telemetry,
    )
