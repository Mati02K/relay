"""Unit tests for the §3.2 cost function. Run with: pytest src/coordinator/test_scheduler.py"""

from __future__ import annotations

import pytest

from coordinator.scheduler import (
    CostWeights,
    compute_cost,
    estimate_complexity_score,
    estimate_ttft_ms,
    messages_to_prompt_text,
    pick_worker,
)
from coordinator.worker_registry import WorkerState
from worker.inference.base import (
    Telemetry,
    prefix_chunk_hashes_from_text,
)


def make_worker(
    node_id: str = "w",
    *,
    qw: int = 0,
    sw_by_bucket: dict | None = None,
    mw: float = 0.0,
    jw: float = 0.0,
    theta_w: int = 0,
    prefix_chunk_hashes: list[str] | None = None,
    online: bool = True,
    compute_strength: float | None = None,
    model_quality: float | None = None,
    rtt_ms_ema: float = 5.0,
    sprefill: float = 100.0,
) -> WorkerState:
    metadata = {"role": "worker", "url": f"http://{node_id}:9090"}
    if compute_strength is not None:
        metadata["compute_strength"] = compute_strength
    if model_quality is not None:
        metadata["model_quality"] = model_quality
    return WorkerState(
        node_id=node_id,
        url=f"http://{node_id}:9090",
        telemetry=Telemetry(
            qw=qw,
            sw_by_bucket=sw_by_bucket or {},
            mw=mw,
            jw=jw,
            theta_w=theta_w,
            prefix_chunk_hashes=prefix_chunk_hashes or [],
            sprefill_tokens_per_sec=sprefill,
        ),
        rtt_ms_ema=rtt_ms_ema,
        jitter_ms_ema=jw,
        last_seen_ms=1.0,
        online=online,
        metadata=metadata,
    )


def equal_weights(**overrides: float) -> CostWeights:
    """CostWeights with all weights = 1.0 unless overridden, so terms are directly comparable."""
    base: dict = dict(
        alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0, phi=0.0, nu=0.0,
        long_prompt_threshold=1024, jitter_max_ms=100.0,
        complexity_length_norm_tokens=2048,
    )
    base.update(overrides)
    return CostWeights(**base)


class TestMessagesToPromptText:
    def test_concatenates_string_contents(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert messages_to_prompt_text(msgs) == "hello\nworld"

    def test_handles_multimodal_blocks(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        assert messages_to_prompt_text(msgs) == "hi"

    def test_skips_non_text(self) -> None:
        msgs = [
            {"role": "user", "content": None},
            {"role": "user"},
            {"role": "user", "content": "ok"},
        ]
        assert messages_to_prompt_text(msgs) == "ok"


class TestPickWorkerNoWorkers:
    def test_no_workers_returns_none(self) -> None:
        winner, scores = pick_worker("hi", {}, weights=equal_weights())
        assert winner is None
        assert scores == []

    def test_only_offline_workers_returns_none(self) -> None:
        workers = {"w1": make_worker("w1", online=False)}
        winner, _ = pick_worker("hi", workers, weights=equal_weights())
        assert winner is None


class TestSingleTermDominance:
    """Each test isolates ONE cost term to confirm the formula honors it."""

    def test_busier_worker_loses(self) -> None:
        bucket_speed = {"<=256": 50.0}
        idle = make_worker("idle", qw=0, sw_by_bucket=bucket_speed)
        busy = make_worker("busy", qw=1000, sw_by_bucket=bucket_speed)
        winner, _ = pick_worker("short prompt", {"idle": idle, "busy": busy},
                                weights=equal_weights())
        assert winner == "idle"

    def test_worker_with_cached_prefix_wins(self) -> None:
        prompt = "the quick brown fox jumps over the lazy dog " * 20
        cached_hashes = prefix_chunk_hashes_from_text(prompt)
        cached = make_worker("cached", prefix_chunk_hashes=cached_hashes)
        cold = make_worker("cold", prefix_chunk_hashes=[])
        winner, scores = pick_worker(prompt, {"cached": cached, "cold": cold},
                                     weights=equal_weights())
        assert winner == "cached"
        cached_sb = next(s for s in scores if s.node_id == "cached")
        cold_sb = next(s for s in scores if s.node_id == "cold")
        assert cached_sb.overlap == pytest.approx(1.0)
        assert cold_sb.overlap == pytest.approx(0.0)
        assert cached_sb.cache_term < cold_sb.cache_term

    def test_partial_prefix_overlap_is_partial_credit(self) -> None:
        prompt = "x" * 200
        full_hashes = prefix_chunk_hashes_from_text(prompt)
        half = full_hashes[: len(full_hashes) // 2]
        half_worker = make_worker("half", prefix_chunk_hashes=half)
        none_worker = make_worker("none", prefix_chunk_hashes=[])
        full_worker = make_worker("full", prefix_chunk_hashes=full_hashes)
        winner, scores = pick_worker(
            prompt, {"half": half_worker, "none": none_worker, "full": full_worker},
            weights=equal_weights(),
        )
        assert winner == "full"
        score_map = {s.node_id: s.cache_term for s in scores}
        assert score_map["full"] < score_map["half"] < score_map["none"]

    def test_overheated_worker_loses(self) -> None:
        cool = make_worker("cool", theta_w=0)
        hot = make_worker("hot", theta_w=1)
        winner, _ = pick_worker("hi", {"cool": cool, "hot": hot},
                                weights=equal_weights())
        assert winner == "cool"

    def test_jittery_worker_loses(self) -> None:
        stable = make_worker("stable", jw=2.0)
        flaky = make_worker("flaky", jw=80.0)
        winner, _ = pick_worker("hi", {"stable": stable, "flaky": flaky},
                                weights=equal_weights())
        assert winner == "stable"

    def test_memory_pressure_penalizes(self) -> None:
        empty = make_worker("empty", mw=0.1)
        full = make_worker("full", mw=0.95)
        winner, _ = pick_worker("hi", {"empty": empty, "full": full},
                                weights=equal_weights())
        assert winner == "empty"


class TestAblationFlags:
    """Zeroing a weight should make that term irrelevant — the basis for the ablation study."""

    def test_zero_beta_ignores_cache(self) -> None:
        prompt = "x" * 200
        cached = make_worker("cached", qw=10,
                             prefix_chunk_hashes=prefix_chunk_hashes_from_text(prompt),
                             sw_by_bucket={"<=256": 10.0})
        empty = make_worker("empty", qw=0,
                            sw_by_bucket={"<=256": 10.0})
        winner, _ = pick_worker(prompt, {"cached": cached, "empty": empty},
                                weights=equal_weights(beta=0.0))
        assert winner == "empty"

    def test_zero_delta_ignores_jitter(self) -> None:
        stable = make_worker("stable", jw=2.0, qw=5,
                             sw_by_bucket={"<=256": 10.0})
        flaky_but_idle = make_worker("flaky", jw=99.0, qw=0,
                                      sw_by_bucket={"<=256": 10.0})
        winner, _ = pick_worker("hi", {"stable": stable, "flaky": flaky_but_idle},
                                weights=equal_weights(delta=0.0))
        assert winner == "flaky"

    def test_round_robin_breaks_ties_arbitrarily(self) -> None:
        a = make_worker("a", qw=100, mw=0.9, theta_w=1)
        b = make_worker("b", qw=0, mw=0.0, theta_w=0)
        winner, scores = pick_worker("hi", {"a": a, "b": b},
                                     weights=CostWeights.round_robin())
        assert winner in {"a", "b"}
        assert all(s.total == 0.0 for s in scores)


class TestPhaseAwareBias:
    def test_long_prompt_favors_strong_compute(self) -> None:
        long_prompt = "z " * 5000  # ~10000 chars, well above 1024 token threshold
        strong = make_worker("strong", compute_strength=1.0)
        weak = make_worker("weak", compute_strength=0.1)
        weights = equal_weights(phi=10.0)
        winner, scores = pick_worker(long_prompt, {"strong": strong, "weak": weak},
                                     weights=weights)
        assert winner == "strong"
        strong_phase = next(s for s in scores if s.node_id == "strong").phase_term
        weak_phase = next(s for s in scores if s.node_id == "weak").phase_term
        assert weak_phase > strong_phase

    def test_short_prompt_ignores_phase_term(self) -> None:
        short_prompt = "hi"
        strong = make_worker("strong", compute_strength=1.0)
        weak = make_worker("weak", compute_strength=0.1)
        winner, scores = pick_worker(short_prompt, {"strong": strong, "weak": weak},
                                     weights=equal_weights(phi=10.0))
        assert winner in {"strong", "weak"}
        for s in scores:
            assert s.phase_term == 0.0


class TestEstimateTtft:
    def test_idle_worker_low_ttft(self) -> None:
        idle = make_worker("idle", qw=0, sprefill=200.0, rtt_ms_ema=5.0)
        assert estimate_ttft_ms(idle, prompt_tokens=100) < 1000.0

    def test_busy_worker_high_ttft(self) -> None:
        busy = make_worker("busy", qw=10000, sprefill=50.0, rtt_ms_ema=5.0)
        assert estimate_ttft_ms(busy, prompt_tokens=100) > 5000.0

    def test_rtt_adds_to_ttft(self) -> None:
        a = make_worker("a", qw=0, sprefill=200.0, rtt_ms_ema=5.0)
        b = make_worker("b", qw=0, sprefill=200.0, rtt_ms_ema=200.0)
        assert estimate_ttft_ms(b, 100) > estimate_ttft_ms(a, 100)


class TestComplexityScore:
    def test_empty_prompt_is_zero(self) -> None:
        assert estimate_complexity_score("") == 0.0

    def test_short_simple_prompt_is_low(self) -> None:
        assert estimate_complexity_score("hi") < 0.1

    def test_keyword_bumps_score(self) -> None:
        plain = estimate_complexity_score("what is ucsc")
        with_kw = estimate_complexity_score(
            "explain why this happens and analyze the trade-offs"
        )
        assert with_kw > plain

    def test_long_prompt_bumps_score(self) -> None:
        short = estimate_complexity_score("hi there")
        long = estimate_complexity_score("a " * 2000)
        assert long > short


class TestQualityAwareTerm:
    """The RouteLLM-style nu term: complex prompts should prefer high-quality models."""

    def test_complex_prompt_favors_high_quality_model(self) -> None:
        complex_prompt = (
            "Explain why distributed schedulers benefit from cache-aware routing. "
            "Compare and analyze the trade-offs against round-robin in two paragraphs. "
            "Include reasoning about thermal throttling."
        )
        strong = make_worker("strong", model_quality=1.0)
        weak = make_worker("weak", model_quality=0.2)
        winner, scores = pick_worker(complex_prompt, {"strong": strong, "weak": weak},
                                     weights=equal_weights(nu=5.0))
        assert winner == "strong"
        strong_q = next(s for s in scores if s.node_id == "strong").quality_term
        weak_q = next(s for s in scores if s.node_id == "weak").quality_term
        assert weak_q > strong_q

    def test_trivial_prompt_does_not_force_strong_model(self) -> None:
        trivial = "hi"
        strong = make_worker("strong", model_quality=1.0, qw=1000,
                             sw_by_bucket={"<=256": 10.0})
        weak = make_worker("weak", model_quality=0.2, qw=0,
                           sw_by_bucket={"<=256": 10.0})
        winner, _ = pick_worker(trivial, {"strong": strong, "weak": weak},
                                weights=equal_weights(nu=5.0))
        assert winner == "weak"

    def test_zero_nu_disables_quality_routing(self) -> None:
        complex_prompt = "Explain and compare and analyze " * 50
        strong = make_worker("strong", model_quality=1.0, qw=1000,
                             sw_by_bucket={"<=256": 10.0})
        weak = make_worker("weak", model_quality=0.2, qw=0,
                           sw_by_bucket={"<=256": 10.0})
        winner, _ = pick_worker(complex_prompt, {"strong": strong, "weak": weak},
                                weights=equal_weights(nu=0.0))
        assert winner == "weak"
