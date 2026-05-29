from typing import cast

from membership.base import MembershipLayer
from telemetry.jitter import (
    FAILURE_PENALTY_BASE_MS,
    FAILURE_PENALTY_MAX_MS,
    JitterProbe,
)


def test_first_sample_initializes_rtt_and_zero_jitter() -> None:
    probe = _probe(alpha=0.15)
    probe.observe("w1", rtt_ms=10.0)

    assert probe.get_jitter_ms("w1") == 0.0


def test_jitter_grows_when_rtt_varies() -> None:
    probe = _probe(alpha=0.15)
    for rtt in (10.0, 50.0, 10.0, 60.0, 10.0):
        probe.observe("w1", rtt_ms=rtt)

    jitter = probe.get_jitter_ms("w1")
    assert jitter is not None
    assert jitter > 5.0


def test_jitter_stays_near_zero_for_stable_rtt() -> None:
    probe = _probe(alpha=0.15)
    for _ in range(20):
        probe.observe("stable", rtt_ms=8.0)

    jitter = probe.get_jitter_ms("stable")
    assert jitter is not None
    assert jitter < 0.5


def test_unknown_worker_returns_none() -> None:
    probe = _probe()
    assert probe.get_jitter_ms("never-probed") is None


def test_ema_update_uses_alpha() -> None:
    probe = _probe(alpha=0.5)
    probe.observe("w1", rtt_ms=10.0)
    probe.observe("w1", rtt_ms=20.0)

    # After two samples with alpha=0.5: rtt_ema = 0.5*20 + 0.5*10 = 15
    # deviation against the *previous* rtt_ema (10) is |20-10|=10
    # jitter_ema = 0.5*10 + 0.5*0 = 5
    assert probe.get_jitter_ms("w1") == 5.0


def test_first_failure_seeds_jitter_to_base_penalty() -> None:
    probe = _probe(alpha=0.5)
    probe.observe_failure("w1")

    # samples == 0 branch seeds jitter_ema directly to the first penalty.
    assert probe.get_jitter_ms("w1") == FAILURE_PENALTY_BASE_MS


def test_consecutive_failures_ramp_penalty_until_cap() -> None:
    probe = _probe(alpha=1.0)
    # alpha=1.0 makes jitter_ema equal the latest penalty exactly,
    # so we can read the ramp directly.
    probe.observe_failure("w1")
    assert probe.get_jitter_ms("w1") == FAILURE_PENALTY_BASE_MS

    probe.observe_failure("w1")
    assert probe.get_jitter_ms("w1") == 2 * FAILURE_PENALTY_BASE_MS

    probe.observe_failure("w1")
    assert probe.get_jitter_ms("w1") == 3 * FAILURE_PENALTY_BASE_MS

    # Drive past the cap; the penalty should saturate at FAILURE_PENALTY_MAX_MS.
    cap_after = int(FAILURE_PENALTY_MAX_MS // FAILURE_PENALTY_BASE_MS) + 2
    for _ in range(cap_after):
        probe.observe_failure("w1")
    assert probe.get_jitter_ms("w1") == FAILURE_PENALTY_MAX_MS


def test_successful_probe_resets_consecutive_failures() -> None:
    probe = _probe(alpha=1.0)
    probe.observe_failure("w1")
    probe.observe_failure("w1")
    probe.observe_failure("w1")
    # Third failure: penalty was 3 * base, so jitter_ema sits there with alpha=1.0.
    assert probe.get_jitter_ms("w1") == 3 * FAILURE_PENALTY_BASE_MS

    # A successful probe resets the counter without altering the penalty escalation.
    probe.observe("w1", rtt_ms=20.0)

    probe.observe_failure("w1")
    # Counter restarts at 1, so next failure's penalty is base again, not 4 * base.
    # alpha=1.0 means jitter_ema = base.
    assert probe.get_jitter_ms("w1") == FAILURE_PENALTY_BASE_MS


def test_failures_do_not_pollute_rtt_ema_so_recovery_is_clean() -> None:
    probe = _probe(alpha=0.5)
    for _ in range(5):
        probe.observe_failure("w1")

    # First successful probe seeds rtt_ema to the observed RTT, not a polluted blend.
    probe.observe("w1", rtt_ms=20.0)

    # A second probe at the same RTT should produce ~zero deviation, so jitter_ema
    # strictly decreases from its post-failure value (the recovery is unblocked).
    before = probe.get_jitter_ms("w1")
    assert before is not None
    probe.observe("w1", rtt_ms=20.0)
    after = probe.get_jitter_ms("w1")
    assert after is not None
    assert after < before


def _probe(alpha: float = 0.15) -> JitterProbe:
    return JitterProbe(membership=cast(MembershipLayer, _DummyMembership()), alpha=alpha)


class _DummyMembership:
    """Minimal membership stub used only to satisfy the constructor."""

    async def getByPrefix(self, prefix: str) -> dict[str, str]:
        return {}
