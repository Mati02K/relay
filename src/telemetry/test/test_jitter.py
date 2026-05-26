from typing import cast

from membership.base import MembershipLayer
from telemetry.jitter import JitterProbe


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


def _probe(alpha: float = 0.15) -> JitterProbe:
    return JitterProbe(membership=cast(MembershipLayer, _DummyMembership()), alpha=alpha)


class _DummyMembership:
    """Minimal membership stub used only to satisfy the constructor."""

    async def getByPrefix(self, prefix: str) -> dict[str, str]:
        return {}
