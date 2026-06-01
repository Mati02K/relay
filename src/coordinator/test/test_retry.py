from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from coordinator import inflight
from coordinator.main import (
    OpenedStream,
    WorkerUnreachable,
    _open_stream_with_retry,
)
from coordinator.scheduler import SchedulingError
from coordinator.worker_registry import WorkerSnapshot
from telemetry.prefix_cache import PrefixHashConfig
from telemetry.schemas import PrefixCacheTelemetry, Telemetry


@pytest.fixture(autouse=True)
def _reset_inflight() -> Any:
    """Clear coordinator in-flight counts so reservations don't leak across tests.

    ``_open_stream_with_retry`` reserves on the chosen worker; the matching
    release lives in the streaming generator, which these unit tests don't drive,
    so the global counter must be reset around each test.
    """
    inflight.reset()
    yield
    inflight.reset()


def test_first_worker_reachable_returns_attempt_one() -> None:
    workers = [_worker("a", qw=0), _worker("b", qw=5)]
    opener = _OpenerScript({})  # nobody fails

    choice, opened, attempts = asyncio.run(
        _open_stream_with_retry(_request(), workers, opener, max_attempts=3)
    )

    assert attempts == 1
    assert choice.worker.node_id == "a"  # lowest cost wins
    assert opener.called_with == ["a"]
    assert opened is not None


def test_skips_unreachable_worker_and_retries_next_best() -> None:
    workers = [_worker("a", qw=0), _worker("b", qw=5)]
    opener = _OpenerScript({"a": _connect_error()})

    choice, _, attempts = asyncio.run(
        _open_stream_with_retry(_request(), workers, opener, max_attempts=3)
    )

    assert attempts == 2
    assert choice.worker.node_id == "b"
    assert opener.called_with == ["a", "b"]


def test_exhausts_all_workers_raises_last_unreachable() -> None:
    workers = [_worker("a", qw=0), _worker("b", qw=5)]
    opener = _OpenerScript(
        {"a": _connect_error("a"), "b": _connect_error("b")},
    )

    with pytest.raises(WorkerUnreachable) as info:
        asyncio.run(_open_stream_with_retry(_request(), workers, opener, max_attempts=3))

    assert info.value.node_id == "b"  # last failure is what's surfaced
    assert opener.called_with == ["a", "b"]


def test_max_attempts_bound_stops_retrying_even_with_workers_remaining() -> None:
    workers = [_worker("a", qw=0), _worker("b", qw=5), _worker("c", qw=10)]
    opener = _OpenerScript({"a": _connect_error(), "b": _connect_error(), "c": _connect_error()})

    with pytest.raises(WorkerUnreachable):
        asyncio.run(_open_stream_with_retry(_request(), workers, opener, max_attempts=2))

    assert opener.called_with == ["a", "b"]  # never reached c


def test_no_workers_raises_scheduling_error() -> None:
    opener = _OpenerScript({})

    with pytest.raises(SchedulingError):
        asyncio.run(_open_stream_with_retry(_request(), [], opener, max_attempts=3))

    assert opener.called_with == []


def test_failed_worker_excluded_from_choose_worker_on_retry() -> None:
    # Tie-broken by node_id when costs are equal; a sorts before b.
    workers = [_worker("a", qw=0), _worker("b", qw=0)]
    opener = _OpenerScript({"a": _connect_error()})

    choice, _, attempts = asyncio.run(
        _open_stream_with_retry(_request(), workers, opener, max_attempts=3)
    )

    assert attempts == 2
    assert choice.worker.node_id == "b"  # a is excluded, b wins by default


class _OpenerScript:
    """Fake opener: returns OpenedStream sentinel for success, raises for failure."""

    def __init__(self, failures: dict[str, Exception]) -> None:
        self._failures = failures
        self.called_with: list[str] = []

    def __call__(
        self,
        worker: WorkerSnapshot,
    ) -> Awaitable[OpenedStream]:
        self.called_with.append(worker.node_id)
        if worker.node_id in self._failures:
            return _raise(self._failures[worker.node_id])
        return _ok()


async def _raise(exc: Exception) -> OpenedStream:
    raise exc


async def _ok() -> OpenedStream:
    return cast(OpenedStream, _SENTINEL_OPENED)


_SENTINEL_OPENED: Any = object()


def _connect_error(node_id: str = "x") -> WorkerUnreachable:
    return WorkerUnreachable(node_id, f"http://{node_id}", RuntimeError("connect refused"))


def _request() -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }


def _worker(node_id: str, *, qw: int) -> WorkerSnapshot:
    telemetry = Telemetry(
        qw=qw,
        mw=0.0,
        jw=0.0,
        theta_w=0,
        prefix_cache=PrefixCacheTelemetry.from_config(PrefixHashConfig(model_id="test-model"), []),
        sprefill_tokens_per_sec=0.0,
    )
    return WorkerSnapshot(
        node_id=node_id,
        address=f"http://{node_id}",
        metadata={"models": [{"id": "test-model", "loaded": True}]},
        telemetry=telemetry,
    )


# Make the script callable with the helper's expected signature: Callable[[W], Awaitable[OS]].
_: Callable[[WorkerSnapshot], Awaitable[OpenedStream]] = _OpenerScript({})
