from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from membership import relay_pb2
from membership.etcd import EtcdMembership


class _FakeStub:
    """Mock MembershipServiceStub that drives the HoldLease/PutWithLease flow."""

    def __init__(self, lease_id: int = 42) -> None:
        self._lease_id = lease_id
        self.put_calls: list[tuple[str, str, int]] = []
        self.stream_closed = asyncio.Event()
        self.stream_started = asyncio.Event()

    def HoldLease(self, request: relay_pb2.LeaseRequest) -> AsyncIterator[Any]:
        self.stream_started.set()
        return _LeaseStream(self._lease_id, self.stream_closed)

    async def PutWithLease(self, request: relay_pb2.LeaseKVPair) -> relay_pb2.Empty:
        self.put_calls.append((request.key, request.value, int(request.lease_id)))
        return relay_pb2.Empty()


class _LeaseStream:
    """Async iterator that yields one LeaseStatus then blocks until cancelled."""

    def __init__(self, lease_id: int, closed: asyncio.Event) -> None:
        self._lease_id = lease_id
        self._closed = closed
        self._sent = False

    def __aiter__(self) -> _LeaseStream:
        return self

    async def __anext__(self) -> Any:
        if not self._sent:
            self._sent = True
            return relay_pb2.LeaseStatus(lease_id=self._lease_id, alive=True)
        # Block forever (until the lease task is cancelled) to mimic a real
        # server-stream that stays open for the worker's lifetime.
        await self._closed.wait()
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_hold_lease_returns_lease_id_from_stream() -> None:
    membership = _membership_with(_FakeStub(lease_id=7))

    lease_id = await membership.holdLease(ttlSeconds=10)

    assert lease_id == 7
    await membership.close()


@pytest.mark.asyncio
async def test_hold_lease_is_idempotent() -> None:
    stub = _FakeStub(lease_id=99)
    membership = _membership_with(stub)

    first = await membership.holdLease(ttlSeconds=10)
    second = await membership.holdLease(ttlSeconds=10)

    assert first == second == 99
    await membership.close()


@pytest.mark.asyncio
async def test_put_with_lease_before_hold_raises() -> None:
    membership = _membership_with(_FakeStub())

    with pytest.raises(RuntimeError, match="putWithLease called before holdLease"):
        await membership.putWithLease("/k", "v")
    await membership.close()


@pytest.mark.asyncio
async def test_put_with_lease_attaches_current_lease() -> None:
    stub = _FakeStub(lease_id=11)
    membership = _membership_with(stub)
    await membership.holdLease(ttlSeconds=10)

    await membership.putWithLease("/relay/workers/a/metadata", '{"x":1}')
    await membership.putWithLease("/relay/workers/a/telemetry", '{"qw":3}')

    assert stub.put_calls == [
        ("/relay/workers/a/metadata", '{"x":1}', 11),
        ("/relay/workers/a/telemetry", '{"qw":3}', 11),
    ]
    await membership.close()


@pytest.mark.asyncio
async def test_close_cancels_lease_stream() -> None:
    stub = _FakeStub(lease_id=5)
    membership = _membership_with(stub)
    await membership.holdLease(ttlSeconds=10)
    assert stub.stream_started.is_set()

    await membership.close()

    # After close the lease task is gone; subsequent putWithLease should raise.
    with pytest.raises(RuntimeError):
        await membership.putWithLease("/k", "v")


def _membership_with(stub: _FakeStub) -> EtcdMembership:
    membership = EtcdMembership.__new__(EtcdMembership)
    membership._channel = _NoopChannel()  # type: ignore[attr-defined]
    membership._stub = stub  # type: ignore[attr-defined]
    membership._leaseId = None  # type: ignore[attr-defined]
    membership._leaseStream = None  # type: ignore[attr-defined]
    membership._leaseTask = None  # type: ignore[attr-defined]
    membership._leaseReady = asyncio.Event()  # type: ignore[attr-defined]
    return membership


class _NoopChannel:
    """Stand-in for ``grpc.aio.Channel`` that supports ``await close()``."""

    async def close(self) -> None:
        return None
