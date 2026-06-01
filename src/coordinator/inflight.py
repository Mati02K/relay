"""Coordinator-owned in-flight request accounting per worker.

The scheduler's queue signal (``q_w``) is sourced from here rather than from
worker-published telemetry. The active coordinator dispatches every request, so
an in-memory counter is real-time and authoritative for scheduled traffic — no
etcd round-trip and no engine-scrape lag.

Lifecycle per request: :func:`reserve` when the coordinator schedules a worker,
:func:`release` exactly once when the request fully completes — success,
failure, or client disconnect. asyncio is single-threaded, so the dict
mutations here are atomic as long as no ``await`` splits a read-modify-write;
callers must keep the read -> choose -> reserve sequence free of awaits so
concurrent requests observe each other's reservations.

State is in-memory only by design: a coordinator restart resets every count to
zero, which is self-healing — no stale reservation can outlive the process.
"""

from __future__ import annotations

_inFlightByNode: dict[str, int] = {}


def depth(nodeId: str) -> int:
    """Return the number of requests the coordinator currently has in flight to a worker."""
    return _inFlightByNode.get(nodeId, 0)


def reserve(nodeId: str) -> None:
    """Count one newly scheduled request against a worker."""
    _inFlightByNode[nodeId] = _inFlightByNode.get(nodeId, 0) + 1


def release(nodeId: str) -> None:
    """Release one completed request; clamps at zero to absorb a double release."""
    current = _inFlightByNode.get(nodeId, 0)
    if current <= 0:
        return
    _inFlightByNode[nodeId] = current - 1


def reset() -> None:
    """Drop all in-flight counts. For tests and a clean restart of accounting."""
    _inFlightByNode.clear()
