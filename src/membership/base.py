from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Callable


@dataclass
class Node:
    """Represents a single registered cluster member."""

    id: str
    metadata: dict


class MembershipLayer(ABC):
    """Abstract interface for cluster membership and distributed coordination."""

    @abstractmethod
    async def register(self, nodeId: str, metadata: dict) -> None:
        """Register this node in the cluster with the given metadata."""

    @abstractmethod
    async def deregister(self, nodeId: str) -> None:
        """Remove this node from the cluster registry."""

    @abstractmethod
    async def getAliveMembers(self) -> list[Node]:
        """Return all currently registered nodes in the cluster."""

    @abstractmethod
    def watchMembership(self, callback: Callable[[str, Node], None]) -> None:
        """Subscribe to membership changes; callback receives (event_type, node)."""

    @abstractmethod
    def holdLeadership(self) -> AsyncIterator[bool]:
        """Stream leadership status. Yields True once when elected; ends when leadership is lost."""

    @abstractmethod
    async def put(self, key: str, value: str) -> None:
        """Write a key-value pair to the distributed store."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Read a value from the distributed store; returns None if key not found."""

    @abstractmethod
    async def getByPrefix(self, prefix: str) -> dict[str, str]:
        """Return all key-value pairs whose keys start with the given prefix."""

    @abstractmethod
    async def holdLease(self, ttlSeconds: int) -> int:
        """Grant a session-backed lease and keep it alive in the background.

        Returns the lease id. All ``putWithLease`` calls in this client are tied
        to this lease until ``close`` is called or the underlying stream dies.
        """

    @abstractmethod
    async def putWithLease(self, key: str, value: str) -> None:
        """Write a key bound to the currently-held lease; auto-deletes on expiry."""

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection."""
