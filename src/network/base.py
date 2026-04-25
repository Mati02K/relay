from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PeerInfo:
    """Describes a discovered peer device on the network."""

    id: str
    address: str
    online: bool
    name: str


class NetworkLayer(ABC):
    """Abstract interface for peer discovery and address resolution."""

    @abstractmethod
    def getMyAddress(self) -> str:
        """Return the stable IP address of this device on the overlay network."""

    @abstractmethod
    def getPeerAddress(self, peerId: str) -> str:
        """Return the current IP address of a known peer by its device ID."""

    @abstractmethod
    def discoverPeers(self) -> list[PeerInfo]:
        """Scan the network and return all currently reachable peers."""

    @abstractmethod
    def isPeerReachable(self, peerId: str) -> bool:
        """Return whether the given peer is currently online and reachable."""
