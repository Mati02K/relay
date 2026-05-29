from abc import ABC, abstractmethod


class NetworkLayer(ABC):
    """Abstract interface for resolving this node's address on the overlay network.

    Today the only consumer is :meth:`getMyAddress`, used by the worker and
    coordinator to learn the IP they should advertise into etcd metadata.
    The class survives as an ABC so different deployment topologies
    (Tailscale, plain LAN) can plug in without changing call sites.
    """

    @abstractmethod
    def getMyAddress(self) -> str:
        """Return the stable IP address of this device on the overlay network."""
