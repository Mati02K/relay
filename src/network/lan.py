import socket

from loguru import logger

from .base import NetworkLayer


class LANNetwork(NetworkLayer):
    """NetworkLayer implementation that resolves the host's non-loopback IPv4 address."""

    def __init__(self) -> None:
        """Initialize the LAN network layer."""
        logger.info("LANNetwork initialised")

    def getMyAddress(self) -> str:
        """Return the first non-loopback IPv4 address of this host."""
        for addrInfo in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = addrInfo[4][0]
            if isinstance(addr, str) and not addr.startswith("127."):
                logger.debug("Local address resolved | ip={}", addr)
                return addr
        logger.warning("No non-loopback IPv4 found, falling back to 127.0.0.1")
        return "127.0.0.1"
