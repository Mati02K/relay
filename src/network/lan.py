import socket

from loguru import logger

from .base import NetworkLayer


class LANNetwork(NetworkLayer):
    """NetworkLayer implementation that resolves the host's non-loopback IPv4 address."""

    def __init__(self) -> None:
        """Initialize the LAN network layer."""
        logger.info("LANNetwork initialised")

    def getMyAddress(self) -> str:
        """Return the outbound LAN IPv4 address of this host.

        Uses a connect-without-sending trick: the kernel selects the correct
        outbound interface for a UDP socket without actually transmitting any
        packets.  This works reliably on Ubuntu and other Debian-based systems
        where the hostname resolves to a loopback address in ``/etc/hosts``
        (e.g. ``127.0.1.1``), which defeats ``gethostname``-based approaches.
        Falls back to ``gethostname`` resolution, then ``127.0.0.1``.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                addr = sock.getsockname()[0]
                if addr and not addr.startswith("127."):
                    logger.debug("Local address resolved via routing | ip={}", addr)
                    return addr
        except OSError:
            pass

        for addrInfo in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = addrInfo[4][0]
            if isinstance(addr, str) and not addr.startswith("127."):
                logger.debug("Local address resolved via hostname | ip={}", addr)
                return addr

        logger.warning("No non-loopback IPv4 found, falling back to 127.0.0.1")
        return "127.0.0.1"
