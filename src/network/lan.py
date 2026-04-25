import socket
import time

from loguru import logger
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

from .base import NetworkLayer, PeerInfo

SERVICE_TYPE = "_homecluster._tcp.local."
BROWSE_TIMEOUT_SECONDS = 3


class LANNetwork(NetworkLayer):
    """NetworkLayer implementation using mDNS/DNS-SD for local subnet discovery."""

    def __init__(self) -> None:
        """Initialize the mDNS client for local peer discovery."""
        self._zeroconf = Zeroconf()
        logger.info("LANNetwork initialised | serviceType={}", SERVICE_TYPE)

    def getMyAddress(self) -> str:
        """Return the first non-loopback IPv4 address of this host."""
        for addrInfo in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = addrInfo[4][0]
            if not addr.startswith("127."):
                logger.debug("Local address resolved | ip={}", addr)
                return addr
        logger.warning("No non-loopback IPv4 found, falling back to 127.0.0.1")
        return "127.0.0.1"

    def getPeerAddress(self, peerId: str) -> str:
        """Resolve a peer's IP address by hostname via mDNS on the local subnet."""
        info = ServiceInfo(SERVICE_TYPE, f"{peerId}.{SERVICE_TYPE}")
        if info.request(self._zeroconf, 3000):
            address = socket.inet_ntoa(info.addresses[0])
            logger.debug("LAN peer resolved | peerId={} ip={}", peerId, address)
            return address
        logger.warning("LAN peer not found | peerId={}", peerId)
        raise RuntimeError(f"Peer {peerId} not found on LAN")

    def discoverPeers(self) -> list[PeerInfo]:
        """Browse mDNS for BROWSE_TIMEOUT_SECONDS and return all HomCluster peers found."""
        discovered: list[PeerInfo] = []
        logger.info("Starting LAN peer discovery | timeout={}s", BROWSE_TIMEOUT_SECONDS)

        class _Handler:
            def add_service(
                self, zc: Zeroconf, serviceType: str, name: str
            ) -> None:
                """Handle a newly discovered mDNS service entry."""
                info = zc.get_service_info(serviceType, name)
                if info:
                    addr = socket.inet_ntoa(info.addresses[0])
                    peerId = name.replace(f".{serviceType}", "")
                    logger.debug("LAN peer discovered | peerId={} ip={}", peerId, addr)
                    discovered.append(
                        PeerInfo(id=peerId, address=addr, online=True, name=peerId)
                    )

            def remove_service(self, *_: object) -> None:
                """Ignore service removal events during a one-shot browse."""

            def update_service(self, *_: object) -> None:
                """Ignore service update events during a one-shot browse."""

        ServiceBrowser(self._zeroconf, SERVICE_TYPE, _Handler())
        time.sleep(BROWSE_TIMEOUT_SECONDS)
        logger.info("LAN peer discovery complete | found={}", len(discovered))
        return discovered

    def isPeerReachable(self, peerId: str) -> bool:
        """Return whether a peer is currently advertising itself via mDNS on the LAN."""
        try:
            self.getPeerAddress(peerId)
            return True
        except RuntimeError:
            return False
