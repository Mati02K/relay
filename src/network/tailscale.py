import httpx
from loguru import logger

from .base import NetworkLayer, PeerInfo


class TailscaleNetwork(NetworkLayer):
    """NetworkLayer implementation using the Tailscale LocalAPI via Unix domain socket."""

    def __init__(
        self, socketPath: str = "/var/run/tailscale/tailscaled.sock"
    ) -> None:
        """Connect to the local Tailscale daemon socket."""
        transport = httpx.HTTPTransport(uds=socketPath)
        self._client = httpx.Client(
            transport=transport,
            base_url="http://local-tailscaled.sock",
        )
        logger.info("TailscaleNetwork initialised | socket={}", socketPath)

    def _localapi(self, path: str) -> dict:
        """Make a GET request to the Tailscale LocalAPI and return parsed JSON."""
        try:
            response = self._client.get(path)
            response.raise_for_status()
            return response.json()
        except Exception:
            logger.exception("Tailscale LocalAPI request failed | path={}", path)
            raise

    def getMyAddress(self) -> str:
        """Return this device's primary Tailscale IP address."""
        status = self._localapi("/localapi/v0/status")
        address = status["Self"]["TailscaleIPs"][0]
        logger.debug("My Tailscale address | ip={}", address)
        return address

    def getPeerAddress(self, peerId: str) -> str:
        """Return the Tailscale IP of a specific peer by its device ID."""
        status = self._localapi("/localapi/v0/status")
        address = status["Peer"][peerId]["TailscaleIPs"][0]
        logger.debug("Peer address resolved | peerId={} ip={}", peerId, address)
        return address

    def discoverPeers(self) -> list[PeerInfo]:
        """Return all peers currently visible in this device's Tailscale network."""
        status = self._localapi("/localapi/v0/status")
        peers = [
            PeerInfo(
                id=peerId,
                address=info["TailscaleIPs"][0],
                online=info["Online"],
                name=info.get("HostName", peerId),
            )
            for peerId, info in status["Peer"].items()
        ]
        onlineCount = sum(1 for p in peers if p.online)
        logger.info("Tailscale peers discovered | total={} online={}", len(peers), onlineCount)
        return peers

    def isPeerReachable(self, peerId: str) -> bool:
        """Return whether a peer is currently marked Online in the Tailscale network."""
        status = self._localapi("/localapi/v0/status")
        reachable = status["Peer"].get(peerId, {}).get("Online", False)
        logger.debug("Peer reachability check | peerId={} reachable={}", peerId, reachable)
        return reachable
