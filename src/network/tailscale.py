from typing import Any

import httpx
from loguru import logger

from .base import NetworkLayer


class TailscaleNetwork(NetworkLayer):
    """NetworkLayer implementation using the Tailscale LocalAPI via Unix domain socket."""

    def __init__(self, socketPath: str = "/var/run/tailscale/tailscaled.sock") -> None:
        """Connect to the local Tailscale daemon socket."""
        transport = httpx.HTTPTransport(uds=socketPath)
        self._client = httpx.Client(
            transport=transport,
            base_url="http://local-tailscaled.sock",
        )
        logger.info("TailscaleNetwork initialised | socket={}", socketPath)

    def _localapi(self, path: str) -> dict[str, Any]:
        """Make a GET request to the Tailscale LocalAPI and return parsed JSON."""
        try:
            response = self._client.get(path)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data
        except Exception:
            logger.exception("Tailscale LocalAPI request failed | path={}", path)
            raise

    def getMyAddress(self) -> str:
        """Return this device's primary Tailscale IP address."""
        status = self._localapi("/localapi/v0/status")
        address = str(status["Self"]["TailscaleIPs"][0])
        logger.debug("My Tailscale address | ip={}", address)
        return address
