"""Network layer factory.

The supervisor propagates :attr:`config.network.backend` into the
``RELAY_NETWORK_BACKEND`` env var when starting worker and coordinator
processes; this module reads it and returns the matching implementation.
Today only ``getMyAddress`` is consumed, but the abstraction stays so
future deployment topologies (e.g. cross-network via Tailscale) can plug
in without touching call sites.
"""

from __future__ import annotations

import os

from network.base import NetworkLayer
from network.lan import LANNetwork
from network.tailscale import TailscaleNetwork

__all__ = ["NetworkLayer", "build_network"]


def build_network() -> NetworkLayer:
    """Return the configured :class:`NetworkLayer` for this process.

    Defaults to LAN to match :class:`NetworkConfig` defaults; Tailscale is
    the opt-in choice for cross-network or multi-site deployments.
    """
    backend = os.getenv("RELAY_NETWORK_BACKEND", "lan").strip().lower()
    if backend == "tailscale":
        return TailscaleNetwork()
    return LANNetwork()
