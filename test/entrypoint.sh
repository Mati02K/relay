#!/bin/sh
set -e

# Start the Tailscale daemon in the background.
tailscaled \
  --state=/var/lib/tailscale/tailscaled.state \
  --socket=/var/run/tailscale/tailscaled.sock &

# Wait for tailscaled to become ready.
sleep 3

# Join the real Tailscale network using the ephemeral auth key from .env.
tailscale up \
  --authkey="${TAILSCALE_AUTHKEY}" \
  --hostname="${HOSTNAME}" \
  --accept-routes

echo "Tailscale up. My address: $(tailscale ip -4)"

# Hand off to the container's CMD.
exec "$@"
