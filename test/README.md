# Integration Tests

Docker-based test environment that simulates a two-node Relay cluster on a single machine. Runs the full stack: 3-node etcd cluster, Go gRPC middleware, two coordinators, two workers, and a test runner that verifies end-to-end message storage and retrieval.

## Prerequisites

- Docker with Compose
- A Tailscale ephemeral auth key ([generate one here](https://login.tailscale.com/admin/settings/keys) — check "Ephemeral")

## Setup

```bash
cp .env.example .env
# Edit .env and set TAILSCALE_AUTHKEY=tskey-auth-...
```

Ephemeral keys auto-remove nodes from your tailnet when containers exit, so test runs don't accumulate phantom devices.

## Run

```bash
# From the repo root
docker compose -f test/docker-compose.yml up --build -d
docker compose -f test/docker-compose.yml -f test/docker-compose.test.yml run --rm test-runner
docker compose -f test/docker-compose.yml down
```

## What the tests check

1. Worker becomes healthy and discovers an elected coordinator
2. POST `/v1/chat` via worker-1 stores a message (forwarded to active coordinator → etcd)
3. GET `/v1/messages` via worker-2 retrieves the same message (different worker, same coordinator)

## Testing failover manually

```bash
# While the stack is running, stop the active coordinator
docker compose -f test/docker-compose.yml stop coordinator-2

# Wait ~10 seconds, then check the worker — it should show coordinator-1 as the new leader
curl http://localhost:9091/health

# Re-run the tests — should still pass against the new coordinator
docker compose -f test/docker-compose.yml -f test/docker-compose.test.yml run --rm test-runner
```

## Container map

| Container | Role |
|-----------|------|
| `test-etcd-{1,2,3}-1` | 3-node etcd Raft cluster |
| `test-membership-etcd-1` | Go gRPC middleware (wraps etcd) |
| `test-coordinator-{1,2}-1` | Coordinator on simulated Node 1 and Node 2 |
| `test-worker-{1,2}-1` | Worker on simulated Node 1 and Node 2 |

In bare metal each node runs all four components locally. See the root [README.md](../README.md) for bare metal setup.
