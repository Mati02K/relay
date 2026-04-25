# Relay

Distributed LLM inference across consumer-grade home devices. Instead of needing one expensive GPU, a group of people pool their machines into a cluster. Each device holds a full copy of the model and handles requests independently. A coordinator elected via etcd decides who serves each request; workers on every node forward client traffic to it.

## Architecture

Each physical node runs four processes:

```
Node A                    Node B                    Node C
├── etcd-A ──────────────├── etcd-B ──────────────├── etcd-C
│   (Raft cluster across all nodes)
├── go-middleware-A       ├── go-middleware-B       ├── go-middleware-C
│   (wraps etcd via gRPC) │                         │
├── coordinator-A         ├── coordinator-B         ├── coordinator-C
│   (campaigns for leader)│   (one wins, rest idle) │
└── worker-A              └── worker-B              └── worker-C
    (always active, proxies    to the elected coordinator)
```

Clients hit any worker. The worker looks up the active coordinator from etcd and forwards the request. If the active coordinator's machine goes offline, etcd detects it within ~10 seconds, re-runs the election, and workers automatically discover the new leader.

## Bare Metal Setup

### Prerequisites

Every node needs:
- [Go 1.24+](https://go.dev/dl/)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- [etcd v3.5](https://github.com/etcd-io/etcd/releases)
- [Tailscale](https://tailscale.com/download) (for cross-network connectivity)

### Build

On each node, clone the repo and build the Go middleware:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
make proto

cd src/membership/etcd-go
go mod download
go build -o membership-etcd .
```

### Start etcd (on every node)

Replace `NODE_NAME`, `NODE_IP`, and the peer list with your actual node names and Tailscale IPs (`tailscale ip -4`):

```bash
etcd \
  --name NODE_NAME \
  --initial-advertise-peer-urls http://NODE_IP:2380 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://NODE_IP:2379 \
  --initial-cluster node-a=http://100.x.x.1:2380,node-b=http://100.x.x.2:2380,node-c=http://100.x.x.3:2380 \
  --initial-cluster-state new \
  --initial-cluster-token relay-cluster
```

### Start Go middleware (on every node)

```bash
ETCD_ENDPOINTS=localhost:2379 GRPC_PORT=50051 NODE_ID=NODE_NAME ./src/membership/etcd-go/membership-etcd
```

### Start coordinator (on every node)

```bash
MEMBERSHIP_HOST=localhost \
MEMBERSHIP_PORT=50051 \
NODE_ID=NODE_NAME \
COORDINATOR_HOST=NODE_IP \
COORDINATOR_PORT=8080 \
uvicorn coordinator.main:app --host 0.0.0.0 --port 8080
```

Only one coordinator will win the election and serve requests. The others campaign and take over automatically if the leader goes offline.

### Start worker (on every node)

```bash
MEMBERSHIP_HOST=localhost \
MEMBERSHIP_PORT=50051 \
NODE_ID=NODE_NAME \
uvicorn worker.main:app --host 0.0.0.0 --port 9090
```

### Verify

Check which coordinator is active:

```bash
etcdctl get /relay/active-coordinator
```

Hit any worker to store a message:

```bash
curl -X POST http://NODE_IP:9090/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
# {"id": "<uuid>", "stored": true}
```

Retrieve it from a different node's worker:

```bash
curl http://OTHER_NODE_IP:9090/v1/messages
```

Test failover: kill the coordinator process on the active node, wait ~10 seconds, then check `/relay/active-coordinator` again — it will point to a different node.

## Running Tests

See [test/README.md](test/README.md) for the Docker-based integration test setup.
