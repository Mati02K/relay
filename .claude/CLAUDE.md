# CLAUDE.md

This file gives Claude Code the project context needed to work safely in this
repository.

Also read:

- [style.md](style.md) for Python style, typing, docstrings, and verification
- [rules.md](rules.md) for engineering behavior before changing code
- [../README.md](../README.md) for user-facing setup and runtime docs
- [../Scheduling_in_D_Edge_Serving.pdf](../Scheduling_in_D_Edge_Serving.pdf) for the scheduling design and cost formula

## Project Summary

Relay is an experimental distributed LLM serving system for consumer/home
machines. Clients call an OpenAI-compatible coordinator endpoint. The
coordinator reads worker metadata and telemetry, chooses one worker for the
request, forwards the request, and streams the response back to the client.

Relay currently uses data-parallel serving:

- Each worker owns a full model/runtime.
- The coordinator schedules whole requests to workers.
- Relay does not split one model across machines.
- Relay does not transfer KV cache between workers.
- Prefix-cache metadata is used as a scheduling hint.

## Runtime Shape

Default local runtime for `relay init --role dual && relay start`:

```text
etcd
membership-etcd
coordinator
worker
llama-server, started lazily by the worker on first inference request
```

Main endpoint:

```text
POST http://127.0.0.1:8080/v1/chat/completions
```

OpenAI-compatible clients should use:

```text
Base URL: http://127.0.0.1:8080/v1
Model: qwen2.5-0.5b or another configured model id
API key: any value for now
```

## Important Code Paths

CLI/runtime:

- `pyproject.toml` exposes `relay = "relay.cli:main"`
- `src/relay/cli.py` parses CLI commands
- `src/relay/init.py` creates config and prepares runtime software
- `src/relay/config.py` defines persistent config schemas
- `src/relay/software.py` installs/resolves etcd, Go, membership-etcd, and llama-server
- `src/relay/supervisor.py` starts/stops/statuses managed processes

Membership:

- `src/membership/etcd.py` is the Python gRPC client
- `src/membership/etcd-go/main.go` is the Go gRPC service backed by etcd
- Worker metadata and telemetry live under `/relay/workers/...`
- Active coordinator state lives under `/relay/active-coordinator`

Coordinator:

- `src/coordinator/main.py` exposes FastAPI routes and streams requests
- `src/coordinator/worker_registry.py` reads worker metadata/telemetry
- `src/coordinator/scheduler.py` chooses the lowest-cost worker

Worker:

- `src/worker/main.py` creates a singleton `WorkerDaemon`
- `src/worker/daemon.py` owns membership, telemetry, inference, and request execution
- `src/worker/inference/base.py` defines the inference interface
- `src/worker/inference/llamacpp.py` runs llama.cpp through `llama-server`

Telemetry:

- `src/telemetry/schemas.py` defines worker telemetry schemas
- `src/telemetry/state.py` owns mutable worker telemetry
- `src/telemetry/request_metrics.py` derives speeds from completed requests
- `src/telemetry/prefix_cache.py` computes/publishes prefix hashes
- `src/telemetry/prometheus.py` parses llama.cpp metrics

## Scheduler Contract

Before changing scheduling behavior, read `Scheduling_in_D_Edge_Serving.pdf`.
The scheduler must stay aligned with the paper-style cost formula:

```text
queue_weight * q_w / s_w(b)
+ prefix_miss_weight * (1 - overlap(w, r))
+ memory_weight * m_w
+ jitter_weight * j_w / j_max
+ thermal_weight * theta_w
```

Current implementation:

- `q_w` comes from engine queue/load telemetry.
- `s_w(b)` comes from `sw_by_bucket`, decode speed by prompt bucket.
- `overlap(w, r)` comes from prefix-cache hash comparison.
- `m_w` comes from engine memory/KV pressure telemetry.
- `j_w` and `theta_w` exist in schema but real collectors are not implemented yet.
- `j_max` is computed from eligible workers with an environment fallback.

Do not replace this with an unrelated latency heuristic without explicitly
discussing it first.

## Current Limitations

- Real jitter collector is not implemented.
- Real thermal collector is not implemented.
- Tailscale install/auth is not automated.
- LAN peer discovery is not implemented.
- vLLM backend is not implemented.
- `/v1/models` may be needed for some OpenAI-compatible clients.
- Prefix hashing currently uses approximate canonical text hashing; exact tokenizer block hashing is not wired yet.
- Multi-node etcd clustering is not fully automated; current CLI primarily manages a local single-node etcd runtime.

## Development Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

For a clean local verification without touching `~/.relay`:

```bash
export RELAY_HOME=/tmp/relay-test
relay init --role dual --network lan --node-id test-node --model qwen2.5-0.5b
relay start
relay status
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"Say hello"}],"max_tokens":8}'
relay stop
unset RELAY_HOME
```

## Verification Commands

Run the narrow checks for touched code when possible:

```bash
.venv/bin/ruff check src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference src/membership/etcd.py
.venv/bin/ruff format --check src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference src/membership/etcd.py
.venv/bin/mypy src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference
.venv/bin/pytest src/relay/test/test_init.py src/coordinator/test/test_scheduler.py
```

Go middleware check:

```bash
cd src/membership/etcd-go
go test ./...
```
