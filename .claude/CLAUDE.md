# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Also read:**
- [style.md](style.md) — Python naming, types, async, docstrings, formatting rules
- [rules.md](rules.md) — What you must do before every change

---

## What This Project Is

Relay is a distributed LLM inference system for consumer-grade edge devices. The idea: instead of needing one expensive GPU to run a large model, a group of people can pool their home devices (laptops, desktops, phones) and serve a model collaboratively. Each device holds a full copy of the model and handles requests independently — no cross-device communication during inference. A central coordinator receives all requests and decides which device handles each one.

The full design is in `Scheduling_in_D_Edge_Serving.pdf`.

## Tech Stack

| Layer | Technology |
|---|---|
| Coordinator + workers | Python (FastAPI, grpc.aio, Pydantic) |
| Cluster membership / KV store | etcd (3-node Raft cluster) |
| etcd gRPC middleware | Go (official etcd client v3, wrapped as a gRPC server) |
| Cross-network connectivity | Tailscale (WireGuard overlay, stable 100.x.x.x IPs) |
| Local network fallback | mDNS/DNS-SD via zeroconf |
| Inference engines | llama.cpp (primary), vLLM (secondary) |
| gRPC contract | `proto/relay.proto` — shared between Go and Python |

## Development Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
make proto                              # generate gRPC stubs (run first, always)
cd src/membership/etcd-go && go mod download && go build ./...
```

## Docker Test Workflow

```bash
cp .env.example .env                   # set TAILSCALE_AUTHKEY (ephemeral key from tailscale.com)
docker compose -f test/docker-compose.yml up --build
docker compose -f test/docker-compose.yml -f test/docker-compose.test.yml run --rm test-runner
docker compose -f test/docker-compose.yml down
```
