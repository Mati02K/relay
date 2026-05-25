# Relay

Relay is an experimental control plane for OpenAI-compatible LLM serving across
consumer machines. It is designed for a small home, lab, or personal-device
cluster where each worker owns its own inference engine and model files. A
coordinator receives client requests, reads worker telemetry, schedules each
request to one worker, forwards the request, and streams the response back to
the client.

The current implementation focuses on data-parallel serving. Relay does not
split a single model across machines and does not transfer KV cache between
machines. Instead, it schedules whole requests using worker health, queue/load,
memory pressure, prefill/decode speed, and prefix-cache hints.

The intended user workflow is:

```bash
relay init
relay start
```

For a local node, Relay prepares and manages the runtime software it needs:
etcd, the Go-built membership middleware, the coordinator, the worker,
`llama-server`, and model files.

## Current Status

Implemented:

- CLI-based initialization, start, stop, status, logs, doctor, and model pull
- etcd-backed membership as the default runtime path
- automatic etcd binary download when etcd is missing
- automatic managed Go toolchain download when system Go is missing or too old
- automatic build of the `membership-etcd` Go middleware
- Hugging Face GGUF model download from a curated catalog
- automatic `llama-server` download from llama.cpp releases when missing
- coordinator and worker process supervision
- OpenAI-compatible streaming chat completions
- worker telemetry publication and prefix-cache-aware scheduling hints

Not implemented yet:

- automatic Tailscale installation and authentication
- LAN peer discovery
- secure invite or pairing flow
- SWIM membership
- vLLM backend
- dashboard

## Prerequisites

For development from this repository:

- Python 3.12+
- `uv`
- network access for downloading models and runtime binaries

You do not need to manually install Go, etcd, or llama.cpp for the default CLI
runtime. Relay downloads or builds those under `~/.relay/` when needed.

Tailscale is not required for a single-machine test. For multi-machine use
across networks, install and authenticate Tailscale manually before using a
Tailscale IP as the coordinator address.

## Install uv

Relay uses `uv` for development environment and dependency management.

On macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `curl` is not available:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the shell or reload your profile if `uv` is not immediately available,
then verify:

```bash
uv --version
```

Official uv installation documentation:

```text
https://docs.astral.sh/uv/getting-started/installation/
```

## Install Relay From Source

Create a virtual environment and install Relay in editable mode:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Verify the CLI:

```bash
relay --help
```

Without activating the virtual environment, use:

```bash
uv run relay --help
```

## Quick Start: Single Machine

Initialize one machine as both coordinator and worker:

```bash
relay init --role all
```

Recommended choices for the first local test:

```text
Network backend: lan or tailscale
Node id: any stable name, for example mathesh-laptop
Model setup: pull
Model: qwen2.5-0.5b
```

During initialization Relay prepares runtime software:

```text
~/.relay/bin/                 managed runtime binaries
~/.relay/cache/               downloaded archives and build cache
~/.relay/models/              GGUF model files
~/.relay/config.json          local node configuration
```

Start Relay:

```bash
relay start
```

Expected process shape for `--role all`:

```text
etcd: log=... pid=...
membership-etcd: log=... pid=...
coordinator: log=... pid=...
worker: log=... pid=...
```

The worker starts `llama-server` lazily on the first inference request. If
`llama-server` is missing, Relay downloads a prebuilt llama.cpp release under
`~/.relay/bin/`.

Check status:

```bash
relay status
```

Health endpoints:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:9090/health
```

Send a streaming chat completion through the coordinator:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one short sentence."
      }
    ],
    "max_tokens": 8
  }'
```

The response is OpenAI-style Server-Sent Events:

```text
data: {"choices":[{"delta":{"content":"Hello"}}], ...}
data: {"choices":[{"delta":{"content":"!"}}], ...}
data: [DONE]
```

Stop Relay:

```bash
relay stop
```

## Verification Test

Use an isolated Relay home when testing changes so the run does not touch an
existing `~/.relay` config or running cluster state:

```bash
export RELAY_HOME=/tmp/relay-test
relay stop
relay init --role all --network lan --node-id test-node --model qwen2.5-0.5b --force
relay start
```

Confirm the managed etcd path is running:

```bash
relay status
```

Expected process list:

```text
etcd             running
membership-etcd  running
coordinator      running
worker           running
```

Check health:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:9090/health
```

Run one inference request through the coordinator:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one short sentence."
      }
    ],
    "max_tokens": 8
  }'
```

The test passes when the response streams `data:` chunks and ends with:

```text
data: [DONE]
```

Clean up:

```bash
relay stop
unset RELAY_HOME
```

## FastAPI Docs

Coordinator API docs:

```text
http://127.0.0.1:8080/docs
```

Worker API docs:

```text
http://127.0.0.1:9090/docs
```

For normal testing, use the coordinator endpoint on port `8080`. That path
exercises scheduling before forwarding to a worker.

Swagger is not ideal for streaming responses, so `curl -N` is usually clearer
for `/v1/chat/completions`.

## CLI Reference

Initialize local config:

```bash
relay init
```

Common role choices:

```bash
relay init --role all
relay init --role coordinator
relay init --role worker --coordinator http://COORDINATOR_IP:8080
```

Start configured processes:

```bash
relay start
```

Stop configured processes:

```bash
relay stop
```

Show process and HTTP health:

```bash
relay status
```

Show logs:

```bash
relay logs etcd
relay logs membership-etcd
relay logs coordinator
relay logs worker
```

Check local runtime configuration and dependencies:

```bash
relay doctor
```

List downloadable models:

```bash
relay models list --catalog
```

List configured local models:

```bash
relay models list
```

Pull a catalog model:

```bash
relay pull qwen2.5-0.5b
```

Register an existing local GGUF:

```bash
relay pull --local /path/to/model.gguf --id my-model
```

## Runtime Files

Relay stores local runtime state under:

```text
~/.relay/
```

Important paths:

```text
~/.relay/config.json
~/.relay/models/
~/.relay/bin/
~/.relay/cache/
~/.relay/logs/
~/.relay/run/
~/.relay/etcd/
```

`config.json` contains node role, network backend, ports, selected model, and
engine settings.

Logs are written to:

```text
~/.relay/logs/etcd.log
~/.relay/logs/membership-etcd.log
~/.relay/logs/coordinator.log
~/.relay/logs/worker.log
~/.relay/logs/relay.log
```

## Managed Software

Relay resolves software in this order:

1. Explicit path from `~/.relay/config.json`
2. System `PATH`
3. Managed installation under `~/.relay/bin/`

For coordinator nodes, Relay needs:

```text
etcd
membership-etcd
Go 1.24 or newer to build membership-etcd
```

If etcd is missing, Relay downloads the configured etcd release into
`~/.relay/bin/`.

If Go is missing or older than the required version, Relay downloads a managed
Go toolchain into `~/.relay/bin/`.

If `membership-etcd` is missing, Relay builds it from:

```text
src/membership/etcd-go/
```

and writes the binary to:

```text
~/.relay/bin/membership-etcd
```

For worker nodes, Relay needs:

```text
llama-server
GGUF model file
```

If `llama-server` is missing, Relay downloads a compatible llama.cpp release
into `~/.relay/bin/`.

## Architecture

Single-machine runtime:

```text
Client
  |
  v
Coordinator :8080
  |
  | schedules request using membership and telemetry
  v
Worker :9090
  |
  | starts or reuses llama-server
  v
llama-server :9081

etcd :2379
  ^
  |
membership-etcd :50051
```

The coordinator and worker talk to the membership interface. The membership
middleware stores node metadata, telemetry, scheduler state, and leader election
state in etcd.

### Coordinator

The coordinator exposes:

```text
POST /v1/chat/completions
```

On each request it:

1. Reads the requested model from the request body.
2. Fetches registered worker metadata and telemetry.
3. Filters workers that advertise the requested model.
4. Computes request prefix hashes for prefix-cache-aware scheduling.
5. Scores eligible workers using the scheduler.
6. Forwards the request to the selected worker.
7. Streams the worker response back to the client.

### Worker

The worker owns:

- one inference engine
- one mutable telemetry state
- local model inventory
- request metrics
- recent prefix-cache hash publication

The worker registers itself through membership, publishes telemetry, receives
scheduled requests from the coordinator, calls the inference engine, and streams
results back.

### Inference Engine

The current inference backend is `llama.cpp` through `llama-server`.

The worker starts `llama-server` on the first generation request. The engine
then talks to:

```text
/v1/chat/completions
/health
/metrics
```

The `/metrics` endpoint is parsed for engine-reported telemetry such as queue
or active request count and KV-cache pressure when available.

### Telemetry

Relay separates telemetry by source:

```text
Engine-reported telemetry:
  qw: queue/load signal
  mw: memory or KV-cache pressure

Request-computed telemetry:
  sw_by_bucket: decode speed by prompt-length bucket
  sprefill_tokens_per_sec: prefill speed estimate
  prefix_cache: recent prefix block hashes

System telemetry:
  jw: network jitter
  theta_w: thermal throttling flag
```

The jitter and thermal signals are currently neutral until dedicated collectors
are added.

## Models

The current catalog focuses on small to 7B-class GGUF models:

```text
qwen2.5-0.5b
qwen2.5-1.5b
qwen2.5-3b
llama-3.2-1b
llama-3.2-3b
phi-3.5-mini
mistral-7b
```

View the catalog:

```bash
relay models list --catalog
```

Pull the smallest smoke-test model:

```bash
relay pull qwen2.5-0.5b
```

## Multi-Machine Use

Current multi-machine flow uses a coordinator address directly. There is no
invite code yet.

Machine A:

```bash
relay init --role all
relay start
```

Machine B:

```bash
relay init --role worker --coordinator http://COORDINATOR_IP:8080
relay start
```

Use a normal LAN IP if both machines can reach each other directly. Use a
Tailscale IP if the machines are on different networks or behind NAT.

LAN discovery is not implemented yet. Selecting `lan` records the backend in
config, but automatic peer discovery is future work.

## Tailscale Setup

Tailscale is only needed when coordinator and worker machines must reach each
other across different networks, NATs, or Wi-Fi environments where normal local
IP addresses are not reliable.

You do not need Tailscale for:

- a single-machine local test
- coordinator and worker running on the same host
- basic CLI development

Install Tailscale on Linux:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

Start and authenticate Tailscale:

```bash
sudo tailscale up
```

Verify the device has a Tailscale IP:

```bash
tailscale ip -4
```

Check peer visibility:

```bash
tailscale status
```

Official Tailscale Linux installation documentation:

```text
https://tailscale.com/docs/install/linux
```

Use the coordinator machine's Tailscale IP when initializing a worker:

```bash
relay init --role worker --coordinator http://100.x.y.z:8080
```

Relay does not currently install Tailscale, run `tailscale up`, or manage
Tailscale authentication.

## Development Commands

Run checks:

```bash
.venv/bin/python -m compileall src
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/mypy src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference
```

Run CLI tests:

```bash
.venv/bin/pytest src/relay/test/test_init.py
```

## Troubleshooting

### `relay` command not found

Activate the virtual environment:

```bash
source .venv/bin/activate
```

or run through uv:

```bash
uv run relay status
```

### Port already in use

Default ports:

```text
2379  etcd client
2380  etcd peer
50051 membership-etcd
8080  coordinator
9090  worker
9081  llama-server
```

Stop Relay:

```bash
relay stop
```

Then inspect logs:

```bash
relay logs etcd
relay logs membership-etcd
relay logs coordinator
relay logs worker
```

### Runtime software installation fails

Check:

```bash
relay doctor
```

The first init or start needs network access to download etcd, Go if needed,
llama.cpp, Go modules, and any selected model. Managed files are stored under
`~/.relay/bin/` and `~/.relay/cache/`.

### Model mismatch

The request model must match a model advertised by a worker. Check:

```bash
relay models list
```

Then use that exact model id in the request body:

```json
{
  "model": "qwen2.5-0.5b"
}
```

### First request is slower

The worker starts `llama-server` lazily on the first generation request. Later
requests reuse the same server process.

### Streaming output looks verbose

The endpoint returns OpenAI-compatible streaming SSE lines. Join the
`delta.content` fields to reconstruct the final assistant message.
