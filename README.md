# Relay

**Control Plane for Distributed Edge Inference.**

Relay is an experimental, OpenAI-compatible control plane that runs across a
small home, lab, or personal-device cluster. Each worker owns its own inference
engine and model files. A coordinator receives client requests, reads worker
telemetry, schedules each request to the best available worker, and streams
the response back to the client.

Relay focuses on data-parallel serving — it does not split a single model
across machines. Scheduling weighs queue depth, memory pressure, prefix-cache
overlap, network jitter, and thermal state.

## Architecture

```
Client
  │
  ▼
Coordinator :8080          (schedules + proxies)
  │
  │  picks worker by cost function
  ▼
Worker :9090               (owns inference engine)
  │
  ▼
llama-server :9081         (llama.cpp, actual inference)

etcd :2379  ←  membership-etcd :50051   (service discovery + telemetry store)
```

The coordinator and every worker register through `membership-etcd`, a
lightweight Go gRPC middleware backed by etcd. Workers publish telemetry every
200 ms; the coordinator reads it on every request to pick the lowest-cost
worker. If a worker goes down its etcd lease expires and it drops out of
scheduling automatically.

## Prerequisites

- Python 3.12+
- `uv`
- Network access for first-time downloads (models, runtime binaries)

You do not need to install Go, etcd, or llama.cpp manually. Relay downloads or
builds those under `~/.relay/` on first use.

## Install uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```bash
uv --version
```

## Install Relay From Source

```bash
git clone <repo-url>
cd Relay
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
relay --help
```

Without activating the virtual environment:

```bash
uv run relay --help
```

---

## Quick Start

### Single machine — dual mode

Run the coordinator and a worker on the same machine. This is the fastest way
to get Relay running and is enough for local testing or a single powerful node.

```bash
relay init --role dual
```

Answer the prompts:

```
Network backend : lan
Node id         : my-machine
Model           : qwen2.5-0.5b   (or pick any from the catalog)
```

Start everything:

```bash
relay start
```

Relay will download etcd, build the membership middleware, and pull the model
on first run. When ready:

```bash
relay status
```

Expected output:

```
etcd             running  pid=...
membership-etcd  running  pid=...
coordinator      running  pid=...
worker           running  pid=...
```

Test with a streaming request:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 32
  }'
```

The first request is slower — the worker starts `llama-server` lazily on the
first inference call. Subsequent requests reuse the same process.

Stop:

```bash
relay stop
```

---

### Adding a worker node

Once you have a coordinator running (either as `--role dual` or
`--role coordinator`), you can join additional machines as workers.

On the worker machine:

```bash
relay init --role worker \
    --coordinator http://COORD_IP:8080 \
    --model qwen2.5-3b
relay start
relay status
```

The worker registers in the coordinator's etcd, starts publishing telemetry,
and immediately becomes eligible for scheduling. You can give each worker a
different model — the coordinator routes requests based on the `model` field in
the request.

#### Launch the dashboard from this worker (optional)

The dashboard is **not tied to the coordinator host** — any machine that can
reach the coordinator over the network can launch it, including this worker.
Useful when you want to inspect the cluster, run a test chat, or tune
scheduler weights without ssh-ing back to the coordinator machine.

On the worker:

```bash
relay dashboard --coordinator http://COORD_IP:8080
```

The browser opens at `http://127.0.0.1:8090` on this machine. From there you
can see all workers on the live map (including this one), open the test chat,
or change scheduler weights — every action goes back to the coordinator and is
reflected cluster-wide.

If `8090` is already in use locally, pass `--port`:

```bash
relay dashboard --port 8095 --coordinator http://COORD_IP:8080
```

The dashboard process is supervised — `relay stop` on this worker will stop it
alongside the worker, and `relay status` will list it. See [Dashboard](#dashboard)
for the full feature walkthrough.

---

## Models

The catalog covers small to 7B-class GGUF models:

| ID | Size |
|---|---|
| `qwen2.5-0.5b` | 0.5 B |
| `qwen2.5-1.5b` | 1.5 B |
| `qwen2.5-3b` | 3 B |
| `llama-3.2-1b` | 1 B |
| `llama-3.2-3b` | 3 B |
| `phi-3.5-mini` | 3.8 B |
| `mistral-7b` | 7 B |

List the catalog:

```bash
relay models list --catalog
```

Pull a model:

```bash
relay pull qwen2.5-0.5b
```

Register an existing local GGUF without downloading:

```bash
relay pull --local /path/to/model.gguf --id my-model
```

List models configured on this node:

```bash
relay models list
```

---

## Multi-Machine Setup

### Using LAN (recommended for same-network setups)

All machines on the same local network — no overlay, no extra daemons.
Relay auto-detects your LAN IP so you do not need to pass `--host`.

#### Step 1 — Open firewall ports

On the **coordinator machine**:

```bash
sudo ufw allow 8080/tcp    # coordinator API
sudo ufw allow 50051/tcp   # membership-etcd gRPC (workers register here)
```

On each **worker machine**:

```bash
sudo ufw allow 9090/tcp    # worker API (coordinator proxies requests here)
```

#### Step 2 — Start the coordinator

```bash
relay init --role dual --network lan --node-id home-server --model qwen2.5-3b
relay start
relay status
```

`relay status` will show the detected LAN IP in the coordinator address.
Note it — you need it for the worker step.

Confirm the coordinator is reachable from any machine on the network:

```bash
curl http://COORD_IP:8080/health
# {"status":"ok","nodeId":"home-server","isLeader":true}
```

#### Step 3 — Join each worker

On every worker machine:

```bash
relay init --role worker \
    --network lan \
    --coordinator http://COORD_IP:8080 \
    --node-id my-worker \
    --model qwen2.5-0.5b

relay start
relay status
```

Each worker auto-detects its own LAN IP and registers it with the coordinator.
You can use a different model on each worker — the coordinator routes by the
`model` field in the request.

#### Step 4 — Verify the cluster

From any machine:

```bash
curl http://COORD_IP:8080/v1/workers | python3 -m json.tool
```

Every registered worker appears with `"healthy": true`. Then send a request:

```bash
curl -sN http://COORD_IP:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-3b",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 32
  }'
```

Check `X-Relay-Worker` in the response headers to see which worker handled it:

```bash
curl -sI -XPOST http://COORD_IP:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-3b","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | grep X-Relay
```

#### Troubleshooting

**Worker registers but requests never reach it:**
The coordinator cannot connect to the worker's port. Check `sudo ufw status`
on the worker machine — port `9090` must be open.

**Worker fails to start with a gRPC error:**
The worker cannot reach `COORD_IP:50051`. Check the coordinator machine's
firewall — port `50051` must be open.

**Wrong IP advertised (worker registers as `127.0.0.1`):**
This can happen on machines with no default route (offline or VPN-only).
Pass `--host YOUR_LAN_IP` explicitly to `relay init` to override auto-detection.

---

### Using Tailscale (for cross-network / multi-site clusters)

If your machines live on different networks or behind NATs, Tailscale gives
every machine a stable `100.x.y.z` address that works across networks without
port-forwarding or VPN configuration.

**Install Tailscale on every machine:**

Linux:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

macOS:

```bash
brew install --cask tailscale
open -a Tailscale   # sign in via the menu bar icon
```

Windows: download from <https://tailscale.com/download/windows>.

Confirm all machines are visible:

```bash
tailscale status
```

Note the `100.x.y.z` address of the coordinator machine — used as `COORD_IP`
below.

**Coordinator node:**

```bash
relay init --role dual --network tailscale --model qwen2.5-3b
relay start
relay status
```

`--role dual` means this machine runs the coordinator, etcd, the membership
middleware, and its own worker. Pass `--skip-model` if you only want it to
coordinate.

Confirm coordinator health:

```bash
curl -s http://127.0.0.1:8080/health
# {"status":"ok","nodeId":"...","isLeader":true}
```

**Each worker node:**

```bash
relay init --role worker \
    --network tailscale \
    --coordinator http://COORD_IP:8080 \
    --model qwen2.5-3b
relay start
relay status
```

**Verify the cluster:**

```bash
curl -s http://COORD_IP:8080/v1/workers | python3 -m json.tool
```

You should see one entry per registered worker with live telemetry.

**Open the dashboard from any node on the tailnet:**

```bash
relay dashboard --coordinator http://COORD_IP:8080
```

Works from the coordinator, any worker, or any other machine on the tailnet —
it's just an HTTP front-end that proxies to the coordinator. See the
[Dashboard](#dashboard) section for what each view does.

**Test a chat request:**

```bash
curl -N http://COORD_IP:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "qwen2.5-3b",
      "messages": [{"role": "user", "content": "Say hello in one sentence."}],
      "max_tokens": 32
    }'
```

Check the response headers to see which worker served the request:

```
X-Relay-Worker: worker-node-b
X-Relay-Cost: 0.3214
X-Relay-Matched-Tokens: 0
X-Relay-Attempts: 1
```

**Connect Open WebUI (optional):**

In Open WebUI → Settings → Connections → OpenAI API → Add Connection:

- **Base URL:** `http://COORD_IP:8080/v1`
- **API key:** anything (e.g. `relay`) — not validated
- The model dropdown auto-populates from `/v1/models`

---

### Worker failure handling

- When a worker crashes its etcd lease expires within ~10 seconds. After that
  it disappears from `/v1/workers` and scheduling skips it.
- If a worker dies mid-request, the coordinator's retry logic catches the
  connect failure and picks a different worker. The response carries
  `X-Relay-Attempts: 2` to indicate a retry occurred.
- If every worker is unreachable the coordinator returns `503 "All eligible
  workers unreachable"` instead of hanging.

To kill a worker and watch it recover:

```bash
kill -9 $(cat ~/.relay/run/worker.pid)
# worker disappears from the map within ~10s
relay start   # bring it back; re-registers immediately
```

---

## CLI Reference

```bash
relay init                        # interactive setup wizard
relay init --role dual            # coordinator + worker on this machine
relay init --role coordinator     # coordinator only
relay init --role worker \
    --coordinator http://IP:8080  # worker only, join existing coordinator

relay start                       # start configured processes
relay stop                        # stop all processes
relay restart                     # stop then start

relay status                      # process state + HTTP health
relay logs worker                 # tail logs (etcd / membership-etcd / coordinator / worker)
relay doctor                      # check runtime dependencies

relay models list                 # local models
relay models list --catalog       # downloadable catalog
relay pull qwen2.5-0.5b           # download a catalog model
relay pull --local /path.gguf \
    --id my-model                 # register an existing file

relay dashboard                   # open the web dashboard (default :8090)
relay dashboard \
    --coordinator http://COORD_IP:8080  # point at a remote coordinator
```

---

## Dashboard

Launch the dashboard from any machine that can reach the coordinator:

```bash
relay dashboard
# opens http://127.0.0.1:8090 in your browser
```

Point at a remote coordinator:

```bash
relay dashboard --coordinator http://COORD_IP:8080
```

**Map (home)** — click the **Relay** logo at any time to return here.

The map shows the live topology: coordinator at the centre, workers around the
rim. Animated pulses on the edges show active traffic. The header bar shows
total workers, healthy count, average thermal pressure, and average network
jitter across the cluster.

Click any worker node to open a details drawer with health state, telemetry
readings (queue depth, memory pressure, jitter, thermal), and loaded models.

**Test chat** — click the **test** button (top right).

A full chat interface that routes requests through the coordinator exactly like
any other client. After each response the right panel shows the scheduler
decision: which worker was picked, the cost score, prefix-cache overlap, and
timing.

**Settings** — click the **⚙ settings** button (top right).

Live scheduler weight tuning. The five sliders control how much each signal
influences worker scoring:

| Weight | What it controls |
|---|---|
| queue | Avoids workers with a long request backlog |
| prefix_miss | Prefers workers that already have your prompt cached |
| memory | Avoids workers running low on GPU/CPU memory |
| jitter | Avoids workers on unstable network paths |
| thermal | Avoids workers that are overheating or throttling |

Changes take effect on the **very next request** — no restart needed.

---

## Runtime Files

Relay stores everything under `~/.relay/`:

```
~/.relay/config.json      node role, ports, model selection, engine settings
~/.relay/models/          downloaded GGUF model files
~/.relay/bin/             managed binaries (etcd, llama-server, Go toolchain)
~/.relay/cache/           download archives and build cache
~/.relay/logs/            per-process log files
~/.relay/run/             PID files for managed processes
~/.relay/etcd/            etcd data directory
```

Logs:

```
~/.relay/logs/etcd.log
~/.relay/logs/membership-etcd.log
~/.relay/logs/coordinator.log
~/.relay/logs/worker.log
```

---

## Development Commands

```bash
.venv/bin/python -m compileall src
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/mypy src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference
.venv/bin/pytest src/relay/test/test_init.py
```

---

## Troubleshooting

### `relay` command not found

```bash
source .venv/bin/activate
# or
uv run relay status
```

### Port already in use

Default ports:

```
2379   etcd client
2380   etcd peer
50051  membership-etcd
8080   coordinator
9090   worker
9081   llama-server
8090   dashboard
```

```bash
relay stop
relay logs worker   # inspect what went wrong
```

### Runtime software download fails

```bash
relay doctor
```

First `init` or `start` needs network access to download etcd, Go (if
missing), llama.cpp, and the selected model. Files are cached under
`~/.relay/bin/` and `~/.relay/cache/`.

### Worker shows unhealthy despite llama-server running

A stale `llama-server` from a previous session may be occupying the inference
port (`9081` by default). `relay stop` now sends signals to the full process
group so this should not happen after a clean stop. If it does:

```bash
relay stop
# find and kill the orphaned process
ss -tlnp | grep 9081
kill <pid>
relay start
```

### Model not found / no workers for model

The `model` field in the request must exactly match a model ID advertised by a
worker:

```bash
relay models list          # see what this node has
curl http://COORD_IP:8080/v1/models   # see what the cluster has
```

### First request is slow

Expected — `llama-server` starts lazily on the first inference call. Subsequent
requests on the same worker reuse the running process.
