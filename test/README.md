# Tests

This repository currently uses the source-tree tests and CLI verification flow
as the primary test path.

## Unit Tests

Run the focused Python tests:

```bash
.venv/bin/pytest src/relay/test/test_init.py
.venv/bin/pytest src/coordinator/test/test_scheduler.py
```

Run both together:

```bash
.venv/bin/pytest src/relay/test/test_init.py src/coordinator/test/test_scheduler.py
```

## Static Checks

```bash
.venv/bin/ruff check src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference src/membership/etcd.py
.venv/bin/ruff format --check src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference src/membership/etcd.py
.venv/bin/mypy src/relay src/coordinator src/telemetry src/worker/daemon.py src/worker/main.py src/worker/inference
```

## Go Middleware Check

```bash
cd src/membership/etcd-go
go test ./...
```

If system Go is unavailable, run `relay init` first; Relay can install a managed
Go toolchain under `~/.relay/bin/`.

## Local Runtime Verification

Use an isolated Relay home so the test does not touch the user's normal
`~/.relay` state:

```bash
export RELAY_HOME=/tmp/relay-test
relay init --role dual --network lan --node-id test-node --model qwen2.5-0.5b
relay start
relay status
```

Expected process shape:

```text
etcd             running
membership-etcd  running
coordinator      running
worker           running
```

Send one OpenAI-compatible request through the coordinator:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 8
  }'
```

The runtime test passes when the stream ends with:

```text
data: [DONE]
```

Clean up:

```bash
relay stop
unset RELAY_HOME
```

## Legacy Docker Notes

Older experiments used Docker Compose to simulate a multi-node etcd/Tailscale
cluster. That path is not the current primary verification flow. Prefer the CLI
runtime test above unless a task explicitly asks for Docker-based cluster work.
