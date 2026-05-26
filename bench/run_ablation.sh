#!/usr/bin/env bash
# Run all ablation configurations end-to-end and accumulate results into one CSV.
#
# Assumes the cluster is already up: etcd, the Go membership service,
# the coordinator, and three workers (worker-a/b/c) as described in
# RELAY_GUIDE.md. The coordinator is restarted between runs to pick up the
# new RELAY_* weight env vars; workers are NOT restarted, so their KV cache
# state and observed telemetry roll forward across configs (mirroring how a
# real deployment would behave when re-weighting the cost function live).
#
# Usage:
#   bench/run_ablation.sh [results_csv] [prompts.jsonl] [concurrency]
#
# Output: appends to the CSV with a run_label column per config.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_CSV="${1:-bench/results/ablation.csv}"
PROMPTS="${2:-bench/data/prompts.jsonl}"
CONCURRENCY="${3:-1}"
MAX_TOKENS="${MAX_TOKENS:-48}"
WARMUP="${WARMUP:-2}"

mkdir -p "$(dirname "$RESULTS_CSV")"

# Reset coordinator-side state between runs so each config starts cleanly.
# We avoid restarting workers because that would lose llama-server warmup
# and the prefix-hash cache, both of which are part of the system under test.
COORD_BIN_ARGS="--app-dir src --host 0.0.0.0 --port 8080 --log-level warning"
COORD_PIDFILE=/tmp/relay-coord.pid

restart_coord() {
  if [[ -f "$COORD_PIDFILE" ]]; then
    kill "$(cat "$COORD_PIDFILE")" 2>/dev/null || true
    sleep 1
  fi
  # Kill any other coordinator that might still be holding port 8080.
  pkill -f 'uvicorn coordinator.main:app' 2>/dev/null || true
  sleep 2
  : > /tmp/relay-coord.log
  (
    cd "$REPO_ROOT"
    source .venv/bin/activate
    NODE_ID=coord-a COORDINATOR_HOST=127.0.0.1 \
      RELAY_ALPHA="${RELAY_ALPHA:-1.0}" \
      RELAY_BETA="${RELAY_BETA:-1.0}" \
      RELAY_GAMMA="${RELAY_GAMMA:-1.0}" \
      RELAY_DELTA="${RELAY_DELTA:-1.0}" \
      RELAY_EPSILON="${RELAY_EPSILON:-1.0}" \
      RELAY_PHI="${RELAY_PHI:-0.5}" \
      RELAY_TTFT_SLO_MS="${RELAY_TTFT_SLO_MS:-0}" \
      .venv/bin/uvicorn coordinator.main:app $COORD_BIN_ARGS \
      >> /tmp/relay-coord.log 2>&1 &
    echo $! > "$COORD_PIDFILE"
  )
  # Wait for coordinator to come back up.
  for _ in $(seq 1 30); do
    if curl -sf -m 1 http://127.0.0.1:8080/v1/cluster -o /dev/null; then
      return 0
    fi
    sleep 0.5
  done
  echo "coordinator failed to start" >&2
  tail -n 30 /tmp/relay-coord.log >&2
  return 1
}

run_config() {
  local label="$1"
  shift
  echo
  echo "=========================================="
  echo "Config: $label"
  echo "=========================================="
  env "$@" bash -c '
    : "${RELAY_ALPHA:=1.0}"
    : "${RELAY_BETA:=1.0}"
    : "${RELAY_GAMMA:=1.0}"
    : "${RELAY_DELTA:=1.0}"
    : "${RELAY_EPSILON:=1.0}"
    : "${RELAY_PHI:=0.5}"
    : "${RELAY_TTFT_SLO_MS:=0}"
    echo "  weights: alpha=$RELAY_ALPHA beta=$RELAY_BETA gamma=$RELAY_GAMMA \
delta=$RELAY_DELTA epsilon=$RELAY_EPSILON phi=$RELAY_PHI \
slo_ms=$RELAY_TTFT_SLO_MS"
  '
  ( export "$@"; restart_coord )
  source .venv/bin/activate
  python bench/replay.py \
    --prompts "$PROMPTS" \
    --out "$RESULTS_CSV" \
    --run-label "$label" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --warmup "$WARMUP"
}

# ---- Configs ----
# Each one has a clear hypothesis attached so reviewers can connect the
# numbers in the CSV back to the paper's claims.

# Baseline: round-robin (all weights zero). Whichever worker wins ties wins
# every request -> shows how brittle uniform routing is on a heterogeneous fleet.
run_config rr \
  RELAY_ALPHA=0 RELAY_BETA=0 RELAY_GAMMA=0 RELAY_DELTA=0 RELAY_EPSILON=0 RELAY_PHI=0

# Full scheduler with all terms on (the proposed approach).
run_config full \
  RELAY_ALPHA=1 RELAY_BETA=1 RELAY_GAMMA=1 RELAY_DELTA=1 RELAY_EPSILON=1 RELAY_PHI=0.5

# Cache off: drops the KV-cache reuse term. If the cluster has any prefix
# locality, this should regress p50 TTFT versus 'full'.
run_config no_cache \
  RELAY_ALPHA=1 RELAY_BETA=0 RELAY_GAMMA=1 RELAY_DELTA=1 RELAY_EPSILON=1 RELAY_PHI=0.5

# Jitter off: ignores network variance. Should over-route to the artificially
# jittery worker, lifting p99 TTFT.
run_config no_jitter \
  RELAY_ALPHA=1 RELAY_BETA=1 RELAY_GAMMA=1 RELAY_DELTA=0 RELAY_EPSILON=1 RELAY_PHI=0.5

# Thermal off: ignores theta_w. Should send more traffic to the 'hot' worker.
run_config no_thermal \
  RELAY_ALPHA=1 RELAY_BETA=1 RELAY_GAMMA=1 RELAY_DELTA=1 RELAY_EPSILON=0 RELAY_PHI=0.5

# SLO on: rejects requests with estimated TTFT > threshold. Demonstrates the
# admission-control behavior from paper section 3.2 (last paragraph).
run_config full_slo \
  RELAY_ALPHA=1 RELAY_BETA=1 RELAY_GAMMA=1 RELAY_DELTA=1 RELAY_EPSILON=1 RELAY_PHI=0.5 \
  RELAY_TTFT_SLO_MS=2000

# RouteLLM-style routing on (nu term). Workers must advertise RELAY_MODEL_QUALITY
# in their env for this to do anything; with all-equal quality the term is inert.
# Set nu = 5.0 so it dominates only on complex prompts (complexity score ~0.4+).
run_config routellm_nu \
  RELAY_ALPHA=1 RELAY_BETA=1 RELAY_GAMMA=1 RELAY_DELTA=1 RELAY_EPSILON=1 RELAY_PHI=0.5 \
  RELAY_NU=5.0

echo
echo "Done. Results: $RESULTS_CSV"
