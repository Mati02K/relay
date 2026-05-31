# Relay Test Framework

A rigorous, reproducible test suite that validates every scheduler signal,
proves prefix-cache routing affinity, and benchmarks Relay against round-robin
and single-worker baselines.

## Prerequisites

1. A running Relay cluster (coordinator + ≥ 2 workers).
2. Python ≥ 3.11 with `uv`.
3. Test dependencies installed:

```bash
uv pip install -e ".[test]"
```

## Quick start

```bash
# 1. Prepare dataset (one-time — downloads ~400 MB of ShareGPT data)
python test/run_tests.py --prepare-data

# 2. Run all signal scenario tests
python test/run_tests.py --coordinator http://192.168.1.10:8080 --signals

# 3. Run a specific scenario
python test/run_tests.py --coordinator http://192.168.1.10:8080 --scenarios memory

# 4. Run baseline comparison (Relay vs round-robin vs single-worker)
python test/run_tests.py --coordinator http://192.168.1.10:8080 --compare

# 5. Run load tests (ramp / spike / sustained)
python test/run_tests.py --coordinator http://192.168.1.10:8080 --load --shape ramp
python test/run_tests.py --coordinator http://192.168.1.10:8080 --load --shape spike
python test/run_tests.py --coordinator http://192.168.1.10:8080 --load --shape sustained

# 6. Run everything at once
python test/run_tests.py --coordinator http://192.168.1.10:8080 --all
```

## Warm-up (automatic)

Every pytest run starts with an automatic warm-up (an autouse session fixture in
`conftest.py`). Before any test measures routing, it sends 5 fixed synthetic
prompts **directly to each healthy worker** — bypassing the scheduler so every
worker is covered — to load each engine and settle queue/speed telemetry. The
prompts carry a `[relay-warmup]` sentinel and do not appear in the ShareGPT
working set, so warming never pollutes prefix-cache or routing state.

You don't run this yourself; it happens on every `run_tests.py`/`pytest`
invocation. It is best-effort: if a worker is slow to load it retries, and if no
healthy worker appears it skips and lets the test's own `wait_for_workers`
report the problem. (Locust load runs use their own process and are not warmed
by this fixture.)

## Folder structure

```
test/
├── README.md              ← you are here
├── conftest.py            ← pytest fixtures (coordinator URL, dataset, run_dir)
├── run_tests.py           ← main entry point CLI
│
├── framework/             ← shared test infrastructure
│   ├── baseline.py        ← round-robin-vs-signal A/B harness (shared by all scenarios)
│   ├── client.py          ← async SSE client; captures TTFT + X-Relay-* headers
│   ├── cluster.py         ← scheduler weight/mode control + worker discovery
│   ├── datasets.py        ← ShareGPT download, filter, and working-set builder
│   ├── warmup.py          ← per-worker engine warm-up (autouse session fixture)
│   ├── workload.py        ← prompt generators (batch, multi-turn replay, complexity tiers)
│   ├── metrics.py         ← RoutingRecord aggregation, percentiles, affinity stats
│   └── report.py          ← matplotlib/seaborn plot generators + JSON summary writer
│
├── scenarios/             ← one file per scheduler signal (all: round-robin vs signal)
│   ├── test_memory.py     ← memory (mw): RR vs memory=1 under real RAM/VRAM load
│   ├── test_thermal.py    ← thermal (θw): RR vs thermal=1 under real heat
│   ├── test_jitter.py     ← jitter (jw): RR vs jitter=1 under real network delay
│   ├── test_queue.py      ← queue (qw): RR vs queue=1 under overload (failed-request A/B)
│   ├── test_prefix_cache.py ← prefix_miss: RR vs prefix_miss=1 (conversation affinity)
│   └── test_quality_routing.py ← nu term: RR vs nu=5 (hard prompts → strong model)
│
├── comparison/
│   └── test_vs_baselines.py ← Relay vs round-robin vs single-worker
│
├── load/
│   └── locustfile.py      ← Locust shapes: ramp / spike / sustained
│
├── data/                  ← auto-populated on first --prepare-data run
│   ├── ShareGPT_V3_unfiltered_cleaned_split.json  (400 MB, downloaded once)
│   └── sharegpt_working.json                      (18 MB, filtered working set)
│
└── results/               ← auto-created, one timestamped folder per run
    └── 20260530_033500/
        ├── run_manifest.json
        ├── summary.json
        ├── *_records.csv
        ├── *_records.json
        └── plots/
            ├── memory_worker_distribution.png      # RR vs signal share, per scenario:
            ├── thermal_worker_distribution.png     #   memory / thermal / jitter / queue
            ├── jitter_worker_distribution.png
            ├── queue_worker_distribution.png
            ├── queue_failures.png                  # round-robin vs queue=1 failed requests
            ├── prefix_cache_heatmap.png
            ├── prefix_cache_affinity_comparison.png
            ├── quality_routing_scatter.png
            ├── quality_routing_bins.png
            ├── latency_cdf_comparison.png          # comparison test
            ├── throughput_comparison.png
            ├── relay_worker_heatmap.png
            └── locust_ramp_latency.png             # load test
```

## Scheduler signal test protocol

Every signal scenario is the **same A/B**: run a workload once under the blind
**round-robin** scheduler (baseline), then once with **only that signal's weight
= 1.0** (cost mode, all other weights 0), and compare. The shared harness
`framework/baseline.py::run_round_robin_vs_signal` handles the mode/weight
switching and a full state reset (weights, mode, per-worker overrides) before
each run, so a prior test can never leak routing state in.

There is **no fake telemetry injection.** The `memory`, `thermal`, and `jitter`
signals need a real condition on one worker — you apply it manually and keep it
applied for the whole run. `queue`, `prefix_cache`, and `quality_routing` are
fully automated.

| Scenario | RR → signal | Condition you apply | Reported / asserted |
|----------|-------------|---------------------|---------------------|
| `memory` | RR → `memory=1` | real RAM/VRAM load on worker-0 (`stress-ng --vm`, GPU fill) | share to the pressured worker (RR vs signal) — observational |
| `thermal` | RR → `thermal=1` | real CPU/GPU heat on worker-0 (`stress-ng --cpu`, GPU job) | share to the hot worker — observational |
| `jitter` | RR → `jitter=1` | real delay on worker-0 (`tc qdisc … netem delay`) | share to the jittery worker — observational (waits for the jitter EMA to settle first) |
| `queue` | RR → `queue=1` | none — the burst overloads the cluster | **failed requests**: assert `queue=1 ≤ round-robin` |
| `prefix_cache` | RR → `prefix_miss=1` | none — replays conversations | **affinity**: assert `prefix on ≥ 85%` and `> round-robin` |
| `quality_routing` | RR → `nu=5` | two **different-quality models** (one per worker) | strong-worker share per complexity tier: assert `high > low + 15pts` |

`memory` / `thermal` / `jitter` send 40 short prompts per run by default
(`RELAY_TEST_REQUESTS_PER_PHASE`, concurrency 8, `max_tokens` 8 — routing is
decided before generation). `queue` is tunable via the `RELAY_TEST_OVERLOAD_*`
env vars (see its module docstring). Each scenario writes
`<scenario>_records.{csv,json}` and a worker-distribution plot; `queue` adds a
failures bar chart and `prefix_cache` a heatmap.

## Prefix cache affinity protocol

Two disjoint slices of 20 multi-turn ShareGPT conversations are replayed
turn-by-turn (each turn sends the full growing context). The two slices differ
so the signal run does not inherit the caches the round-robin run warmed.

- **Baseline** — round-robin mode, conversations 0–19
- **Signal** — `prefix_miss=1.0` (all other weights 0), conversations 20–39

- **Assert**: prefix-on overall same-worker rate ≥ 85% (and higher than round-robin)

Round-robin rotates blindly, so a conversation's turns bounce between workers →
low affinity; the prefix term pulls each turn back to the worker holding its
cache → high affinity. `prefix_cache_heatmap.png` shows it directly (solid
one-color rows = perfect affinity). On a fast-GPU + slow-CPU 2-worker pair,
conversations tend to concentrate on the fast worker regardless, which can make
the round-robin baseline noisy — the delta sharpens with comparable nodes.

## Baseline comparison protocol

Three configurations run the same 500-prompt pool:

| Config        | Scheduler behaviour                |
|---------------|------------------------------------|
| `relay`       | Cost-function routing (default)    |
| `round_robin` | Blind round-robin rotation         |
| `single_worker` | All traffic pinned to one worker |

**Assert**: Relay P99 TTFT ≤ round-robin P99 × 1.15 **and** < single-worker P99 × 0.95.

## Locust load shapes

| Shape      | Duration  | Pattern                                          |
|------------|-----------|--------------------------------------------------|
| `ramp`     | 5 min     | 5 → 50 concurrent users over 3 min, hold 2 min  |
| `spike`    | 3 min     | 5 users → instant burst to 80 → back to 5       |
| `sustained`| 10 min    | Constant 40 concurrent users                     |

## Dataset

| Slice             | Count | Used for                        |
|-------------------|-------|---------------------------------|
| `short_prompts`   | 600   | memory / thermal / jitter (40/run)  |
| `mixed_prompts`   | 1000  | queue, load, comparison tests       |
| `conversations`   | 200   | prefix cache affinity               |

Source: [ShareGPT V3](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
— publicly available, no auth required.
