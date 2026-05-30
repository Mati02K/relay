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

## Folder structure

```
test/
├── README.md              ← you are here
├── conftest.py            ← pytest fixtures (coordinator URL, dataset, run_dir)
├── run_tests.py           ← main entry point CLI
│
├── framework/             ← shared test infrastructure
│   ├── client.py          ← async SSE client; captures TTFT + X-Relay-* headers
│   ├── cluster.py         ← scheduler weight control + fake telemetry injection
│   ├── datasets.py        ← ShareGPT download, filter, and working-set builder
│   ├── workload.py        ← prompt generators (batch, multi-turn replay, complexity tiers)
│   ├── metrics.py         ← RoutingRecord aggregation, percentiles, affinity stats
│   └── report.py          ← matplotlib/seaborn plot generators + JSON summary writer
│
├── scenarios/             ← one file per scheduler signal
│   ├── test_memory.py     ← mw: VRAM/RAM pressure routing deviation
│   ├── test_thermal.py    ← theta_w: CPU/GPU thermal routing deviation
│   ├── test_jitter.py     ← j_w: network jitter routing deviation
│   ├── test_queue.py      ← qw: queue saturation routing deviation
│   ├── test_prefix_cache.py ← prefix miss: conversation affinity ≥ 90%
│   └── test_quality_routing.py ← nu term: complex prompts → strong worker
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
            ├── memory_worker_distribution.png
            ├── memory_latency_boxes.png
            ├── memory_pressure_vs_routing.png
            ├── thermal_*.png
            ├── jitter_*.png
            ├── queue_*.png
            ├── prefix_cache_heatmap.png
            ├── prefix_cache_affinity.png
            ├── quality_routing_scatter.png
            ├── quality_routing_bins.png
            ├── latency_cdf_comparison.png
            ├── throughput_comparison.png
            ├── relay_worker_heatmap.png
            └── locust_ramp_latency.png
```

## Fake telemetry injection

Signal scenarios (memory, thermal, jitter) inject artificial pressure by
calling a test-only HTTP endpoint on each worker:

```
POST /v1/test/override_telemetry
{"mw": 0.8, "theta_w": null, "jitter_delay_ms": 0}
```

- `mw` overrides the memory pressure value published to etcd
- `theta_w` overrides the thermal pressure value published to etcd
- `jitter_delay_ms` adds an artificial delay to `/health` responses so the
  coordinator's jitter probe observes inflated RTT → elevated `j_w`

All overrides are cleared automatically after each scenario.  Pass
`{"mw": null, "theta_w": null, "jitter_delay_ms": 0}` to clear manually.

## Scheduler signal test protocol

Each signal test follows the same five-phase pattern:

| Phase      | Pressure | Assertion                                        |
|------------|----------|--------------------------------------------------|
| baseline   | 0.0      | Measure baseline routing distribution            |
| moderate   | 0.5      | Routing starts to shift                          |
| high       | 0.8      | Noticeable shift away from pressurised worker    |
| critical   | 0.9–0.95 | ≤ 60% of baseline share (assert)                |
| recovery   | 0.0      | ≥ 55% of baseline restored (assert)             |

120 requests are sent per phase (concurrency = 8) using short ShareGPT prompts.

## Prefix cache affinity protocol

- 20 ShareGPT multi-turn conversations replayed turn-by-turn
- Each turn sends the full growing context (all prior messages)
- **Assert**: overall same-worker rate ≥ 90%

The test sets `prefix_miss=1.0` (max) and drops the other weights to `0.2` so
prefix cache is the dominant routing signal during the test. All weights are
restored to defaults afterwards.

## Baseline comparison protocol

Three configurations run the same 500-prompt pool:

| Config        | Scheduler behaviour                |
|---------------|------------------------------------|
| `relay`       | Cost-function routing (default)    |
| `round_robin` | Blind round-robin rotation         |
| `single_worker` | All traffic pinned to one worker |

**Assert**: Relay P99 TTFT ≤ round-robin P99 × 1.15 and < single-worker P99.

## Locust load shapes

| Shape      | Duration  | Pattern                                          |
|------------|-----------|--------------------------------------------------|
| `ramp`     | 5 min     | 5 → 50 concurrent users over 3 min, hold 2 min  |
| `spike`    | 3 min     | 5 users → instant burst to 80 → back to 5       |
| `sustained`| 10 min    | Constant 40 concurrent users                     |

## Dataset

| Slice             | Count | Used for                        |
|-------------------|-------|---------------------------------|
| `short_prompts`   | 600   | Signal phase tests (120/phase)  |
| `mixed_prompts`   | 1000  | Load / comparison tests         |
| `conversations`   | 200   | Prefix cache affinity           |

Source: [ShareGPT V3](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
— publicly available, no auth required.
