| config | n_ok | rejected | ttft p50 | ttft p90 | ttft p99 | total p50 | decode tok/s | worker mix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rr | 30 | 0 | 30 | 71 | 778 | 479 | 107.0 | worker-a:11, worker-b:11, worker-c:8 |
| no_cache | 30 | 0 | 29 | 73 | 241 | 483 | 103.9 | worker-a:30 |
| no_jitter | 30 | 0 | 28 | 32 | 104 | 467 | 108.1 | worker-a:25, worker-b:5 |
| no_thermal | 30 | 0 | 26 | 32 | 46 | 465 | 109.2 | worker-a:30 |
| full | 30 | 0 | 53 | 188 | 774 | 507 | 105.7 | worker-a:20, worker-b:10 |
| full_slo | 30 | 0 | 22 | 27 | 29 | 480 | 100.9 | worker-a:30 |
| full_slo_tight | 10 | 20 | 57 | 765 | 900 | 926 | 43.2 | worker-a:10 |
| routellm_nu | 30 | 0 | 43 | 141 | 756 | 474 | 110.6 | worker-a:27, worker-b:3 |
| full_gemma | 30 | 0 | 113 | 336 | 2250 | 637 | 68.2 | worker-a:14, worker-b:16 |
| routellm_nu_gemma | 30 | 0 | 32 | 204 | 274 | 561 | 69.3 | worker-a:14, worker-b:16 |
| routellm_nu_gemma_strong | 30 | 0 | 184 | 250 | 1508 | 1654 | 60.1 | worker-a:18, worker-b:12 |
