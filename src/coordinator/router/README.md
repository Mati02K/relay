# `coordinator/router/` — quality- and modality-aware request router

This package owns the routing decisions that sit on top of the
paper-style cost function in `coordinator/scheduler.py`:

* **Quality routing** (RouteLLM-style): score how complex a prompt is,
  bias scheduling toward workers serving higher-quality models.
* **Modality routing**: detect what a request needs (text / image /
  audio / video) and filter out workers that don't support it.

The scheduler imports the public API from `coordinator.router` and
calls into it from `choose_worker`. Nothing else in the codebase
imports this package directly.

## Files

| file | what it does |
|---|---|
| `classifier.py` | `estimate_complexity_score(prompt)` → `[0, 1]`. Heuristic stand-in for the BERT classifier from RouteLLM (Ong et al., 2024). Length signal + reasoning-keyword signal + question-mark signal. |
| `quality.py` | `worker_model_quality(metadata)` reads the worker-advertised quality. `quality_routing_term(nu, complexity, quality)` returns `nu * complexity * (1 - quality)` — the cost added to the scheduler's per-worker score. |
| `modality.py` | `detect_request_modalities(request)` walks `messages[*].content` and returns the modality set (`{"text"}`, `{"text","image"}`, etc.). `worker_supports_modalities(metadata, required)` is the eligibility filter. |
| `__init__.py` | re-exports the public API (5 functions + 4 constants). |

## How it plugs into the scheduler

```text
incoming request
        │
        ▼
detect_request_modalities(request)        ── modality.py
        │
        ▼
filter eligible workers by:
  worker.supports_model(model_id)           (existing)
  worker_supports_modalities(metadata, required) ── modality.py
        │
        ▼
estimate_complexity_score(prompt_text)    ── classifier.py
        │
        ▼
for each eligible worker, _score_worker computes:
  base 5-term cost  (queue + prefix_miss + memory + jitter + thermal)
+ quality_routing_term(nu, complexity, model_quality)  ── quality.py
        │
        ▼
choose_worker returns the min-cost survivor
```

## Environment variables

### On the coordinator
* `RELAY_SCHED_NU_WEIGHT` — float, default `0.0`. Weight of the
  quality term. Set to e.g. `5.0` to enable RouteLLM-style routing.
  Hot-swappable at runtime via `POST /v1/scheduler/weights`.

### On each worker
* `RELAY_MODEL_QUALITY` — float in `[0, 1]`. The worker self-advertises
  this; default `0.5` if unset. Strong models like Gemma-3-4B should
  set this to `1.0`; small models like TinyLlama 1.1B to `0.3`.
* `RELAY_MODALITIES` — comma-separated. Default `"text"`. Examples:
  `RELAY_MODALITIES=text` (text-only worker), `RELAY_MODALITIES=text,image`
  (vision-capable worker).

Both are read in `worker/daemon.py:_router_metadata()` and published as
worker metadata at registration time.

## Response headers (for debugging)

Each `POST /v1/chat/completions` response carries the routing decision
in headers (in addition to the existing scheduler headers):

* `X-Relay-Complexity` — complexity score for this prompt
* `X-Relay-Model-Quality` — chosen worker's quality value
* `X-Relay-Quality-Term` — final `nu * complexity * (1 - quality)` cost contribution

## Public API

```python
from coordinator.router import (
    estimate_complexity_score,
    quality_routing_term,
    worker_model_quality,
    detect_request_modalities,
    worker_modalities,
    worker_supports_modalities,
    DEFAULT_LENGTH_NORM_TOKENS,
    DEFAULT_MODEL_QUALITY,
    DEFAULT_REQUEST_MODALITIES,
    DEFAULT_WORKER_MODALITIES,
)
```

Pure functions, no global state, no I/O. Safe to call from any thread.

## Swapping the classifier

`estimate_complexity_score` is a heuristic. To replace it with the
fine-tuned BERT classifier from the RouteLLM paper:

1. Add the dependency to `pyproject.toml` (`transformers`, `torch`).
2. Load the model once at coordinator startup (e.g. in `main.py`'s
   lifespan handler) and stash it in module state.
3. Reimplement `estimate_complexity_score(prompt_text)` to call
   `model.predict(prompt_text)` and return a value in `[0, 1]`.

Nothing else in the package needs to change — the function signature
and return type are the swap boundary.

## Tests

* `src/coordinator/test/test_router.py` — 17 unit tests covering
  classifier edges, quality lookup and clamping, the cost term,
  modality detection, capability filtering, and end-to-end
  `choose_worker` behaviour with `nu` on/off and image-only requests.
* `test/modelrouter/verify_router.py` — runtime sanity check that hits
  a live coordinator and confirms hard prompts land on high-quality
  workers.

Run the unit tests:

```bash
PYTHONPATH=src python -m pytest src/coordinator/test/test_router.py
```

## Where the design and ablation results live

`SCHEDULER.md` at the repo root — full design rationale, telemetry
inputs, methodology, and Phase A/B/C ablation results.
