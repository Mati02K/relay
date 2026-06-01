"""Locust load test file for Relay.

Three shapes are defined.  Select one at runtime with the ``--shape`` env var:

  RELAY_LOAD_SHAPE=ramp       python -m locust -f test/load/locustfile.py
  RELAY_LOAD_SHAPE=spike      ...
  RELAY_LOAD_SHAPE=sustained  ...

Running via run_tests.py
------------------------
  python test/run_tests.py --load --coordinator http://host:8080

That will run all three shapes in sequence and write CSV results to
``test/results/<timestamp>/``.

Manual run (headless)
---------------------
  uv run locust -f test/load/locustfile.py \\
      --host http://192.168.1.10:8080 \\
      --headless --users 50 --spawn-rate 5 --run-time 5m \\
      --csv test/results/manual/locust_ramp

Dataset
-------
The file ``test/data/sharegpt_working.json`` must exist.  Run the dataset
preparation step first if it does not:

  python -c "
  from pathlib import Path
  from framework.datasets import ShareGPTDataset
  import sys; sys.path.insert(0,'test')
  ShareGPTDataset(Path('test/data')).mixed_prompts()
  "
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow `from framework.x import …` when running locust from repo root
_test_dir = Path(__file__).resolve().parent.parent
if str(_test_dir) not in sys.path:
    sys.path.insert(0, str(_test_dir))

from locust import HttpUser, LoadTestShape, between, events, task  # type: ignore[import-untyped]

# ── Prompt pool ───────────────────────────────────────────────────────────────

_WORKING_SET_PATH = _test_dir / "data" / "sharegpt_working.json"
_PROMPTS: list[list[dict[str, str]]] = []


def _load_prompts() -> list[list[dict[str, str]]]:
    if _WORKING_SET_PATH.exists():
        data = json.loads(_WORKING_SET_PATH.read_text())
        return data.get("mixed_prompts", [])
    # Fallback: synthetic prompts so the test can still run without the dataset
    print("[locust] WARNING: sharegpt_working.json not found — using synthetic prompts")
    topics = [
        "Explain how TCP/IP works.",
        "What is gradient descent?",
        "How does garbage collection work in Java?",
        "What are the SOLID principles?",
        "Explain the difference between SQL and NoSQL databases.",
    ]
    return [[{"role": "user", "content": t}] for t in topics * 200]


@events.init.add_listener
def on_init(environment: Any, **kwargs: Any) -> None:
    global _PROMPTS
    _PROMPTS = _load_prompts()
    print(f"[locust] Loaded {len(_PROMPTS)} prompts")


# ── User class ────────────────────────────────────────────────────────────────

class RelayUser(HttpUser):
    """Simulates one concurrent user hitting ``/v1/chat/completions``."""

    wait_time = between(0.05, 0.3)

    @task
    def chat_completion(self) -> None:
        if not _PROMPTS:
            return
        messages = random.choice(_PROMPTS)
        body = {"messages": messages, "max_tokens": 64, "stream": True}
        t_start = time.perf_counter()
        ttft_ms: float | None = None

        with self.client.post(
            "/v1/chat/completions",
            json=body,
            stream=True,
            catch_response=True,
            name="/v1/chat/completions",
        ) as response:
            if response.status_code >= 400:
                response.failure(f"HTTP {response.status_code}")
                return

            for raw in response.iter_lines():
                line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                if ttft_ms is None and _has_content(payload):
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0

            response.success()

        worker = response.headers.get("x-relay-worker", "unknown")
        self.environment.events.request.fire(
            request_type="SSE",
            name="ttft",
            response_time=ttft_ms or 0.0,
            response_length=0,
            context={"worker": worker},
        )


def _has_content(payload: str) -> bool:
    try:
        obj = json.loads(payload)
        choices = obj.get("choices") or []
        if not choices:
            return False
        delta = choices[0].get("delta") or {}
        return bool(delta.get("content"))
    except Exception:
        return False


# ── Load shape ────────────────────────────────────────────────────────────────
# Stage tables per shape. A stage applies until ``duration`` seconds of run time.
#   ramp      gradual 5 → 50 users over 3 min, hold 2 min
#   spike     5 users → instant burst to 80 → back to 5
#   sustained constant 40 users for 10 min (stability / soak)
_SHAPES: dict[str, list[dict[str, int]]] = {
    "ramp": [
        {"duration": 30,  "users": 5,  "spawn_rate": 5},
        {"duration": 60,  "users": 15, "spawn_rate": 5},
        {"duration": 120, "users": 30, "spawn_rate": 5},
        {"duration": 180, "users": 50, "spawn_rate": 5},
        {"duration": 300, "users": 50, "spawn_rate": 1},
    ],
    "spike": [
        {"duration": 60,  "users": 5,  "spawn_rate": 5},
        {"duration": 75,  "users": 80, "spawn_rate": 80},
        {"duration": 180, "users": 5,  "spawn_rate": 10},
    ],
    "sustained": [
        {"duration": 600, "users": 40, "spawn_rate": 10},
    ],
}


class RelayLoadShape(LoadTestShape):
    """The single load shape, selected at import time by ``RELAY_LOAD_SHAPE``.

    Locust errors if a file defines more than one ``LoadTestShape``, so there is
    exactly one class here; its stage table is chosen from :data:`_SHAPES`.
    """

    stages = _SHAPES.get(os.getenv("RELAY_LOAD_SHAPE", "ramp"), _SHAPES["ramp"])

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
