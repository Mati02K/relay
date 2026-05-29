"""Quick router-correctness verifier.

Probes a running coordinator and confirms the quality-aware router is
behaving sensibly:

* trivial prompts should not always pin to the strongest worker
* hard prompts should land on a worker advertising ``model_quality >= 0.8``
  whenever such a worker is online

Run:

    python test/modelrouter/verify_router.py --coordinator http://127.0.0.1:8080

Output prints one line per probe + a final pass/fail summary. Exits 0
if all hard prompts went to high-quality workers, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import httpx

_TRIVIAL_PROMPTS = [
    "Say hi.",
    "What is 2 + 2?",
    "Name a colour.",
]

_HARD_PROMPTS = [
    (
        "Explain step by step why a memory-pressured worker should be "
        "deprioritised, and analyze the trade-offs between thermal awareness "
        "and queue-length awareness in a scheduler. Compare two designs."
    ),
    (
        "Implement a function that takes a list of dictionaries and groups "
        "them by a nested key path. Explain your design choices, handle "
        "missing keys, and justify the complexity of your approach."
    ),
    (
        "Derive the cost function that combines queue depth, KV-cache "
        "overlap, memory pressure, and jitter for a heterogeneous edge "
        "inference cluster. Why do these terms compose additively?"
    ),
]

_HIGH_QUALITY_THRESHOLD = 0.8


@dataclass
class Probe:
    prompt: str
    expected: str  # "high_quality" or "any"
    chosen_node: str
    chosen_quality: float
    headers: dict


def _list_workers(client: httpx.Client, coordinator: str) -> dict[str, float]:
    """Return ``{node_id: model_quality}`` for every online worker."""
    resp = client.get(f"{coordinator}/v1/workers", timeout=10)
    resp.raise_for_status()
    out: dict[str, float] = {}
    for entry in resp.json():
        node = entry.get("node_id") or entry.get("nodeId")
        raw = entry.get("model_quality")
        if raw is None:
            meta = entry.get("metadata") or {}
            raw = meta.get("model_quality", 0.5)
        try:
            quality = float(raw)
        except (TypeError, ValueError):
            quality = 0.5
        if node:
            out[node] = quality
    return out


def _probe_one(client: httpx.Client, coordinator: str, prompt: str) -> tuple[str, dict]:
    """Send one prompt, return ``(chosen_node_id, response_headers)``."""
    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "stream": False,
    }
    resp = client.post(
        f"{coordinator}/v1/chat/completions",
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    chosen = (
        resp.headers.get("x-relay-worker")
        or resp.headers.get("X-Relay-Worker")
        or "?"
    )
    return chosen, dict(resp.headers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--require-high-quality",
        action="store_true",
        help="Fail if any hard prompt lands on a non-high-quality worker.",
    )
    args = parser.parse_args()

    with httpx.Client() as client:
        try:
            quality_by_node = _list_workers(client, args.coordinator)
        except httpx.HTTPError as exc:
            print(f"ERROR: could not reach coordinator: {exc}", file=sys.stderr)
            return 2
        if not quality_by_node:
            print("ERROR: no workers registered at the coordinator", file=sys.stderr)
            return 2

        print(f"Cluster has {len(quality_by_node)} online workers:")
        for node, q in sorted(quality_by_node.items()):
            tier = "STRONG" if q >= _HIGH_QUALITY_THRESHOLD else "weak"
            print(f"  {node:25s} model_quality={q:.2f}  ({tier})")
        print()

        probes: list[Probe] = []
        for prompt in _TRIVIAL_PROMPTS:
            chosen, headers = _probe_one(client, args.coordinator, prompt)
            probes.append(
                Probe(
                    prompt=prompt,
                    expected="any",
                    chosen_node=chosen,
                    chosen_quality=quality_by_node.get(chosen, -1.0),
                    headers=headers,
                )
            )
        for prompt in _HARD_PROMPTS:
            chosen, headers = _probe_one(client, args.coordinator, prompt)
            probes.append(
                Probe(
                    prompt=prompt,
                    expected="high_quality",
                    chosen_node=chosen,
                    chosen_quality=quality_by_node.get(chosen, -1.0),
                    headers=headers,
                )
            )

    print(f"{'kind':6s}  {'chosen':25s}  {'quality':>7s}  prompt[0:60]")
    print("-" * 110)
    fails = 0
    for p in probes:
        kind = "HARD" if p.expected == "high_quality" else "easy"
        ok = (
            p.expected == "any"
            or p.chosen_quality >= _HIGH_QUALITY_THRESHOLD
        )
        marker = "ok " if ok else "BAD"
        if not ok:
            fails += 1
        head = p.prompt[:60].replace("\n", " ")
        print(
            f"{kind:6s}  {p.chosen_node:25s}  {p.chosen_quality:7.2f}  {marker} {head}"
        )

    has_strong = any(q >= _HIGH_QUALITY_THRESHOLD for q in quality_by_node.values())
    if not has_strong:
        print(
            "\nNote: no worker advertises model_quality >= 0.8 — the router "
            "cannot make a quality-routing call in this cluster.",
        )
        return 0

    if fails:
        msg = f"\n{fails} hard prompt(s) did not land on a high-quality worker."
        if args.require_high_quality:
            print(msg, file=sys.stderr)
            return 1
        print(msg + " (re-run with --require-high-quality to fail.)")
        return 0

    print("\nAll hard prompts routed to high-quality workers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
