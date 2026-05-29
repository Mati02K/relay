"""Materialize a deterministic prompt workload for the ablation study.

We don't depend on the ShareGPT dump (it's large, gated, and adds reviewer
friction). Instead we synthesize a small, well-mixed workload with the four
properties the cost function actually needs to be exercised:

  1. Variable lengths (short / medium / long) so prompt-length buckets differ.
  2. A handful of *shared prefixes* repeated across requests, so the
     KV-cache term beta is exercisable.
  3. A small fraction of long prompts (> ~1024 tokens) so the phase-aware
     phi term fires.
  4. Light topical variation so completions aren't all identical.

The output is JSONL with two fields per line: ``prompt_id`` and ``content``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SHARED_PREFIXES: list[str] = [
    "You are a helpful assistant. ",
    "You are a senior software engineer. ",
    "You are a friendly tutor explaining things simply. ",
]

SHORT_QUESTIONS: list[str] = [
    "What is UCSC?",
    "Explain what an LLM is in one sentence.",
    "Why is the sky blue?",
    "List three benefits of distributed systems.",
    "What does TTFT stand for in LLM serving?",
    "Give one example of edge computing.",
    "Define KV cache in transformer inference.",
    "What is etcd used for?",
    "Why is gRPC popular for microservices?",
    "Name a consumer-grade LLM inference engine.",
]

MEDIUM_QUESTIONS: list[str] = [
    "Compare TCP and UDP in two paragraphs and recommend which to use for a low-latency video call.",
    "Walk through the lifecycle of a single HTTP request from the browser to a backend behind a load balancer.",
    "Explain what makes Apple silicon's unified memory architecture different from a discrete GPU setup.",
    "Describe how a distributed key-value store like etcd performs leader election in two paragraphs.",
    "Outline three trade-offs between running an LLM on the edge versus a centralized cloud.",
    "Summarize how a transformer decoder uses a KV cache during autoregressive generation.",
    "Explain what 'closed-loop' versus 'open-loop' load generation means for benchmarking servers.",
    "Describe how server-sent events differ from WebSockets for streaming an LLM response.",
]

LONG_CONTEXT_PARAGRAPHS: list[str] = [
    (
        "The transformer architecture, introduced in 'Attention is All You Need' (2017), "
        "replaced recurrence with self-attention. Each layer projects inputs to query, key, "
        "and value tensors and computes scaled dot-product attention. During autoregressive "
        "decoding, the key and value tensors for previously generated tokens can be cached, "
        "so each new token only requires a single forward pass over the most recent position. "
        "This optimization, called the KV cache, turns generation latency from O(n^2) per "
        "token into O(n), where n is the current sequence length. In production LLM serving "
        "systems the KV cache typically dominates GPU memory usage, often exceeding the "
        "weight footprint for long contexts. Systems like vLLM and Mooncake further exploit "
        "the cache by sharing prefixes across requests so that common system prompts and "
        "few-shot examples don't have to be re-prefilled. "
    ),
    (
        "Edge inference assumes that the device performing the computation is heterogeneous "
        "with respect to its peers: consumer laptops, mini-PCs, and phones differ in CPU "
        "generation, memory capacity, GPU availability, and thermal envelope. A scheduler "
        "for such a fleet cannot assume identical service rates. Instead it must measure "
        "each worker's recent decode tokens-per-second and choose accordingly. Static "
        "weighted round-robin tends to underperform because it ignores transient effects "
        "like thermal throttling, KV-cache locality, and momentary queue depth spikes. "
        "Cost-function based schedulers, which combine queue length, cached-prefix overlap, "
        "memory pressure, and network jitter into a single score, have been shown to improve "
        "tail TTFT by 30-60 percent in heterogeneous edge clusters relative to static "
        "round-robin baselines. "
    ),
    (
        "When designing benchmark workloads for an LLM serving system, it is important "
        "to mix prompt lengths, share prefixes across requests, and vary the requested "
        "completion length. Open-loop generators feed Poisson arrivals into the system "
        "regardless of how fast it can drain them; closed-loop generators wait for one "
        "response before sending the next. Each mode reveals different system behavior: "
        "open-loop exposes saturation and the goodput cliff, while closed-loop is the "
        "cleanest setting to compare TTFT between configurations because the system is "
        "never overloaded. For an ablation study comparing scheduler weights, closed-loop "
        "with a fixed prompt list is the more controllable choice; an open-loop sweep "
        "is then layered on to characterize the goodput curve. "
    ),
]


def build_workload(n: int, seed: int) -> list[dict]:
    """Generate ``n`` prompts mixing short / medium / long with shared prefixes."""
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        roll = rng.random()
        prefix = rng.choice(SHARED_PREFIXES)
        if roll < 0.55:
            body = rng.choice(SHORT_QUESTIONS)
        elif roll < 0.85:
            body = rng.choice(MEDIUM_QUESTIONS)
        else:
            # Long prompt: 3-5 stitched paragraphs followed by a focused ask.
            chunks = [rng.choice(LONG_CONTEXT_PARAGRAPHS) for _ in range(rng.randint(3, 5))]
            ask = rng.choice([
                "Given the above, summarize the main idea in one paragraph.",
                "Given the above, list three concrete consequences.",
                "Given the above, recommend a benchmark methodology.",
            ])
            body = "".join(chunks) + "\n\n" + ask
        out.append({"prompt_id": f"p{i:04d}", "content": prefix + body})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompts = build_workload(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    lengths = [len(p["content"]) for p in prompts]
    print(f"wrote {len(prompts)} prompts to {args.out}")
    print(
        f"  char lengths: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} "
        f"max={max(lengths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
