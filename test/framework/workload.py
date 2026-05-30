"""Prompt workload generators — concurrent dispatch and multi-turn replay."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from framework.client import RelayClient, RoutingRecord


async def send_batch(
    client: RelayClient,
    prompts: list[list[dict[str, str]]],
    *,
    scenario: str,
    phase: str,
    concurrency: int = 8,
    max_tokens: int = 64,
    model: str | None = None,
) -> list[RoutingRecord]:
    """Send all ``prompts`` through ``client`` with bounded concurrency.

    Returns records in completion order (not submission order).
    """
    sem = asyncio.Semaphore(concurrency)
    records: list[RoutingRecord] = []

    async def _one(msgs: list[dict[str, str]]) -> None:
        async with sem:
            record = await client.chat_completion(
                msgs,
                scenario=scenario,
                phase=phase,
                max_tokens=max_tokens,
                model=model,
            )
            records.append(record)

    await asyncio.gather(*(_one(p) for p in prompts))
    return records


async def replay_conversation(
    client: RelayClient,
    messages: list[dict[str, str]],
    *,
    scenario: str,
    phase: str,
    conversation_id: str,
    max_tokens: int = 64,
    model: str | None = None,
    turn_delay_seconds: float = 0.0,
) -> list[RoutingRecord]:
    """Replay one multi-turn conversation and return per-turn routing records.

    Each successive turn extends the context by accumulating the growing
    ``messages`` list (user + assistant alternating), which exercises the
    coordinator's prefix-cache routing.  The assistant content in
    ``messages`` is taken directly from the ShareGPT dataset — the inference
    worker is not called to generate the previous response.

    The conversation is replayed sequentially (turn 1 → 2 → … → N) because
    turn N's context includes all preceding messages.
    """
    records: list[RoutingRecord] = []
    context: list[dict[str, str]] = []
    turn_idx = 0

    for i, msg in enumerate(messages):
        context.append(msg)
        if msg["role"] != "user":
            continue
        turn_idx += 1
        record = await client.chat_completion(
            list(context),
            scenario=scenario,
            phase=phase,
            max_tokens=max_tokens,
            model=model,
            conversation_id=conversation_id,
            turn=turn_idx,
        )
        records.append(record)
        if turn_delay_seconds > 0:
            await asyncio.sleep(turn_delay_seconds)

    return records


async def replay_conversations_concurrent(
    client: RelayClient,
    conversations: list[list[dict[str, str]]],
    *,
    scenario: str,
    phase: str,
    concurrency: int = 4,
    max_tokens: int = 64,
    model: str | None = None,
) -> list[RoutingRecord]:
    """Replay multiple conversations with bounded concurrency.

    Each conversation is replayed sequentially within itself (to preserve
    prefix ordering), but conversations run concurrently up to ``concurrency``
    at a time.
    """
    sem = asyncio.Semaphore(concurrency)
    all_records: list[RoutingRecord] = []

    async def _one_conv(conv: list[dict[str, str]], conv_id: str) -> None:
        async with sem:
            records = await replay_conversation(
                client,
                conv,
                scenario=scenario,
                phase=phase,
                conversation_id=conv_id,
                max_tokens=max_tokens,
                model=model,
            )
            all_records.extend(records)

    conv_ids = [f"conv_{i:04d}" for i in range(len(conversations))]
    await asyncio.gather(*(_one_conv(c, cid) for c, cid in zip(conversations, conv_ids)))
    return all_records


def slow_prompts(
    n: int,
    min_chars: int = 50,
    max_chars: int = 200,
) -> list[list[dict[str, str]]]:
    """Generate ``n`` synthetic prompts that ask for long responses.

    Used to saturate a worker's queue depth (``qw``) for queue-routing tests.
    """
    topics = [
        "Explain the entire history of the Roman Empire in exhaustive detail.",
        "Write a comprehensive 2000-word essay on climate change and its global effects.",
        "Describe in great detail how the internet works, from physical cables to web browsers.",
        "Provide a thorough explanation of quantum mechanics for a physics PhD student.",
        "Write a long story about a spaceship crew stranded on an alien planet.",
    ]
    rng = random.Random(42)
    return [[{"role": "user", "content": rng.choice(topics)}] for _ in range(n)]


def complexity_tier_prompts() -> dict[str, list[list[dict[str, str]]]]:
    """Return prompts organised into complexity tiers for quality-routing tests.

    Each tier maps to a set of MT-Bench-style prompts spanning from trivial
    (low complexity) to expert-level reasoning (high complexity).  The
    coordinator's RouteLLM complexity classifier should score them
    correspondingly so the ``nu`` term routes high-complexity prompts to the
    stronger worker.
    """
    return {
        "low": [
            [{"role": "user", "content": "What is 2 + 2?"}],
            [{"role": "user", "content": "What color is the sky?"}],
            [{"role": "user", "content": "Name three fruits."}],
            [{"role": "user", "content": "What is the capital of France?"}],
            [{"role": "user", "content": "How do you say hello in Spanish?"}],
            [{"role": "user", "content": "What is a noun?"}],
            [{"role": "user", "content": "What year did World War II end?"}],
            [{"role": "user", "content": "List the days of the week."}],
        ],
        "medium": [
            [{"role": "user", "content": "Explain the water cycle in 3 sentences."}],
            [{"role": "user", "content": "What are the main differences between Python and Java?"}],
            [{"role": "user", "content": "Summarize the plot of Romeo and Juliet."}],
            [{"role": "user", "content": "How does photosynthesis work?"}],
            [{"role": "user", "content": "What are the causes of inflation?"}],
            [{"role": "user", "content": "Explain what recursion is in programming."}],
            [{"role": "user", "content": "What is the difference between DNA and RNA?"}],
            [{"role": "user", "content": "How does a CPU execute a program?"}],
        ],
        "high": [
            [{"role": "user", "content": (
                "Prove that there are infinitely many prime numbers using Euclid's proof, "
                "then explain what Riemann's hypothesis implies about their distribution."
            )}],
            [{"role": "user", "content": (
                "Derive the time complexity of merge sort from first principles "
                "and explain why it achieves the comparison-sort lower bound of O(n log n)."
            )}],
            [{"role": "user", "content": (
                "Explain the CAP theorem and describe a real distributed system design "
                "that makes a deliberate trade-off between consistency and availability."
            )}],
            [{"role": "user", "content": (
                "Walk through the backpropagation algorithm step by step for a 3-layer "
                "neural network, including the chain rule application at each layer."
            )}],
            [{"role": "user", "content": (
                "Explain Gödel's incompleteness theorems and their implications for "
                "the foundations of mathematics. What does it mean for a formal system "
                "to be consistent and complete?"
            )}],
            [{"role": "user", "content": (
                "Write a correct, O(n log n) solution to the longest increasing subsequence "
                "problem and prove its correctness using a loop invariant."
            )}],
            [{"role": "user", "content": (
                "Explain how transformers work, starting from the attention mechanism, "
                "through multi-head attention, positional encodings, and layer normalization, "
                "then describe why the architecture scales so well."
            )}],
            [{"role": "user", "content": (
                "Derive Maxwell's equations in differential form from their integral form "
                "using Stokes' and Gauss's theorems, and explain their physical meaning."
            )}],
        ],
    }
