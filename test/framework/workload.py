"""Prompt workload generators — concurrent dispatch and multi-turn replay."""

from __future__ import annotations

import asyncio
import random
import time

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
    category: str = "",
) -> list[RoutingRecord]:
    """Send all ``prompts`` through ``client`` with bounded concurrency.

    Each record is tagged with ``category`` so routing can be grouped by prompt
    kind (e.g. trivial / chat / reasoning / coding). Returns records in
    completion order (not submission order).
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
            record.category = category
            records.append(record)

    await asyncio.gather(*(_one(p) for p in prompts))
    return records


async def send_constant_rate(
    client: RelayClient,
    prompts: list[list[dict[str, str]]],
    *,
    scenario: str,
    phase: str,
    rps: float,
    duration: float,
    run_start: float,
    max_tokens: int = 64,
    model: str | None = None,
) -> list[RoutingRecord]:
    """Fire requests at a fixed arrival rate (open-loop) for ``duration`` seconds.

    Unlike :func:`send_batch`, a new request is launched every ``1/rps`` seconds
    regardless of how many are still in flight — so offered load stays constant
    even as workers die. This is what makes throughput collapse and recovery
    visible during a chaos run. Prompts are cycled if exhausted. ``run_start`` is
    a shared ``time.perf_counter()`` reference so records align with the sampler.
    """
    records: list[RoutingRecord] = []
    tasks: list[asyncio.Task[None]] = []
    interval = 1.0 / rps
    n_prompts = len(prompts)
    deadline = run_start + duration

    async def _one(msgs: list[dict[str, str]]) -> None:
        record = await client.chat_completion(
            msgs, scenario=scenario, phase=phase, max_tokens=max_tokens, model=model
        )
        records.append(record)

    i = 0
    next_fire = run_start
    while time.perf_counter() < deadline:
        tasks.append(asyncio.create_task(_one(prompts[i % n_prompts])))
        i += 1
        next_fire += interval
        sleep = next_fire - time.perf_counter()
        if sleep > 0:
            await asyncio.sleep(sleep)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
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
            [
                {
                    "role": "user",
                    "content": (
                        "Prove that there are infinitely many prime numbers using Euclid's proof, "
                        "then explain what Riemann's hypothesis implies about their distribution."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Derive the time complexity of merge sort from first principles "
                        "and explain why it achieves the comparison-sort lower bound of O(n log n)."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Explain the CAP theorem and describe a real distributed system design "
                        "that makes a deliberate trade-off between consistency and availability."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Walk through the backpropagation algorithm step by step for a 3-layer "
                        "neural network, including the chain rule application at each layer."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Explain Gödel's incompleteness theorems and their implications for "
                        "the foundations of mathematics. What does it mean for a formal system "
                        "to be consistent and complete?"
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Write a correct, O(n log n) solution to the longest increasing "
                        "subsequence problem and prove its correctness using a loop invariant."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Explain how transformers work, starting from the attention "
                        "mechanism, through multi-head attention, positional encodings, "
                        "and layer normalization, then describe why it scales so well."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Derive Maxwell's equations in differential form from their integral form "
                        "using Stokes' and Gauss's theorems, and explain their physical meaning."
                    ),
                }
            ],
        ],
    }


def category_prompts() -> dict[str, list[list[dict[str, str]]]]:
    """Return prompts grouped by intent category for routing-by-category tests.

    Four categories, each the natural target of a different model in the
    cluster:

    * ``trivial``   — simple lookups → cheapest model (e.g. qwen2.5-1.5b)
    * ``chat``      — general prose → mid-quality chat model (e.g. mistral-7b)
    * ``reasoning`` — proofs/derivations/analysis → strong reasoning model
                      (e.g. qwen2.5-7b); kept free of coding keywords so the
                      soft skill filter doesn't divert them to the coder
    * ``coding``    — write/debug/explain code (fenced) → coding model
                      (e.g. qwen2.5-coder-7b) via the coding skill filter

    The classifier should score trivial near 0 and reasoning/coding well above
    0.5 so the ``nu`` term (with queue) escalates the hard categories.
    """
    return {
        "trivial": [
            [{"role": "user", "content": "What is 2 + 2?"}],
            [{"role": "user", "content": "What color is the sky?"}],
            [{"role": "user", "content": "Name three fruits."}],
            [{"role": "user", "content": "What is the capital of France?"}],
            [{"role": "user", "content": "Translate the word 'hello' into Spanish."}],
            [{"role": "user", "content": "What is a noun?"}],
            [{"role": "user", "content": "What year did World War II end?"}],
            [{"role": "user", "content": "List the days of the week."}],
            [{"role": "user", "content": "What is the largest planet in our solar system?"}],
            [{"role": "user", "content": "Give me a synonym for 'happy'."}],
        ],
        "chat": [
            [{"role": "user", "content": "Tell me about the history of the Roman Empire."}],
            [{"role": "user", "content": "Describe a typical day in ancient Egypt."}],
            [{"role": "user", "content": "Recommend three classic novels worth reading."}],
            [{"role": "user", "content": "Write a short paragraph about the autumn season."}],
            [{"role": "user", "content": "Give me a few fun facts about dolphins."}],
            [{"role": "user", "content": "Describe what a thunderstorm feels like."}],
            [{"role": "user", "content": "Tell me a short story about a lighthouse keeper."}],
            [{"role": "user", "content": "What are some popular tourist attractions in Japan?"}],
        ],
        "reasoning": [
            [
                {
                    "role": "user",
                    "content": (
                        "Prove that there are infinitely many prime numbers "
                        "using Euclid's argument."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Explain the CAP theorem and analyze a real trade-off between consistency "
                        "and availability in a distributed system."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Explain Gödel's incompleteness theorems and justify their "
                        "implications for the foundations of mathematics."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Derive Maxwell's equations in differential form from their integral form "
                        "and explain their physical meaning."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Compare and contrast utilitarian and deontological ethics with examples, "
                        "and justify which is more defensible."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": ("Prove that the square root of 2 is irrational, step by step."),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Analyze the main causes of the 2008 financial crisis and evaluate which "
                        "was most significant."
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Explain why entropy tends to increase in an isolated system, deriving the "
                        "intuition from statistical mechanics."
                    ),
                }
            ],
        ],
        "coding": [
            [
                {
                    "role": "user",
                    "content": (
                        "Write a Python function to reverse a singly linked list:\n"
                        "```python\n"
                        "class Node:\n"
                        "    def __init__(self, val):\n"
                        "        self.val = val\n"
                        "        self.next = None\n"
                        "```"
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Debug this Python code, it returns the wrong result:\n"
                        "```python\ndef add(a, b):\n    return a - b\n```"
                    ),
                }
            ],
            [{"role": "user", "content": "Implement a binary search algorithm in Rust."}],
            [
                {
                    "role": "user",
                    "content": (
                        "Refactor this JavaScript function to be more efficient:\n"
                        "```javascript\n"
                        "function sum(arr) {\n"
                        "  let s = 0;\n"
                        "  for (const x of arr) s += x;\n"
                        "  return s;\n"
                        "}\n"
                        "```"
                    ),
                }
            ],
            [{"role": "user", "content": "Explain what this regex does: `^\\d{3}-\\d{4}$`."}],
            [
                {
                    "role": "user",
                    "content": ("Write a SQL query to find the top 5 customers by total revenue."),
                }
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "Fix the syntax error in this Python snippet:\n"
                        "```python\nfor i in range(10) print(i)\n```"
                    ),
                }
            ],
            [
                {
                    "role": "user",
                    "content": ("Write pytest unit tests for a function that adds two integers."),
                }
            ],
            [{"role": "user", "content": "Implement quicksort in Python with a clear docstring."}],
            [{"role": "user", "content": "Convert this bash one-liner into a Dockerfile step."}],
        ],
    }
