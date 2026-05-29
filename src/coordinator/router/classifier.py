"""Prompt-complexity classifier used by the quality-aware router.

This is the stand-in for the BERT-style classifier from RouteLLM
(Ong et al., 2024). The contract is intentionally narrow so the
heuristic body of :func:`estimate_complexity_score` can be replaced by
a fine-tuned model later without touching the scheduler.

The classifier must be:

* cheap (called once per request, on the hot routing path)
* pure (no I/O, no global mutation)
* bounded in [0, 1]
"""

from __future__ import annotations

import re

DEFAULT_LENGTH_NORM_TOKENS = 2048

_COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "explain",
    "why",
    "how",
    "compare",
    "design",
    "derive",
    "prove",
    "analyze",
    "trade-off",
    "tradeoff",
    "step by step",
    "step-by-step",
    "implement",
    "debug",
    "outline",
    "summarize",
    "summarise",
    "evaluate",
    "justify",
)

_WORD_PATTERN = re.compile(r"\w+")


def _estimate_tokens(text: str) -> int:
    """Approximate token count from word count.

    Real tokenizers split sub-words so this overestimates by ~25% on
    English prose; that is acceptable because the score is normalized
    against :data:`DEFAULT_LENGTH_NORM_TOKENS`.
    """
    return len(_WORD_PATTERN.findall(text))


def estimate_complexity_score(
    prompt_text: str,
    length_norm_tokens: int = DEFAULT_LENGTH_NORM_TOKENS,
) -> float:
    """Return a [0, 1] heuristic complexity score for a prompt.

    The score combines three signals, weighted equally between the
    length signal and the lexical/structural signal:

    * length signal: prompt token count divided by ``length_norm_tokens``
    * lexical signal: occurrences of reasoning keywords + code fences
    * structural signal: number of question marks

    A real classifier (e.g. the fine-tuned BERT from the RouteLLM
    paper) can replace this function while keeping the same signature.
    """
    if not prompt_text:
        return 0.0

    tokens = _estimate_tokens(prompt_text)
    length_score = min(1.0, tokens / max(1, length_norm_tokens))

    lowered = prompt_text.lower()
    keyword_hits = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in lowered)
    code_blocks = lowered.count("```")
    questions = prompt_text.count("?")
    signal_raw = keyword_hits + 2 * code_blocks + 0.5 * questions
    signal_score = min(1.0, signal_raw / 10.0)

    return min(1.0, 0.5 * length_score + 0.5 * signal_score)
