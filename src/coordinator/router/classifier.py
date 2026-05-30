"""Prompt-complexity + intent classifier used by the quality-aware router.

This is the stand-in for the BERT-style classifier from RouteLLM
(Ong et al., 2024). The contract is intentionally narrow so the
heuristic body can be replaced by a fine-tuned model later without
touching the scheduler.

The classifier returns two outputs per prompt:

* ``complexity`` — float in ``[0, 1]``. Drives the ``nu`` cost term.
* ``skills_needed`` — set of strings. Drives the soft skill filter in
  :mod:`coordinator.scheduler`. Always non-empty; a plain prompt
  resolves to ``{"instruct"}``.

The classifier must be:

* cheap (called once per request, on the hot routing path)
* pure (no I/O, no global mutation)
* deterministic
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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

# Keyword sets that trigger a given skill. Order doesn't matter — any
# match adds the skill. Patterns are matched case-insensitively against
# the full prompt text.
_CODING_KEYWORDS: tuple[str, ...] = (
    "code",
    "function",
    "method",
    "class ",
    "def ",
    "import ",
    "compile",
    "traceback",
    "stacktrace",
    "syntax error",
    "exception",
    "refactor",
    "regex",
    "sql",
    "bash",
    "shell",
    "docker",
    "kubernetes",
    "javascript",
    "typescript",
    "python",
    "rust",
    "golang",
    "java ",
    "c++",
    "algorithm",
    "leetcode",
    "unit test",
    "pytest",
    "npm",
    "package.json",
)

_REASONING_KEYWORDS: tuple[str, ...] = (
    "prove",
    "derive",
    "theorem",
    "trade-off",
    "tradeoff",
    "compare and contrast",
    "step by step",
    "step-by-step",
    "analyze",
    "evaluate",
    "justify",
)

# Code fences and code-like punctuation density also count toward
# "this is a coding prompt" — the keyword list alone misses pasted
# tracebacks and short fragments.
_CODE_FENCE = "```"
_CODE_CHAR_RATIO_THRESHOLD = 0.04


@dataclass(frozen=True)
class Classification:
    """Result of running the heuristic classifier on a prompt.

    Attributes:
        complexity: ``[0, 1]`` complexity score; see
            :func:`estimate_complexity_score`.
        skills_needed: Non-empty set of skill tags. Used as a soft filter
            by the scheduler — if any worker advertises every needed
            skill, only those workers are scored; otherwise the
            scheduler falls back to all eligible workers.
    """

    complexity: float
    skills_needed: frozenset[str]


def classify(request_or_prompt: Mapping[str, Any] | str) -> Classification:
    """Run the heuristic classifier on a request body or prompt string.

    Accepts either the raw OpenAI-style request (so the caller can
    forward what they already have) or a pre-extracted prompt string.
    """
    if isinstance(request_or_prompt, str):
        prompt_text = request_or_prompt
    else:
        prompt_text = _prompt_text_from_request(request_or_prompt)
    complexity = estimate_complexity_score(prompt_text)
    skills = detect_skills(prompt_text)
    return Classification(complexity=complexity, skills_needed=skills)


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
    code_blocks = lowered.count(_CODE_FENCE)
    questions = prompt_text.count("?")
    signal_raw = keyword_hits + 2 * code_blocks + 0.5 * questions
    signal_score = min(1.0, signal_raw / 10.0)

    return min(1.0, 0.5 * length_score + 0.5 * signal_score)


def detect_skills(prompt_text: str) -> frozenset[str]:
    """Return the skill tags this prompt needs from a serving model.

    Always includes ``"instruct"`` as a baseline; every serving model
    advertises ``instruct`` so the floor is "any instruct model".
    ``"coding"`` is added when code fences, syntax-heavy text, or coding
    keywords are present. ``"reasoning"`` is added when the prompt asks
    for derivations, comparisons, or step-by-step analysis.

    The set is intentionally minimal: every extra skill narrows the
    soft filter and risks falling through to the all-workers fallback.
    """
    skills = {"instruct"}
    if not prompt_text:
        return frozenset(skills)
    lowered = prompt_text.lower()
    if _CODE_FENCE in lowered or _has_code_signal(lowered):
        skills.add("coding")
    if any(kw in lowered for kw in _REASONING_KEYWORDS):
        skills.add("reasoning")
    return frozenset(skills)


def _has_code_signal(lowered_prompt: str) -> bool:
    """Decide whether a fence-less prompt still looks like code.

    Two-pass check: keyword hits, then a char-density check on
    code-flavoured punctuation (``{``, ``;``, ``=>``, etc.) to catch
    pasted tracebacks and inline snippets.
    """
    if any(kw in lowered_prompt for kw in _CODING_KEYWORDS):
        return True
    if len(lowered_prompt) < 40:
        return False
    code_chars = sum(lowered_prompt.count(ch) for ch in ("{", "}", ";", "=>", "::", "->"))
    return code_chars / len(lowered_prompt) >= _CODE_CHAR_RATIO_THRESHOLD


def _estimate_tokens(text: str) -> int:
    """Approximate token count from word count.

    Real tokenizers split sub-words so this overestimates by ~25% on
    English prose; that is acceptable because the score is normalized
    against :data:`DEFAULT_LENGTH_NORM_TOKENS`.
    """
    return len(_WORD_PATTERN.findall(text))


def _prompt_text_from_request(request: Mapping[str, Any]) -> str:
    """Extract a flat prompt string from an OpenAI chat-completions body."""
    messages = request.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(parts)
