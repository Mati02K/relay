"""Dataset management — download, filter, and expose ShareGPT working sets.

The full ShareGPT V3 JSON (~400 MB) is downloaded once and cached in the
``data/`` directory.  A pre-processed working set (~18 MB) is extracted on
first use and cached as ``data/sharegpt_working.json`` so subsequent runs
start immediately.

Working-set slices
------------------
``short_prompts``    600 single-turn prompts, 165–990 chars (~50–300 tokens)
``mixed_prompts``   1000 single-turn prompts, up to 6750 chars (~2048 tokens)
``conversations``    200 multi-turn conversations with 5–20 human turns

All prompts are converted to OpenAI ``messages`` format.  The multi-turn
conversations carry the full ShareGPT exchange (human + GPT turns) so they
can be replayed in order to exercise prefix-cache routing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import httpx

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
WORKING_SET_FILENAME = "sharegpt_working.json"

# Character-length proxies for token ranges (4 chars ≈ 1 token)
SHORT_MIN_CHARS = 165    # ≈ 50 tokens
SHORT_MAX_CHARS = 990    # ≈ 300 tokens
MIXED_MAX_CHARS = 6750   # ≈ 2048 tokens
MIN_CONV_TURNS = 5
MAX_CONV_TURNS = 20

SHORT_PROMPTS_TARGET = 600
MIXED_PROMPTS_TARGET = 1000
CONVERSATIONS_TARGET = 200

SEED = 42


class ShareGPTDataset:
    """Lazy-loaded, cached ShareGPT working set."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._working_set: dict[str, Any] | None = None

    def short_prompts(self, n: int = SHORT_PROMPTS_TARGET) -> list[list[dict[str, str]]]:
        """Return up to ``n`` short single-turn prompt message lists."""
        ws = self._load_working_set()
        return ws["short_prompts"][:n]

    def mixed_prompts(self, n: int = MIXED_PROMPTS_TARGET) -> list[list[dict[str, str]]]:
        """Return up to ``n`` mixed-length single-turn prompt message lists."""
        ws = self._load_working_set()
        return ws["mixed_prompts"][:n]

    def conversations(self, n: int = CONVERSATIONS_TARGET) -> list[list[dict[str, str]]]:
        """Return up to ``n`` multi-turn conversations.

        Each conversation is a flat list of messages alternating
        ``user`` / ``assistant`` roles, exactly as ShareGPT recorded them.
        Callers should replay them in order and slice at each ``user`` turn to
        simulate the growing context of a real multi-turn session.
        """
        ws = self._load_working_set()
        return ws["conversations"][:n]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_working_set(self) -> dict[str, Any]:
        if self._working_set is not None:
            return self._working_set
        working_path = self.data_dir / WORKING_SET_FILENAME
        if working_path.exists():
            self._working_set = json.loads(working_path.read_text())
            return self._working_set
        print("Building ShareGPT working set (first run)…")
        raw_path = self.data_dir / SHAREGPT_FILENAME
        if not raw_path.exists():
            _download(SHAREGPT_URL, raw_path)
        self._working_set = _build_working_set(raw_path, working_path)
        return self._working_set


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with a progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  → {dest}  (this may take a few minutes the first time)")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%  ({downloaded >> 20} MB / {total >> 20} MB)", end="")
    print(f"\n  Download complete → {dest}")


def _build_working_set(raw_path: Path, working_path: Path) -> dict[str, Any]:
    """Parse the raw ShareGPT file and extract the three working-set slices."""
    print("  Parsing raw JSON…")
    raw = json.loads(raw_path.read_text())
    rng = random.Random(SEED)
    rng.shuffle(raw)

    short_prompts: list[list[dict[str, str]]] = []
    mixed_prompts: list[list[dict[str, str]]] = []
    conversations: list[list[dict[str, str]]] = []

    for item in raw:
        convs = item.get("conversations") or []
        if not convs:
            continue
        if not _is_english(convs):
            continue

        human_turns = [c for c in convs if c.get("from") == "human"]
        if not human_turns:
            continue

        first_human = human_turns[0].get("value", "")
        char_len = len(first_human)

        # Single-turn slices use only the first human message
        first_message = [{"role": "user", "content": first_human}]

        if SHORT_MIN_CHARS <= char_len <= SHORT_MAX_CHARS:
            if len(short_prompts) < SHORT_PROMPTS_TARGET:
                short_prompts.append(first_message)

        if char_len <= MIXED_MAX_CHARS:
            if len(mixed_prompts) < MIXED_PROMPTS_TARGET:
                mixed_prompts.append(first_message)

        # Multi-turn conversations
        if len(conversations) < CONVERSATIONS_TARGET:
            messages = _conv_to_messages(convs)
            n_user = sum(1 for m in messages if m["role"] == "user")
            if MIN_CONV_TURNS <= n_user <= MAX_CONV_TURNS:
                conversations.append(messages)

        if (
            len(short_prompts) >= SHORT_PROMPTS_TARGET
            and len(mixed_prompts) >= MIXED_PROMPTS_TARGET
            and len(conversations) >= CONVERSATIONS_TARGET
        ):
            break

    result = {
        "short_prompts": short_prompts,
        "mixed_prompts": mixed_prompts,
        "conversations": conversations,
        "stats": {
            "short_prompts": len(short_prompts),
            "mixed_prompts": len(mixed_prompts),
            "conversations": len(conversations),
        },
    }
    working_path.write_text(json.dumps(result))
    print(
        f"  Working set saved → {working_path}  "
        f"({len(short_prompts)} short, {len(mixed_prompts)} mixed, "
        f"{len(conversations)} conversations)"
    )
    return result


def _conv_to_messages(convs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert ShareGPT ``human/gpt`` turns to OpenAI ``user/assistant`` format."""
    role_map = {"human": "user", "gpt": "assistant"}
    messages = []
    for turn in convs:
        role = role_map.get(turn.get("from", ""), "")
        content = turn.get("value", "").strip()
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def _is_english(convs: list[dict[str, str]]) -> bool:
    """Heuristic: conversation is English if >80% of chars are ASCII."""
    text = " ".join(c.get("value", "") for c in convs[:3])
    if not text:
        return False
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / max(len(text), 1) > 0.80
