"""Prefix-cache hash helpers shared by the coordinator and workers.

The current implementation is intentionally engine-independent: it computes
stable, chained hashes from a canonical prompt representation. When a real
tokenizer is wired in, the same module can hash model token IDs instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

DEFAULT_PREFIX_HASH_SCHEME: Final[str] = "relay-text-approx-sha256-chain-v1"
DEFAULT_PREFIX_TOKENIZER_ID: Final[str] = "utf8-approx-v1"
DEFAULT_PREFIX_BLOCK_SIZE_TOKENS: Final[int] = 16
DEFAULT_PREFIX_BYTES_PER_TOKEN: Final[float] = 4.0
DEFAULT_PREFIX_CACHE_BLOCKS: Final[int] = 512


@dataclass(frozen=True)
class PrefixHashConfig:
    """Configuration that defines whether two prefix hashes are comparable."""

    hash_scheme: str = DEFAULT_PREFIX_HASH_SCHEME
    model_id: str = "unknown"
    tokenizer_id: str = DEFAULT_PREFIX_TOKENIZER_ID
    block_size_tokens: int = DEFAULT_PREFIX_BLOCK_SIZE_TOKENS
    bytes_per_token: float = DEFAULT_PREFIX_BYTES_PER_TOKEN
    max_blocks: int = DEFAULT_PREFIX_CACHE_BLOCKS

    @classmethod
    def from_env(cls) -> PrefixHashConfig:
        """Build prefix-cache hash settings from worker/coordinator environment."""
        model_id = os.getenv("RELAY_MODEL_ID") or os.getenv("LLAMA_MODEL_ID")
        if not model_id:
            model_path = os.getenv("LLAMA_MODEL_PATH", "")
            model_id = Path(model_path).name if model_path else "unknown"

        return cls(
            hash_scheme=os.getenv("RELAY_PREFIX_HASH_SCHEME", DEFAULT_PREFIX_HASH_SCHEME),
            model_id=model_id,
            tokenizer_id=os.getenv("RELAY_TOKENIZER_ID", DEFAULT_PREFIX_TOKENIZER_ID),
            block_size_tokens=int(
                os.getenv(
                    "RELAY_PREFIX_BLOCK_SIZE_TOKENS",
                    str(DEFAULT_PREFIX_BLOCK_SIZE_TOKENS),
                )
            ),
            bytes_per_token=float(
                os.getenv(
                    "RELAY_PREFIX_BYTES_PER_TOKEN",
                    str(DEFAULT_PREFIX_BYTES_PER_TOKEN),
                )
            ),
            max_blocks=int(
                os.getenv(
                    "RELAY_KV_HASH_CACHE_CHUNKS",
                    str(DEFAULT_PREFIX_CACHE_BLOCKS),
                )
            ),
        )

    def comparable_key(self) -> tuple[str, str, str, int, float]:
        """Return the fields that must match before comparing prefix hashes."""
        return (
            self.hash_scheme,
            self.model_id,
            self.tokenizer_id,
            self.block_size_tokens,
            self.bytes_per_token,
        )


@dataclass(frozen=True)
class PrefixHashResult:
    """Prefix block hashes for one prompt."""

    config: PrefixHashConfig
    block_hashes: list[str]
    estimated_tokens: int


class PrefixHashLRU:
    """Bounded recent set of prefix hashes in least-recently-used order."""

    def __init__(self, max_blocks: int) -> None:
        self._max_blocks = max(1, max_blocks)
        self._hashes: OrderedDict[str, None] = OrderedDict()

    def observe(self, block_hashes: Iterable[str]) -> None:
        """Mark prefix hashes as recently observed and evict stale hashes."""
        for block_hash in block_hashes:
            if block_hash in self._hashes:
                self._hashes.move_to_end(block_hash)
            else:
                self._hashes[block_hash] = None
            while len(self._hashes) > self._max_blocks:
                self._hashes.popitem(last=False)

    def snapshot(self) -> list[str]:
        """Return hashes from oldest to newest."""
        return list(self._hashes.keys())


def messages_to_prefix_text(messages: object) -> str:
    """Convert OpenAI-style messages into stable text for prefix hashing.

    This is not a model chat template. It is a deterministic routing key used
    until model tokenizer integration is available.
    """
    normalized = _normalize_messages(messages)
    if not normalized:
        return ""
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def request_to_prefix_text(request: Mapping[str, object]) -> str:
    """Extract the hashable prompt text from an OpenAI-style request body."""
    return messages_to_prefix_text(request.get("messages"))


def estimate_tokens_from_text(
    text: str,
    bytes_per_token: float = DEFAULT_PREFIX_BYTES_PER_TOKEN,
) -> int:
    """Estimate token count when a real tokenizer is unavailable."""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text.encode("utf-8")) / max(bytes_per_token, 1e-6))))


def compute_prefix_hashes_from_text(
    text: str,
    config: PrefixHashConfig | None = None,
) -> PrefixHashResult:
    """Compute chained approximate prefix-block hashes from canonical text."""
    config = config or PrefixHashConfig.from_env()
    estimated_tokens = estimate_tokens_from_text(text, config.bytes_per_token)
    block_bytes = max(1, int(math.ceil(config.block_size_tokens * config.bytes_per_token)))
    data = text.encode("utf-8")
    blocks = [
        data[start : start + block_bytes]
        for start in range(0, len(data) - len(data) % block_bytes, block_bytes)
    ]
    return PrefixHashResult(
        config=config,
        block_hashes=_compute_chained_hashes(blocks, config),
        estimated_tokens=estimated_tokens,
    )


def compute_prefix_hashes_from_token_ids(
    token_ids: Sequence[int],
    config: PrefixHashConfig | None = None,
) -> PrefixHashResult:
    """Compute Mooncake/vLLM-style chained hashes from full token blocks."""
    config = config or PrefixHashConfig.from_env()
    block_size = max(1, config.block_size_tokens)
    blocks: list[bytes] = []
    full_token_count = len(token_ids) - len(token_ids) % block_size
    for start in range(0, full_token_count, block_size):
        block = token_ids[start : start + block_size]
        blocks.append(_encode_token_block(block))
    return PrefixHashResult(
        config=config,
        block_hashes=_compute_chained_hashes(blocks, config),
        estimated_tokens=len(token_ids),
    )


def longest_prefix_match(request_hashes: Sequence[str], worker_hashes: Iterable[str]) -> int:
    """Return the number of contiguous request prefix blocks present on a worker."""
    worker_hash_set = set(worker_hashes)
    matched = 0
    for block_hash in request_hashes:
        if block_hash not in worker_hash_set:
            break
        matched += 1
    return matched


def prefix_hit_tokens(matched_blocks: int, block_size_tokens: int) -> int:
    """Convert matched prefix blocks to approximate matched prompt tokens."""
    return max(0, matched_blocks) * max(1, block_size_tokens)


def prefix_configs_match(left: PrefixHashConfig, right: PrefixHashConfig) -> bool:
    """Return whether two prefix-cache configs produce comparable hashes."""
    return left.comparable_key() == right.comparable_key()


def _compute_chained_hashes(blocks: Sequence[bytes], config: PrefixHashConfig) -> list[str]:
    parent = _root_hash(config)
    out: list[str] = []
    for block in blocks[: config.max_blocks]:
        digest = hashlib.sha256()
        digest.update(parent)
        digest.update(len(block).to_bytes(8, "big", signed=False))
        digest.update(block)
        parent = digest.digest()
        out.append(parent.hex())
    return out


def _root_hash(config: PrefixHashConfig) -> bytes:
    root_payload = {
        "hash_scheme": config.hash_scheme,
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id,
        "block_size_tokens": config.block_size_tokens,
        "bytes_per_token": config.bytes_per_token,
    }
    data = json.dumps(root_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).digest()


def _encode_token_block(token_ids: Sequence[int]) -> bytes:
    encoded = bytearray()
    for token_id in token_ids:
        encoded.extend(int(token_id).to_bytes(8, "big", signed=True))
    return bytes(encoded)


def _normalize_messages(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        normalized.append(
            {
                "role": role if isinstance(role, str) else "",
                "content": _normalize_content(content),
            }
        )
    return normalized


def _normalize_content(content: object) -> object:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized_blocks: list[object] = []
        for block in content:
            if isinstance(block, Mapping):
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    normalized_blocks.append({"type": "text", "text": block["text"]})
                else:
                    normalized_blocks.append(_json_safe_mapping(block))
            else:
                normalized_blocks.append(str(block))
        return normalized_blocks
    if content is None:
        return ""
    return str(content)


def _json_safe_mapping(value: Mapping[Any, Any]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, item in value.items():
        key_str = str(key)
        if isinstance(item, str | int | float | bool) or item is None:
            out[key_str] = item
        elif isinstance(item, Mapping):
            out[key_str] = _json_safe_mapping(item)
        elif isinstance(item, list):
            out[key_str] = [
                _json_safe_mapping(i) if isinstance(i, Mapping) else str(i) for i in item
            ]
        else:
            out[key_str] = str(item)
    return out
