"""Coordinator-side model chart.

The chart is the authority on what each model can do (``skills``), what
input modalities it accepts (``input_modalities``), and how strong it is
(``quality`` in ``[0, 1]``). Workers no longer self-advertise these
fields — they only publish which model id they have loaded. The
coordinator looks the rest up here.

Entries are hand-tuned for the demo set. Unknown ids fall through to
:func:`_derive_default_entry`, which infers skills from id keywords and
estimates quality from parameter count and quantization.

Operators can override the built-in chart by pointing
``RELAY_MODEL_CHART`` at a JSON file shaped like the :data:`_DEFAULT_CHART`
mapping. The override is loaded once at first lookup.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_QUALITY = 0.4
DEFAULT_SKILLS: frozenset[str] = frozenset({"instruct"})
DEFAULT_INPUT_MODALITIES: frozenset[str] = frozenset({"text"})


@dataclass(frozen=True)
class ModelEntry:
    """Chart row for one model id."""

    model_id: str
    skills: frozenset[str]
    quality: float
    input_modalities: frozenset[str] = DEFAULT_INPUT_MODALITIES
    params_b: float | None = None
    quant: str | None = None
    source: str = "chart"


_DEFAULT_CHART: dict[str, dict[str, Any]] = {
    "qwen2.5-coder-7b": {
        "skills": ["coding", "instruct"],
        "params_b": 7.6,
        "quant": "q4_k_m",
        "quality": 0.78,
    },
    "qwen2.5-coder-3b": {
        "skills": ["coding", "instruct"],
        "params_b": 3.1,
        "quant": "q4_k_m",
        "quality": 0.62,
    },
    "qwen2.5-coder-1.5b": {
        "skills": ["coding", "instruct"],
        "params_b": 1.5,
        "quant": "q4_k_m",
        "quality": 0.48,
    },
    "qwen2.5-7b": {
        "skills": ["instruct", "chat", "reasoning"],
        "params_b": 7.6,
        "quant": "q4_k_m",
        "quality": 0.80,
    },
    "qwen2.5-3b": {
        "skills": ["instruct", "chat"],
        "params_b": 3.1,
        "quant": "q4_k_m",
        "quality": 0.58,
    },
    "qwen2.5-1.5b": {
        "skills": ["instruct", "chat"],
        "params_b": 1.5,
        "quant": "q4_k_m",
        "quality": 0.45,
    },
    "qwen2.5-0.5b": {
        "skills": ["instruct", "chat"],
        "params_b": 0.5,
        "quant": "q4_k_m",
        "quality": 0.28,
    },
    "llama-3.2-3b": {
        "skills": ["instruct", "chat"],
        "params_b": 3.2,
        "quant": "q4_k_m",
        "quality": 0.55,
    },
    "llama-3.2-1b": {
        "skills": ["instruct", "chat"],
        "params_b": 1.2,
        "quant": "q4_k_m",
        "quality": 0.38,
    },
    "phi-3.5-mini": {
        "skills": ["instruct", "chat", "reasoning"],
        "params_b": 3.8,
        "quant": "q4_k_m",
        "quality": 0.60,
    },
    "mistral-7b": {
        "skills": ["instruct", "chat"],
        "params_b": 7.2,
        "quant": "q4_k_m",
        "quality": 0.72,
    },
    "qwen-moe-a2.7b": {
        "skills": ["instruct", "chat"],
        "params_b": 2.7,
        "quant": "q4_k_m",
        "quality": 0.56,
    },
    # MLX variants — same quality numbers as their non-MLX twins so a
    # request routed to a Mac running MLX is scored fairly against a
    # llama.cpp box running the equivalent GGUF.
    "qwen2.5-7b-mlx": {
        "skills": ["instruct", "chat", "reasoning"],
        "params_b": 7.6,
        "quant": "4bit",
        "quality": 0.80,
    },
    "qwen2.5-1.5b-mlx": {
        "skills": ["instruct", "chat"],
        "params_b": 1.5,
        "quant": "4bit",
        "quality": 0.45,
    },
    "qwen2.5-0.5b-mlx": {
        "skills": ["instruct", "chat"],
        "params_b": 0.5,
        "quant": "4bit",
        "quality": 0.28,
    },
    "llama-3.2-1b-mlx": {
        "skills": ["instruct", "chat"],
        "params_b": 1.2,
        "quant": "4bit",
        "quality": 0.38,
    },
}


# Quantization quality multipliers used by :func:`_derive_default_entry`.
# Numbers are heuristic but monotone with bit-depth; demo-grade.
_QUANT_QUALITY_FACTOR: dict[str, float] = {
    "fp16": 1.00,
    "f16": 1.00,
    "bf16": 1.00,
    "q8_0": 0.97,
    "q6_k": 0.94,
    "q5_k_m": 0.92,
    "q5_k_s": 0.90,
    "q5_0": 0.89,
    "q4_k_m": 0.86,
    "q4_k_s": 0.84,
    "q4_0": 0.82,
    "iq4_xs": 0.80,
    "q3_k_m": 0.74,
    "q3_k_s": 0.70,
    "q2_k": 0.58,
}

_QUANT_PATTERN = re.compile(
    r"(fp16|bf16|f16|q8_0|q6_k|q5_k_m|q5_k_s|q5_0|q4_k_m|q4_k_s|q4_0|iq4_xs|q3_k_m|q3_k_s|q2_k)",
    re.IGNORECASE,
)

_PARAMS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

_CODING_KEYWORDS = ("coder", "code", "starcoder", "codellama", "deepseek-coder")
_REASONING_KEYWORDS = ("reason", "o1", "r1", "qwq")


@dataclass
class _ChartCache:
    """One-shot cache for the merged built-in + override chart."""

    entries: dict[str, ModelEntry] = field(default_factory=dict)
    loaded: bool = False


_CACHE = _ChartCache()


def lookup(model_id: str | None, *, quant_hint: str | None = None) -> ModelEntry:
    """Return the chart entry for ``model_id``.

    Falls back to a heuristic entry when the id is unknown. ``quant_hint``
    refines that heuristic when the caller knows the worker's quantization
    (e.g. from ``ModelConfig.quant``) but the id alone is ambiguous.
    """
    _ensure_loaded()
    if not model_id:
        return _unknown_entry("", quant_hint)
    key = model_id.lower()
    entry = _CACHE.entries.get(key)
    if entry is not None:
        return entry
    return _derive_default_entry(model_id, quant_hint)


def reload_chart() -> None:
    """Force-rebuild the chart cache. Mainly for tests."""
    _CACHE.entries.clear()
    _CACHE.loaded = False
    _ensure_loaded()


def _ensure_loaded() -> None:
    if _CACHE.loaded:
        return
    merged: dict[str, dict[str, Any]] = {key: dict(value) for key, value in _DEFAULT_CHART.items()}
    override_path = os.getenv("RELAY_MODEL_CHART")
    if override_path:
        overrides = _load_override(Path(override_path))
        if overrides:
            for key, value in overrides.items():
                merged[key.lower()] = value
    for model_id, raw in merged.items():
        entry = _entry_from_dict(model_id, raw)
        if entry is not None:
            _CACHE.entries[model_id.lower()] = entry
    _CACHE.loaded = True


def _load_override(path: Path) -> dict[str, dict[str, Any]] | None:
    try:
        text = path.read_text()
    except OSError as exc:
        logger.warning("model chart override unreadable | path={} error={}", path, exc)
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("model chart override invalid JSON | path={} error={}", path, exc)
        return None
    if not isinstance(obj, dict):
        logger.warning("model chart override must be a JSON object | path={}", path)
        return None
    cleaned: dict[str, dict[str, Any]] = {}
    for key, value in obj.items():
        if isinstance(key, str) and isinstance(value, dict):
            cleaned[key] = value
    return cleaned


def _entry_from_dict(model_id: str, raw: dict[str, Any]) -> ModelEntry | None:
    try:
        skills = _normalise_skills(raw.get("skills"))
        input_modalities = _normalise_modalities(raw.get("input_modalities"))
        quality = _clamp_quality(raw.get("quality"))
        params_b = _coerce_float(raw.get("params_b"))
        quant_value = raw.get("quant")
        quant = quant_value.lower() if isinstance(quant_value, str) else None
    except (TypeError, ValueError) as exc:
        logger.warning("invalid chart entry | id={} error={}", model_id, exc)
        return None
    return ModelEntry(
        model_id=model_id,
        skills=skills or DEFAULT_SKILLS,
        quality=quality,
        input_modalities=input_modalities or DEFAULT_INPUT_MODALITIES,
        params_b=params_b,
        quant=quant,
        source="chart",
    )


def _derive_default_entry(model_id: str, quant_hint: str | None) -> ModelEntry:
    """Derive a best-effort entry for an id not in the chart."""
    lowered = model_id.lower()
    skills: set[str] = {"instruct"}
    if any(kw in lowered for kw in _CODING_KEYWORDS):
        skills.add("coding")
    if any(kw in lowered for kw in _REASONING_KEYWORDS):
        skills.add("reasoning")
    if "chat" in lowered or "instruct" in lowered:
        skills.add("chat")

    params_b = _extract_params_from_id(lowered)
    quant = _extract_quant_from_id(lowered) or (quant_hint.lower() if quant_hint else None)
    quality = _estimate_quality(params_b, quant)
    return ModelEntry(
        model_id=model_id,
        skills=frozenset(skills),
        quality=quality,
        input_modalities=DEFAULT_INPUT_MODALITIES,
        params_b=params_b,
        quant=quant,
        source="derived",
    )


def _unknown_entry(model_id: str, quant_hint: str | None) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        skills=DEFAULT_SKILLS,
        quality=_estimate_quality(None, quant_hint),
        input_modalities=DEFAULT_INPUT_MODALITIES,
        params_b=None,
        quant=quant_hint.lower() if quant_hint else None,
        source="unknown",
    )


def _extract_params_from_id(model_id: str) -> float | None:
    match = _PARAMS_PATTERN.search(model_id)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_quant_from_id(model_id: str) -> str | None:
    match = _QUANT_PATTERN.search(model_id)
    return match.group(1).lower() if match else None


def _estimate_quality(params_b: float | None, quant: str | None) -> float:
    """Heuristic quality from size + quant. Monotone, bounded in [0, 1]."""
    if params_b is None or params_b <= 0.0:
        base = DEFAULT_QUALITY
    else:
        from math import log2

        base = min(1.0, log2(params_b + 1.0) / log2(73.0))
    factor = _QUANT_QUALITY_FACTOR.get((quant or "").lower(), 0.85)
    return max(0.0, min(1.0, base * factor))


def _normalise_skills(raw: Any) -> frozenset[str]:
    if not isinstance(raw, list):
        return frozenset()
    cleaned = {str(item).strip().lower() for item in raw if isinstance(item, str) and item.strip()}
    cleaned.discard("")
    return frozenset(cleaned)


def _normalise_modalities(raw: Any) -> frozenset[str]:
    if raw is None:
        return DEFAULT_INPUT_MODALITIES
    if not isinstance(raw, list):
        return frozenset()
    cleaned = {str(item).strip().lower() for item in raw if isinstance(item, str) and item.strip()}
    cleaned.discard("")
    return frozenset(cleaned)


def _clamp_quality(raw: Any) -> float:
    if raw is None:
        return DEFAULT_QUALITY
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_QUALITY
    return max(0.0, min(1.0, value))


def _coerce_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
