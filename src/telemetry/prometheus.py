"""Small Prometheus text-format helpers used by inference engine adapters."""

from __future__ import annotations

from collections.abc import Mapping


def parse_prometheus_samples(body: str) -> dict[str, float]:
    """Parse Prometheus text samples into metric-name to value mappings."""
    result: dict[str, float] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric, value_s = parts
        try:
            value = float(value_s)
        except ValueError:
            continue
        base = metric.split("{", 1)[0]
        result[base] = value
        result[metric] = value
    return result


def find_metric(samples: Mapping[str, float], *candidates: str) -> float | None:
    """Return the first available metric from alternate backend naming styles."""
    for name in candidates:
        if name in samples:
            return float(samples[name])
    stripped = {k.split("{", 1)[0]: v for k, v in samples.items()}
    for name in candidates:
        if name in stripped:
            return float(stripped[name])
    tail_by_candidate = [c.split(":", 1)[-1].replace(":", "_") for c in candidates]
    for key, val in stripped.items():
        for tail in tail_by_candidate:
            if key == tail or key.endswith(tail):
                return float(val)
    return None
