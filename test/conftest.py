"""Pytest configuration and shared fixtures for the Relay test suite.

All fixtures are session-scoped to avoid repeated dataset loading and cluster
setup overhead.  Individual scenario tests can override per-test with function
scope if isolation is required.

Usage
-----
  pytest test/scenarios/test_memory.py --coordinator http://192.168.1.10:8080
  pytest test/ --coordinator http://192.168.1.10:8080 --results-dir /tmp/relay_run
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

# Make `framework` importable from within test files
_TEST_DIR = Path(__file__).parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from framework.client import RelayClient
from framework.cluster import ClusterClient
from framework.datasets import ShareGPTDataset


# ── CLI options ───────────────────────────────────────────────────────────────

def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--coordinator",
        default="http://localhost:8080",
        help="Relay coordinator base URL (default: http://localhost:8080)",
    )
    parser.addoption(
        "--results-dir",
        default=None,
        help="Directory to store run results. Defaults to test/results/<timestamp>/",
    )
    parser.addoption(
        "--model",
        default=None,
        help="Model name to pass in chat completion requests (optional)",
    )


# ── Core fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def coordinator_url(request: Any) -> str:
    return request.config.getoption("--coordinator")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def model_name(request: Any) -> str | None:
    return request.config.getoption("--model")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def run_dir(request: Any, tmp_path_factory: Any) -> Path:
    """Create and return the results directory for this run."""
    given = request.config.getoption("--results-dir")
    if given:
        d = Path(given)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = _TEST_DIR / "results" / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(exist_ok=True)
    return d


@pytest.fixture(scope="session")
def data_dir() -> Path:
    d = _TEST_DIR / "data"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture(scope="session")
def cluster(coordinator_url: str) -> ClusterClient:
    return ClusterClient(coordinator_url)


@pytest.fixture(scope="session")
def relay_client(coordinator_url: str, model_name: str | None) -> RelayClient:
    client = RelayClient(coordinator_url)
    if model_name:
        # Monkey-patch default model into the client for convenience
        _orig = client.chat_completion

        async def _with_model(*args: Any, **kwargs: Any) -> Any:
            if "model" not in kwargs or kwargs["model"] is None:
                kwargs["model"] = model_name
            return await _orig(*args, **kwargs)

        client.chat_completion = _with_model  # type: ignore[method-assign]
    return client


# ── Dataset fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dataset(data_dir: Path) -> ShareGPTDataset:
    return ShareGPTDataset(data_dir)


@pytest.fixture(scope="session")
def short_prompts(dataset: ShareGPTDataset) -> list[list[dict[str, str]]]:
    """600 short single-turn prompts (~50–300 tokens) for signal phase tests."""
    return dataset.short_prompts()


@pytest.fixture(scope="session")
def mixed_prompts(dataset: ShareGPTDataset) -> list[list[dict[str, str]]]:
    """1000 mixed-length single-turn prompts for load and comparison tests."""
    return dataset.mixed_prompts()


@pytest.fixture(scope="session")
def conversations(dataset: ShareGPTDataset) -> list[list[dict[str, str]]]:
    """200 multi-turn conversations (5–20 turns) for prefix-cache tests."""
    return dataset.conversations()


# ── Git metadata ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_TEST_DIR.parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"
