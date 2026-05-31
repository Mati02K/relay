#!/usr/bin/env python3
"""Relay test suite entry point.

Usage
-----
  python test/run_tests.py --coordinator http://192.168.1.10:8080 --all
  python test/run_tests.py --coordinator http://192.168.1.10:8080 --signals
  python test/run_tests.py --coordinator http://192.168.1.10:8080 --scenarios memory thermal
  python test/run_tests.py --coordinator http://192.168.1.10:8080 --compare
  python test/run_tests.py --coordinator http://192.168.1.10:8080 --load
  python test/run_tests.py --coordinator http://192.168.1.10:8080 --load --shape spike

  # Prepare dataset before running (one-time, ~400 MB download):
  python test/run_tests.py --prepare-data

Flags
-----
  --all          Run signal tests + comparison + locust ramp/spike/sustained
  --signals      All scheduler signal scenario tests
  --compare      Relay vs round-robin vs single-worker
  --load         Locust load tests (ramp, spike, sustained — or specify --shape)
  --scenarios S  Comma-separated or space-separated list:
                 memory thermal jitter queue prefix_cache quality_routing
  --shape NAME   Locust shape to run: ramp | spike | sustained (default: ramp)
  --model M      Model name to pass in requests
  --workers N    Minimum number of healthy workers expected (default: 2)
  --verbose      Pass -v to pytest
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_DIR = Path(__file__).resolve().parent
_ALL_SIGNALS = ["memory", "thermal", "jitter", "queue", "prefix_cache", "quality_routing"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relay test suite runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--coordinator", default="http://localhost:8080",
                        help="Relay coordinator URL (default: http://localhost:8080)")
    parser.add_argument("--model", default=None,
                        help="Pin a specific model in requests (optional). Omit it to let any "
                             "worker serve them — each runs its own loaded model")
    parser.add_argument("--scenarios", nargs="+", choices=_ALL_SIGNALS, metavar="SCENARIO",
                        help="Which signal scenarios to run")
    parser.add_argument("--signals", action="store_true", help="Run all signal scenarios")
    parser.add_argument("--compare", action="store_true", help="Run baseline comparison")
    parser.add_argument("--load", action="store_true", help="Run locust load tests")
    parser.add_argument("--shape", choices=["ramp", "spike", "sustained", "all"],
                        default="ramp", help="Locust load shape (default: ramp)")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Download and prepare the ShareGPT dataset then exit")
    parser.add_argument("--workers", type=int, default=2,
                        help="Minimum expected healthy workers (default: 2)")
    parser.add_argument("--verbose", action="store_true", help="Verbose pytest output")
    args = parser.parse_args()

    # ── Dataset preparation ───────────────────────────────────────────────────
    if args.prepare_data:
        _prepare_dataset()
        return

    # ── Results directory ─────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = _TEST_DIR / "results" / ts
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    print(f"Results directory: {results_dir}")

    exit_codes: list[int] = []

    # ── Pytest signal + comparison tests ─────────────────────────────────────
    test_paths: list[str] = []

    run_all = args.all
    run_signals = run_all or args.signals or bool(args.scenarios)
    run_compare = run_all or args.compare

    if run_signals:
        if args.scenarios:
            for s in args.scenarios:
                test_paths.append(str(_TEST_DIR / "scenarios" / f"test_{s}.py"))
        else:
            test_paths.append(str(_TEST_DIR / "scenarios"))

    if run_compare:
        test_paths.append(str(_TEST_DIR / "comparison"))

    if test_paths:
        code = _run_pytest(args, results_dir, test_paths)
        exit_codes.append(code)

    # ── Locust load tests ─────────────────────────────────────────────────────
    if run_all or args.load:
        shapes = ["ramp", "spike", "sustained"] if args.shape == "all" else [args.shape]
        for shape in shapes:
            code = _run_locust(args.coordinator, results_dir, shape)
            exit_codes.append(code)

    # ── Default: if nothing selected, run signal tests ────────────────────────
    if not test_paths and not (run_all or args.load):
        print("No tests selected. Running all signal scenarios by default.")
        code = _run_pytest(args, results_dir, [str(_TEST_DIR / "scenarios")])
        exit_codes.append(code)

    # ── Final summary ─────────────────────────────────────────────────────────
    overall = max(exit_codes) if exit_codes else 0
    _write_run_manifest(results_dir, args, overall)
    print(f"\nAll results saved to: {results_dir}")
    print(f"Exit code: {overall}")
    sys.exit(overall)


def _run_pytest(
    args: argparse.Namespace, results_dir: Path, test_paths: list[str]
) -> int:
    cmd = [
        sys.executable, "-m", "pytest",
        f"--coordinator={args.coordinator}",
        f"--results-dir={results_dir}",
        "--tb=short",
        "--asyncio-mode=auto",
        # Stream prints live (per-phase progress) instead of capturing them; a
        # long run otherwise looks frozen until pytest finishes.
        "-s",
    ]
    if args.model:
        cmd.append(f"--model={args.model}")
    if args.verbose:
        cmd.append("-v")
    cmd += test_paths

    print(f"\nRunning pytest: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=_REPO_ROOT)
    return result.returncode


def _run_locust(coordinator_url: str, results_dir: Path, shape: str) -> int:
    csv_prefix = str(results_dir / f"locust_{shape}")
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(_TEST_DIR / "load" / "locustfile.py"),
        "--host", coordinator_url,
        "--headless",
        "--csv", csv_prefix,
    ]
    shape_params = {
        "ramp":      ["--run-time", "5m"],
        "spike":     ["--run-time", "3m"],
        "sustained": ["--run-time", "10m"],
    }
    cmd += shape_params.get(shape, ["--run-time", "5m"])

    import os
    env = {**os.environ, "RELAY_LOAD_SHAPE": shape}
    print(f"\nRunning locust ({shape}): {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=_REPO_ROOT, env=env)

    # Generate locust plots if CSV files were produced
    stats_file = Path(f"{csv_prefix}_stats.csv")
    if stats_file.exists():
        _plot_locust_results(stats_file, results_dir, shape)

    return result.returncode


def _plot_locust_results(stats_csv: Path, results_dir: Path, shape: str) -> None:
    """Generate a TTFT-over-time plot from locust stats CSV if pandas is available."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.read_csv(stats_csv)
        if "50%ile (ms)" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        for col, label in [("50%ile (ms)", "P50"), ("90%ile (ms)", "P90"), ("99%ile (ms)", "P99")]:
            if col in df.columns:
                ax.plot(df.index, df[col], label=label, linewidth=2)
        ax.set_xlabel("Time bucket")
        ax.set_ylabel("Response time (ms)")
        ax.set_title(f"Locust {shape.title()} — Latency over Time")
        ax.legend()
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        out = results_dir / "plots" / f"locust_{shape}_latency.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Locust plot saved: {out}")
    except ImportError:
        pass


def _prepare_dataset() -> None:
    sys.path.insert(0, str(_TEST_DIR))
    from framework.datasets import ShareGPTDataset

    data_dir = _TEST_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    ds = ShareGPTDataset(data_dir)
    ws = ds._load_working_set()
    print(
        f"\nDataset ready:\n"
        f"  short_prompts:  {ws['stats']['short_prompts']}\n"
        f"  mixed_prompts:  {ws['stats']['mixed_prompts']}\n"
        f"  conversations:  {ws['stats']['conversations']}\n"
        f"  Location: {data_dir / 'sharegpt_working.json'}"
    )


def _write_run_manifest(results_dir: Path, args: argparse.Namespace, exit_code: int) -> None:

    manifest = {
        "coordinator": args.coordinator,
        "model": args.model,
        "exit_code": exit_code,
        "results_dir": str(results_dir),
        "git_sha": _git_sha(),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
