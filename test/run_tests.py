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
  --chaos        Interactive worker-churn resilience test (manual kills; --rps/--duration)
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
    parser.add_argument("--chaos", action="store_true",
                        help="Run the interactive worker-churn resilience test "
                             "(constant load, manual kills, cost vs round-robin)")
    parser.add_argument("--rps", type=float, default=5.0,
                        help="Chaos: constant arrival rate in req/s (default: 5)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Chaos: seconds per mode run (default: 300)")
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

    # ── Chaos / worker-churn resilience test (interactive; not part of --all) ──
    if args.chaos:
        code = _run_chaos(args, results_dir)
        exit_codes.append(code)

    # ── Default: if nothing selected, run signal tests ────────────────────────
    if not test_paths and not (run_all or args.load or args.chaos):
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


def _run_chaos(args: argparse.Namespace, results_dir: Path) -> int:
    """Run the worker-churn chaos test, passing rps/duration through the environment."""
    import os

    cmd = [
        sys.executable, "-m", "pytest",
        f"--coordinator={args.coordinator}",
        f"--results-dir={results_dir}",
        "--tb=short",
        "--asyncio-mode=auto",
        "-s",
        str(_TEST_DIR / "chaos"),
    ]
    if args.model:
        cmd.append(f"--model={args.model}")
    if args.verbose:
        cmd.append("-v")

    env = {
        **os.environ,
        "RELAY_CHAOS_RPS": str(args.rps),
        "RELAY_CHAOS_DURATION": str(args.duration),
    }
    print(f"\nRunning chaos (worker churn): {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=_REPO_ROOT, env=env)
    return result.returncode


def _run_locust(coordinator_url: str, results_dir: Path, shape: str) -> int:
    csv_prefix = str(results_dir / f"locust_{shape}")
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(_TEST_DIR / "load" / "locustfile.py"),
        "--host", coordinator_url,
        "--headless",
        "--csv", csv_prefix,
        # Per-endpoint rows per interval in _stats_history.csv (not just the
        # aggregate) so latency and TTFT can be plotted separately over time.
        "--csv-full-history",
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

    # Plot latency + throughput over time from the stats-history CSV.
    history_file = Path(f"{csv_prefix}_stats_history.csv")
    if history_file.exists():
        _plot_locust_results(history_file, results_dir, shape)

    return result.returncode


def _plot_locust_results(history_csv: Path, results_dir: Path, shape: str) -> None:
    """Emit three separate plots from a locust stats-history CSV (over time).

    1. ``locust_<shape>_latency.png``    — full request P50/P90/P99 (the POST)
    2. ``locust_<shape>_ttft.png``       — TTFT P90/P95/P99 (the SSE ``ttft`` event)
    3. ``locust_<shape>_throughput.png`` — requests/sec + user count

    The history file has one row per ~2s tick per endpoint (``/v1/chat/completions``,
    ``ttft``) plus an ``Aggregated`` row. No-ops if matplotlib/pandas or the needed
    columns are absent rather than failing the run.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    df = pd.read_csv(history_csv)
    if "Timestamp" not in df.columns or "Name" not in df.columns:
        return
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _rows(name: str):  # type: ignore[no-untyped-def]
        sub = df[df["Name"] == name].copy()
        if sub.empty:
            return None
        ts = pd.to_numeric(sub["Timestamp"], errors="coerce")
        sub["_t"] = ts - ts.iloc[0]
        return sub

    def _pct_plot(names, pcts, title, fname, ylabel):  # type: ignore[no-untyped-def]
        sub = next((r for r in (_rows(n) for n in names) if r is not None), None)
        if sub is None:
            return
        cols = [(c, lbl) for c, lbl in pcts if c in sub.columns]
        if not cols:
            return
        fig, ax = plt.subplots(figsize=(11, 5))
        for col, lbl in cols:
            ax.plot(sub["_t"], pd.to_numeric(sub[col], errors="coerce"), label=lbl, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Percentile")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        out = plots_dir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  plot: {out}")

    title = f"Locust {shape.title()}"
    _pct_plot(("/v1/chat/completions", "Aggregated"),
              [("50%", "P50"), ("90%", "P90"), ("99%", "P99")],
              f"{title} — Request latency over time", f"locust_{shape}_latency.png",
              "Response time (ms)")
    _pct_plot(("ttft",), [("90%", "P90"), ("95%", "P95"), ("99%", "P99")],
              f"{title} — TTFT over time", f"locust_{shape}_ttft.png", "TTFT (ms)")

    # Throughput from the POST rows (the SSE event would double-count requests).
    sub = _rows("/v1/chat/completions")
    if sub is None:
        sub = _rows("Aggregated")
    rps_col = next((c for c in ("Requests/s", "Total Requests/s")
                    if sub is not None and c in sub.columns), None)
    if sub is not None and rps_col:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(sub["_t"], pd.to_numeric(sub[rps_col], errors="coerce"),
                color="#388E3C", linewidth=2, label="Requests/s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Requests/s")
        if "User Count" in sub.columns:
            ax2 = ax.twinx()
            ax2.plot(sub["_t"], pd.to_numeric(sub["User Count"], errors="coerce"),
                     color="gray", linestyle="--", alpha=0.6, label="Users")
            ax2.set_ylabel("Users", color="gray")
        ax.set_title(f"{title} — Throughput over time")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper left")
        fig.tight_layout()
        out = plots_dir / f"locust_{shape}_throughput.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  plot: {out}")


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
