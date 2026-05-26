"""Relay command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from relay.config import ConfigError, load_config, save_config
from relay.init import InitOptions, run_init
from relay.models import catalog_rows, pull_catalog_model, register_local_model
from relay.paths import RelayPaths
from relay.software import doctor_checks
from relay.supervisor import (
    SupervisorError,
    health_summary,
    process_status_json,
    read_log_tail,
    start,
    status,
    stop,
)
from relay.ui import accent, banner, err, muted, name_column, ok, status_label, warn


def main(argv: list[str] | None = None) -> int:
    """Run the Relay CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, SupervisorError, RuntimeError) as e:
        print(f"{err('error:')} {e}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relay", description="Relay local cluster runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="create local Relay config")
    init_parser.add_argument("--role", choices=("coordinator", "worker", "dual"))
    init_parser.add_argument("--network", choices=("tailscale", "lan"))
    init_parser.add_argument("--coordinator", help="coordinator URL/IP for worker nodes")
    init_parser.add_argument("--node-id")
    init_parser.add_argument("--host", help="bind/advertise host for this node")
    init_parser.add_argument("--model", help="catalog model id to pull")
    init_parser.add_argument("--model-path", help="local GGUF model path")
    init_parser.add_argument("--skip-model", action="store_true")
    init_parser.set_defaults(func=_cmd_init)

    start_parser = sub.add_parser("start", help="start configured Relay processes")
    start_parser.add_argument("--foreground", action="store_true")
    start_parser.set_defaults(func=_cmd_start)

    stop_parser = sub.add_parser("stop", help="stop Relay processes")
    stop_parser.set_defaults(func=_cmd_stop)

    restart_parser = sub.add_parser(
        "restart",
        help="stop and then start Relay processes",
    )
    restart_parser.add_argument("--foreground", action="store_true")
    restart_parser.set_defaults(func=_cmd_restart)

    status_parser = sub.add_parser("status", help="show process and HTTP health")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=_cmd_status)

    logs_parser = sub.add_parser("logs", help="show process log tail")
    logs_parser.add_argument("process", nargs="?", default="worker")
    logs_parser.add_argument("--lines", type=int, default=80)
    logs_parser.set_defaults(func=_cmd_logs)

    doctor_parser = sub.add_parser("doctor", help="check local Relay runtime prerequisites")
    doctor_parser.set_defaults(func=_cmd_doctor)

    pull_parser = sub.add_parser("pull", help="download/register a model")
    pull_parser.add_argument("model", nargs="?", help="catalog model id")
    pull_parser.add_argument("--local", help="register a local GGUF model path")
    pull_parser.add_argument("--id", help="model id for --local")
    pull_parser.set_defaults(func=_cmd_pull)

    models_parser = sub.add_parser("models", help="model commands")
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="list configured and catalog models")
    models_list.add_argument("--catalog", action="store_true")
    models_list.set_defaults(func=_cmd_models_list)

    dashboard_parser = sub.add_parser(
        "dashboard",
        help="launch the testing dashboard (chat UI + worker telemetry)",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RELAY_DASHBOARD_PORT", "8090")),
        help="HTTP port to bind (default 8090, env RELAY_DASHBOARD_PORT)",
    )
    dashboard_parser.add_argument(
        "--host",
        default=os.getenv("RELAY_DASHBOARD_HOST", "127.0.0.1"),
        help="HTTP host to bind (default 127.0.0.1)",
    )
    dashboard_parser.add_argument(
        "--coordinator",
        help="coordinator base URL (default: from config or http://127.0.0.1:8080)",
    )
    dashboard_parser.add_argument(
        "--no-open",
        action="store_true",
        help="don't open a browser automatically",
    )
    dashboard_parser.set_defaults(func=_cmd_dashboard)

    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    print(banner(), end="")
    config = run_init(
        InitOptions(
            role=args.role,
            network=args.network,
            coordinator=args.coordinator,
            node_id=args.node_id,
            host=args.host,
            model=args.model,
            model_path=args.model_path,
            skip_model=bool(args.skip_model),
        )
    )
    print(f"{ok('config:')} {RelayPaths.from_home().config}")
    print(f"{accent('role:')}   {config.role}")
    if config.runs_worker and not config.models:
        print(
            warn("note:"),
            "no model configured. Run `relay pull <model>` or register a local GGUF.",
        )
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    config = load_config()
    print(banner(), end="")
    statuses = start(config, foreground=bool(args.foreground))
    _print_start_or_stop(statuses)
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError:
        config = None
    _print_start_or_stop(stop(config))
    return 0


def _cmd_restart(args: argparse.Namespace) -> int:
    config = load_config()
    print(banner(), end="")
    print(accent("stopping...\n"))
    _print_start_or_stop(stop(config))
    print()
    print(accent("starting...\n"))
    _print_start_or_stop(start(config, foreground=bool(args.foreground)))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError:
        config = None
    statuses = status(config)
    if args.json:
        print(process_status_json(statuses))
    else:
        print()
        for item in statuses:
            pid_text = str(item.pid) if item.pid is not None else "-"
            state = status_label(item.running, item.detail)
            # Pad the plain state text first, then optionally colour; the
            # ANSI escapes don't count toward visible width.
            print(f"  {name_column(item.name)} {state.ljust(20)} {muted('pid=' + pid_text)}")
        print()
    if config is not None and any(item.running for item in statuses):
        health = health_summary(config)
        if health:
            print(json.dumps(health, indent=2))
    return 0


def _print_start_or_stop(statuses: list) -> None:  # type: ignore[type-arg]
    """Render a list of ProcessStatus rows with consistent column alignment."""
    print()
    for item in statuses:
        pid = muted(f"pid={item.pid}") if item.pid else muted("")
        detail = item.detail.lower()
        if "no pid" in detail or "not running" in detail or "stopped" in detail:
            label = muted(item.detail)
        else:
            label = ok(item.detail)
        print(f"  {name_column(item.name)} {label}  {pid}")
    print()


def _cmd_logs(args: argparse.Namespace) -> int:
    print(read_log_tail(str(args.process), lines=int(args.lines)))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError:
        config = None
    checks = doctor_checks(config)
    failed = False
    for check in checks:
        state = "ok" if check.ok else "missing"
        print(f"{check.name:18} {state:8} {check.detail}")
        failed = failed or not check.ok
    return 1 if failed else 0


def _cmd_pull(args: argparse.Namespace) -> int:
    config = load_config()
    if args.local:
        model_id = args.id or Path(args.local).stem
        config = register_local_model(config, model_id, Path(args.local))
        save_config(config)
        print(f"Registered local model {model_id}: {args.local}")
        return 0
    if not args.model:
        print("Available models:")
        for row in catalog_rows():
            print("  " + row)
        return 0
    config, model = pull_catalog_model(config, str(args.model))
    save_config(config)
    print(f"Downloaded {model.id}: {model.path}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from dashboard.server import PortInUseError, run_dashboard

    coordinator_url = args.coordinator
    if not coordinator_url:
        try:
            config = load_config()
            coordinator_url = config.coordinator.url
        except ConfigError:
            coordinator_url = "http://127.0.0.1:8080"

    try:
        run_dashboard(
            host=str(args.host),
            port=int(args.port),
            coordinator_url=str(coordinator_url),
            open_browser=not bool(args.no_open),
        )
    except PortInUseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    if args.catalog:
        for row in catalog_rows():
            print(row)
        return 0
    config = load_config()
    if not config.models:
        print("No local models configured.")
        return 0
    for model in config.models:
        loaded = "loaded" if model.loaded else "available"
        print(f"{model.id:16} {loaded:9} {model.engine:9} {model.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
