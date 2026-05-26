"""Local process supervisor used by ``relay start`` and ``relay stop``."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from relay.config import RelayConfig, model_inventory_json
from relay.paths import RelayPaths
from relay.software import (
    ensure_etcd,
    ensure_llama_server,
    ensure_membership_etcd,
    resolve_executable,
)


@dataclass(frozen=True)
class ManagedProcess:
    """A process controlled by the Relay CLI."""

    name: str
    args: list[str]
    env: dict[str, str]
    log_path: Path
    pid_path: Path


@dataclass(frozen=True)
class ProcessStatus:
    """Runtime status for a managed process."""

    name: str
    running: bool
    pid: int | None
    detail: str


class SupervisorError(RuntimeError):
    """Raised when process startup or shutdown fails."""


def start(config: RelayConfig, *, foreground: bool = False) -> list[ProcessStatus]:
    """Start all processes required by this node role."""
    paths = RelayPaths.from_home()
    paths.ensure()
    specs = build_process_specs(config, paths)
    statuses: list[ProcessStatus] = []
    for spec in specs:
        existing = _read_pid(spec.pid_path)
        if existing is not None and _pid_matches_spec(existing, spec):
            statuses.append(ProcessStatus(spec.name, True, existing, "already running"))
            continue
        if existing is not None:
            spec.pid_path.unlink(missing_ok=True)
        if foreground:
            _run_foreground(spec)
            statuses.append(ProcessStatus(spec.name, True, None, "foreground exited"))
            continue
        pid = _start_background(spec)
        statuses.append(ProcessStatus(spec.name, True, pid, f"log={spec.log_path}"))
        if spec.name == "etcd":
            _wait_for_tcp("127.0.0.1", int(spec.env["ETCD_CLIENT_PORT"]), "etcd")
        elif spec.name == "membership-etcd":
            _wait_for_tcp("127.0.0.1", int(spec.env["GRPC_PORT"]), "membership-etcd")
        time.sleep(0.2)
    return statuses


def stop(config: RelayConfig | None = None) -> list[ProcessStatus]:
    """Stop managed processes for the config, or all known Relay process pid files."""
    paths = RelayPaths.from_home()
    pid_paths = _managed_pid_paths(config, paths)
    statuses: list[ProcessStatus] = []
    for pid_path in reversed(pid_paths):
        name = pid_path.stem
        pid = _read_pid(pid_path)
        if pid is None:
            statuses.append(ProcessStatus(name, False, None, "no pid file"))
            continue
        if not _pid_running(pid):
            pid_path.unlink(missing_ok=True)
            statuses.append(ProcessStatus(name, False, pid, "not running"))
            continue
        _signal_group(pid, signal.SIGTERM)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _pid_running(pid):
            time.sleep(0.2)
        if _pid_running(pid):
            _signal_group(pid, signal.SIGKILL)
        pid_path.unlink(missing_ok=True)
        statuses.append(ProcessStatus(name, False, pid, "stopped"))
    return statuses


def status(config: RelayConfig | None = None) -> list[ProcessStatus]:
    """Return process status for configured or known Relay processes."""
    paths = RelayPaths.from_home()
    pid_paths = _managed_pid_paths(config, paths)
    statuses: list[ProcessStatus] = []
    for pid_path in pid_paths:
        pid = _read_pid(pid_path)
        running = pid is not None and _pid_running(pid)
        statuses.append(
            ProcessStatus(
                name=pid_path.stem,
                running=running,
                pid=pid,
                detail="running" if running else "stopped",
            )
        )
    return statuses


def _managed_pid_paths(config: RelayConfig | None, paths: RelayPaths) -> list[Path]:
    if config is None:
        return sorted(paths.run.glob("*.pid"))
    names: list[str] = []
    if config.runs_coordinator:
        names.extend(["etcd", "membership-etcd"])
        names.append("coordinator")
    if config.runs_worker:
        names.append("worker")
    # Dashboard is launched ad-hoc via `relay dashboard`, not by `relay start`,
    # so include its pid file only when one is actually present.
    if (paths.run / "dashboard.pid").exists():
        names.append("dashboard")
    return [paths.run / f"{name}.pid" for name in names]


def health_summary(config: RelayConfig) -> dict[str, Any]:
    """Return HTTP health responses for local coordinator/worker endpoints."""
    result: dict[str, Any] = {}
    with httpx.Client(timeout=2.0) as client:
        if config.runs_coordinator:
            result["coordinator"] = _health_get(client, f"{config.coordinator.url}/health")
        if config.runs_worker:
            result["worker"] = _health_get(
                client,
                f"http://{config.worker.host}:{config.worker.port}/health",
            )
    return result


def build_process_specs(config: RelayConfig | None, paths: RelayPaths) -> list[ManagedProcess]:
    """Build role-aware process specs."""
    if config is None:
        return []

    specs: list[ManagedProcess] = []
    common_env = os.environ.copy()
    common_env["PYTHONUNBUFFERED"] = "1"
    common_env["NODE_ID"] = config.node_id
    common_env["MEMBERSHIP_HOST"] = config.membership.host
    common_env["MEMBERSHIP_PORT"] = str(config.membership.grpc_port)
    common_env["LOG_FILE"] = str(paths.logs / "relay.log")

    if config.runs_coordinator:
        specs.extend(_membership_process_specs(config, paths, common_env))

        coordinator_env = common_env.copy()
        coordinator_env["COORDINATOR_HOST"] = config.coordinator.host
        coordinator_env["COORDINATOR_PORT"] = str(config.coordinator.port)
        # Scheduler weights are read from env at scheduler.py import time
        # (see RELAY_SCHED_*_WEIGHT constants there). Pipe the config values
        # through so the persisted choice from `relay init` actually takes
        # effect on the coordinator process.
        coordinator_env["RELAY_SCHED_QUEUE_WEIGHT"] = str(config.scheduler.queue)
        coordinator_env["RELAY_SCHED_PREFIX_MISS_WEIGHT"] = str(config.scheduler.prefix_miss)
        coordinator_env["RELAY_SCHED_MEMORY_WEIGHT"] = str(config.scheduler.memory)
        coordinator_env["RELAY_SCHED_JITTER_WEIGHT"] = str(config.scheduler.jitter)
        coordinator_env["RELAY_SCHED_THERMAL_WEIGHT"] = str(config.scheduler.thermal)
        specs.append(
            ManagedProcess(
                name="coordinator",
                args=[
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "coordinator.main:app",
                    "--host",
                    config.coordinator.host,
                    "--port",
                    str(config.coordinator.port),
                ],
                env=coordinator_env,
                log_path=paths.logs / "coordinator.log",
                pid_path=paths.run / "coordinator.pid",
            )
        )

    if config.runs_worker:
        model = config.active_model()
        if model is None:
            raise SupervisorError(
                "No model configured. Run `relay pull` or `relay init` with a model."
            )
        worker_env = common_env.copy()
        worker_env["WORKER_HOST"] = config.worker.host
        worker_env["WORKER_PORT"] = str(config.worker.port)
        worker_env["RELAY_MODEL_ID"] = model.id
        worker_env["LLAMA_MODEL_ID"] = model.id
        worker_env["LLAMA_MODEL_PATH"] = model.path
        worker_env["LLAMA_SERVER_BIN"] = _llama_server_binary(config.engine.server_bin)
        worker_env["LLAMA_SERVER_HOST"] = config.engine.server_host
        worker_env["LLAMA_SERVER_PORT"] = str(config.engine.server_port)
        worker_env["LLAMA_SERVER_EXTRA_ARGS"] = config.engine.extra_args
        worker_env["RELAY_MODEL_INVENTORY_JSON"] = model_inventory_json(config)
        specs.append(
            ManagedProcess(
                name="worker",
                args=[
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "worker.main:app",
                    "--host",
                    config.worker.host,
                    "--port",
                    str(config.worker.port),
                ],
                env=worker_env,
                log_path=paths.logs / "worker.log",
                pid_path=paths.run / "worker.pid",
            )
        )
    return specs


def _membership_process_specs(
    config: RelayConfig,
    paths: RelayPaths,
    common_env: dict[str, str],
) -> list[ManagedProcess]:
    etcd = resolve_executable(config.membership.etcd_bin, "etcd") or ensure_etcd()
    etcd_env = common_env.copy()
    etcd_env["ETCD_CLIENT_PORT"] = str(config.membership.etcd_client_port)
    data_dir = config.membership.etcd_data_dir or str(paths.etcd_data)
    membership = (
        resolve_executable(config.membership.service_bin, "membership-etcd")
        or ensure_membership_etcd()
    )
    membership_env = common_env.copy()
    membership_env["ETCD_ENDPOINTS"] = config.membership.etcd_endpoints
    membership_env["GRPC_PORT"] = str(config.membership.grpc_port)
    return [
        ManagedProcess(
            name="etcd",
            args=[
                etcd,
                "--name",
                config.node_id,
                "--data-dir",
                data_dir,
                "--listen-client-urls",
                f"http://0.0.0.0:{config.membership.etcd_client_port}",
                "--advertise-client-urls",
                f"http://{config.membership.host}:{config.membership.etcd_client_port}",
                "--listen-peer-urls",
                f"http://0.0.0.0:{config.membership.etcd_peer_port}",
                "--initial-advertise-peer-urls",
                f"http://{config.membership.host}:{config.membership.etcd_peer_port}",
                "--initial-cluster",
                f"{config.node_id}=http://{config.membership.host}:{config.membership.etcd_peer_port}",
                "--initial-cluster-state",
                "new",
                "--initial-cluster-token",
                config.cluster_id,
            ],
            env=etcd_env,
            log_path=paths.logs / "etcd.log",
            pid_path=paths.run / "etcd.pid",
        ),
        ManagedProcess(
            name="membership-etcd",
            args=[membership],
            env=membership_env,
            log_path=paths.logs / "membership-etcd.log",
            pid_path=paths.run / "membership-etcd.pid",
        ),
    ]


def _llama_server_binary(configured: str) -> str:
    resolved = resolve_executable(configured, "llama-server")
    if resolved:
        return resolved
    return ensure_llama_server()


def _signal_group(pid: int, sig: signal.Signals) -> None:
    """Send a signal to the process group of pid.

    Each managed process is started with ``start_new_session=True``, making it
    the leader of its own process group. Sending to the group rather than just
    the PID ensures all children (e.g. llama-server spawned by the worker) are
    reached in a single call. Falls back to a direct pid kill if the group
    lookup fails (process already gone, permission denied, etc.).
    """
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _start_background(spec: ManagedProcess) -> int:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = spec.log_path.open("ab")
    try:
        process = subprocess.Popen(
            spec.args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=spec.env,
            start_new_session=True,
        )
    finally:
        log_file.close()
    spec.pid_path.write_text(str(process.pid))
    return process.pid


def _run_foreground(spec: ManagedProcess) -> None:
    process = subprocess.Popen(spec.args, env=spec.env)
    process.wait()


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_matches_spec(pid: int, spec: ManagedProcess) -> bool:
    if not _pid_running(pid):
        return False
    cmdline = _process_cmdline(pid)
    if not cmdline:
        return False
    if spec.name == "coordinator":
        return "coordinator.main:app" in cmdline
    if spec.name == "worker":
        return "worker.main:app" in cmdline
    if spec.name == "membership-etcd":
        return "membership-etcd" in cmdline
    if spec.name == "etcd":
        return "etcd" in cmdline
    return False


def _process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="replace")


def _wait_for_tcp(host: str, port: int, name: str) -> None:
    deadline = time.monotonic() + 10.0
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as e:
            last_error = e
            time.sleep(0.1)
    detail = f": {last_error}" if last_error else ""
    raise SupervisorError(f"{name} did not open {host}:{port} in time{detail}")


def _health_get(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url)
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "body": response.json(),
        }
    except (httpx.RequestError, ValueError) as e:
        return {"ok": False, "error": str(e)}


def read_log_tail(name: str, *, lines: int = 80) -> str:
    """Read the last lines from a managed process log."""
    path = RelayPaths.from_home().logs / f"{name}.log"
    if not path.exists():
        raise SupervisorError(f"No log file for {name}: {path}")
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def process_status_json(statuses: list[ProcessStatus]) -> str:
    """Return statuses as JSON for scripts."""
    return json.dumps([status.__dict__ for status in statuses], indent=2)
