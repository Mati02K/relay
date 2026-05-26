"""Process runner for the dashboard.

Detects port conflicts before launching uvicorn so the CLI can surface a clean
error to the user. Writes a PID file under ``~/.relay/run/dashboard.pid`` so
``relay stop`` and ``relay status`` see the dashboard alongside the other
managed processes. Also opens the browser after the server is reachable.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser

import uvicorn

from relay.paths import RelayPaths


class PortInUseError(RuntimeError):
    """Raised when the requested dashboard port is already bound."""


def assert_port_available(host: str, port: int) -> None:
    """Bind-and-release ``host:port`` to confirm it is free, or raise."""
    bind_host = "" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in host and host != "::" else socket.AF_INET
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError as e:
        raise PortInUseError(f"Could not open a socket on {host}: {e}") from e
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
    except OSError as e:
        raise PortInUseError(
            f"Port {port} on {host or '0.0.0.0'} is already in use. "
            f"Pass --port to choose a different one."
        ) from e
    finally:
        sock.close()


def run_dashboard(
    *,
    host: str,
    port: int,
    coordinator_url: str,
    open_browser: bool,
) -> None:
    """Start uvicorn for the dashboard app, optionally opening a browser.

    Writes ``~/.relay/run/dashboard.pid`` so the supervisor sees the dashboard
    alongside coordinator/worker/etcd; removes it on clean exit. ``relay stop``
    picks the dashboard up via the same pid-file mechanism it uses for the rest.
    """
    assert_port_available(host, port)
    os.environ["RELAY_COORDINATOR_URL"] = coordinator_url

    paths = RelayPaths.from_home()
    paths.ensure()
    pid_path = paths.run / "dashboard.pid"
    pid_path.write_text(str(os.getpid()))

    if open_browser:
        threading.Thread(
            target=_open_when_ready,
            args=(host, port),
            daemon=True,
        ).start()
    try:
        uvicorn.run("dashboard.main:app", host=host, port=port, log_level="info")
    finally:
        pid_path.unlink(missing_ok=True)


def _open_when_ready(host: str, port: int) -> None:
    target_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((target_host, port), timeout=0.5):
                webbrowser.open(f"http://{target_host}:{port}")
                return
        except OSError:
            time.sleep(0.2)
