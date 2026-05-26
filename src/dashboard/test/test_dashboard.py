from __future__ import annotations

import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from dashboard import main as dashboard_main
from dashboard.server import PortInUseError, assert_port_available


def test_assert_port_available_passes_on_free_port() -> None:
    free_port = _find_free_port()
    assert_port_available("127.0.0.1", free_port)


def test_assert_port_available_raises_when_bound() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        with pytest.raises(PortInUseError) as info:
            assert_port_available("127.0.0.1", port)
    assert str(port) in str(info.value)
    assert "--port" in str(info.value)


def test_config_endpoint_returns_coordinator_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_COORDINATOR_URL", "http://example:8080")
    with TestClient(dashboard_main.app) as client:
        response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json() == {"coordinator_url": "http://example:8080"}


def test_workers_endpoint_proxies_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"node_id": "w1", "address": "http://w1", "telemetry": {"qw": 0}}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/workers"
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)

    with TestClient(dashboard_main.app) as client:
        response = client.get("/api/workers")
    assert response.status_code == 200
    assert response.json() == payload


def test_workers_endpoint_502_on_coordinator_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_mock_transport(monkeypatch, handler)

    with TestClient(dashboard_main.app) as client:
        response = client.get("/api/workers")
    assert response.status_code == 502


def test_chat_completions_lifts_relay_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            content=b"data: hello\n\ndata: [DONE]\n\n",
            headers={
                "X-Relay-Worker": "w1",
                "X-Relay-Cost": "0.42",
                "X-Relay-Matched-Tokens": "12",
                "X-Relay-Prompt-Tokens": "50",
                "X-Relay-Overlap": "0.24",
            },
        )

    _install_mock_transport(monkeypatch, handler)

    with TestClient(dashboard_main.app) as client:
        response = client.post(
            "/api/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.headers["x-relay-worker"] == "w1"
    assert response.headers["x-relay-cost"] == "0.42"
    assert response.headers["x-relay-matched-tokens"] == "12"
    assert b"hello" in response.content


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: "httpx._types.RequestHandler",  # type: ignore[name-defined]
) -> None:
    """Make ``dashboard.main`` use an httpx client routed through ``MockTransport``."""
    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return original_async_client(transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("dashboard.main.httpx.AsyncClient", factory)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
