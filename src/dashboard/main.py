"""FastAPI proxy for the Relay testing dashboard.

The dashboard is a small same-origin HTTP front-end. It serves the static UI
and proxies two endpoints to the coordinator:

* ``POST /api/chat/completions`` — streams the SSE response through and lifts
  the ``X-Relay-*`` headers so the UI can show which worker handled the
  request and the scheduler diagnostics.
* ``GET /api/workers`` — passes through the coordinator's worker registry so
  the sidebar can render live telemetry.

Running the dashboard as a proxy avoids needing CORS on the coordinator.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

DEFAULT_COORDINATOR_URL = "http://127.0.0.1:8080"
_RELAY_HEADER_NAMES = (
    "x-relay-worker",
    "x-relay-cost",
    "x-relay-matched-tokens",
    "x-relay-prompt-tokens",
    "x-relay-overlap",
    "x-relay-attempts",
)
_STATIC_DIR = Path(__file__).parent / "static"


def coordinator_url() -> str:
    """Return the configured coordinator base URL."""
    return os.getenv("RELAY_COORDINATOR_URL", DEFAULT_COORDINATOR_URL).rstrip("/")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open a shared HTTP client for proxy calls and close it on shutdown."""
    timeout = httpx.Timeout(timeout=None, connect=10.0)
    app.state.http = httpx.AsyncClient(timeout=timeout)
    logger.info("Dashboard ready | coordinator={}", coordinator_url())
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="Relay Dashboard", lifespan=lifespan)

if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR), name="assets")


@app.get("/")
async def index() -> FileResponse:
    """Serve the dashboard single-page UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/config")
async def config() -> dict[str, str]:
    """Return runtime config the UI needs (currently just the coordinator URL)."""
    return {"coordinator_url": coordinator_url()}


@app.get("/api/workers")
async def workers(request: Request) -> list[dict[str, Any]]:
    """Proxy the coordinator's worker registry for the sidebar."""
    client: httpx.AsyncClient = request.app.state.http
    url = f"{coordinator_url()}/v1/workers"
    try:
        response = await client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Coordinator unreachable: {e}") from e
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text[:500],
        )
    payload = response.json()
    if not isinstance(payload, list):
        raise HTTPException(status_code=502, detail="Unexpected /v1/workers payload")
    return [item for item in payload if isinstance(item, dict)]


@app.post("/api/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    """Proxy a chat completion request to the coordinator and stream back the response."""
    client: httpx.AsyncClient = request.app.state.http
    body = await request.json()
    url = f"{coordinator_url()}/v1/chat/completions"

    stream_ctx = client.stream("POST", url, json=body)
    try:
        upstream = await stream_ctx.__aenter__()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Coordinator unreachable: {e}") from e

    if upstream.status_code >= 400:
        error_body = (await upstream.aread()).decode("utf-8", errors="replace")
        await stream_ctx.__aexit__(None, None, None)
        raise HTTPException(status_code=upstream.status_code, detail=error_body[:500])

    headers = {
        name: upstream.headers[name] for name in _RELAY_HEADER_NAMES if name in upstream.headers
    }

    async def streamer() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)

    return StreamingResponse(streamer(), media_type="text/event-stream", headers=headers)
