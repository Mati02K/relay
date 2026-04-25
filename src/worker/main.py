import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from loguru import logger

import logger as loggerSetup
from membership.etcd import EtcdMembership

NODE_ID: str = os.getenv("NODE_ID", "worker")
ACTIVE_COORDINATOR_KEY: str = "/relay/active-coordinator"

membershipClient: EtcdMembership | None = None
_coordinatorUrl: str | None = None


async def _watchCoordinator() -> None:
    """Background task: poll etcd until a coordinator URL is found, then keep it fresh."""
    global _coordinatorUrl, membershipClient
    while True:
        try:
            assert membershipClient is not None
            url = await membershipClient.get(ACTIVE_COORDINATOR_KEY)
            if url and url != _coordinatorUrl:
                _coordinatorUrl = url
                logger.info("Coordinator discovered | url={}", url)
            elif not url and _coordinatorUrl:
                _coordinatorUrl = None
                logger.warning("Active coordinator key cleared, awaiting re-election")
        except Exception:
            logger.exception("Error polling coordinator key")
        await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Set up logging, membership client, and coordinator discovery on startup."""
    global membershipClient
    loggerSetup.setup()
    membershipClient = EtcdMembership(
        host=os.getenv("MEMBERSHIP_HOST", "localhost"),
        port=int(os.getenv("MEMBERSHIP_PORT", "50051")),
    )
    asyncio.create_task(_watchCoordinator())
    logger.info("Worker started | nodeId={}", NODE_ID)
    yield
    await membershipClient.close()
    logger.info("Worker shutting down | nodeId={}", NODE_ID)


app = FastAPI(title="Relay Worker", lifespan=lifespan)


@app.middleware("http")
async def logRequests(request: Request, callNext: object) -> Response:
    """Log every incoming request with response status and duration."""
    start = time.perf_counter()
    response: Response = await callNext(request)  # type: ignore[operator]
    duration = (time.perf_counter() - start) * 1000
    logger.info("{} {} -> {} | {:.1f}ms", request.method, request.url.path, response.status_code, duration)
    return response


async def _getCoordinatorUrl() -> str:
    """Return the cached active coordinator URL, re-fetching from etcd if stale."""
    global _coordinatorUrl, membershipClient
    if _coordinatorUrl:
        return _coordinatorUrl
    assert membershipClient is not None
    url = await membershipClient.get(ACTIVE_COORDINATOR_KEY)
    if not url:
        raise HTTPException(status_code=503, detail="No active coordinator available")
    _coordinatorUrl = url
    logger.info("Coordinator discovered | url={}", url)
    return url


async def _forward(method: str, path: str, body: dict | None = None) -> dict:
    """Forward a request to the active coordinator; invalidate cache on failure."""
    global _coordinatorUrl
    url = await _getCoordinatorUrl()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if method == "POST":
                resp = await client.post(f"{url}{path}", json=body)
            else:
                resp = await client.get(f"{url}{path}")
        if resp.status_code == 503:
            # Coordinator is no longer active — clear cache and retry once.
            _coordinatorUrl = None
            logger.warning("Coordinator returned 503, retrying discovery")
            url = await _getCoordinatorUrl()
            async with httpx.AsyncClient(timeout=30) as client:
                if method == "POST":
                    resp = await client.post(f"{url}{path}", json=body)
                else:
                    resp = await client.get(f"{url}{path}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()  # type: ignore[no-any-return]
    except httpx.RequestError:
        _coordinatorUrl = None
        logger.exception("Failed to reach coordinator, clearing cache")
        raise HTTPException(status_code=503, detail="Coordinator unreachable")


@app.get("/health")
async def health() -> dict:
    """Return worker health; includes whether a coordinator is currently known."""
    return {"status": "ok", "nodeId": NODE_ID, "coordinatorUrl": _coordinatorUrl}


@app.post("/v1/chat")
async def chat(request: Request) -> dict:
    """Forward a chat request to the active coordinator."""
    body = await request.json()
    return await _forward("POST", "/v1/chat", body)


@app.get("/v1/messages")
async def listMessages() -> dict:
    """Forward a list-messages request to the active coordinator."""
    return await _forward("GET", "/v1/messages")


@app.get("/v1/messages/{msgId}")
async def getMessage(msgId: str) -> dict:
    """Forward a get-message request to the active coordinator."""
    return await _forward("GET", f"/v1/messages/{msgId}")
