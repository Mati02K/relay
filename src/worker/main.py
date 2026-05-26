import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

import logger as loggerSetup
from membership.etcd import EtcdMembership
from worker.inference.llamacpp import LlamaCppEngine

NODE_ID: str = os.getenv("NODE_ID", "worker")
WORKER_PORT: int = int(os.getenv("WORKER_PORT", "9090"))
ACTIVE_COORDINATOR_KEY: str = "/relay/active-coordinator"

membershipClient: EtcdMembership | None = None
_coordinatorUrl: str | None = None
# Created on startup; model subprocess is launched lazily on the first /v1/generate call.
inferenceEngine: LlamaCppEngine | None = None


def _resolveWorkerHost() -> str:
    """Pick this worker's reachable host: WORKER_HOST env > Tailscale IP > 127.0.0.1 fallback."""
    explicit = os.getenv("WORKER_HOST")
    if explicit:
        return explicit
    try:
        from network.tailscale import TailscaleNetwork

        return TailscaleNetwork().getMyAddress()
    except Exception:
        logger.warning("Could not resolve Tailscale address, falling back to 127.0.0.1")
        return "127.0.0.1"


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
    """Set up logging, membership client, coordinator discovery, and inference engine on startup."""
    global membershipClient, inferenceEngine
    loggerSetup.setup()
    membershipClient = EtcdMembership(
        host=os.getenv("MEMBERSHIP_HOST", "localhost"),
        port=int(os.getenv("MEMBERSHIP_PORT", "50051")),
    )
    inferenceEngine = LlamaCppEngine()

    workerHost = _resolveWorkerHost()
    workerUrl = f"http://{workerHost}:{WORKER_PORT}"
    metadata: dict = {"role": "worker", "ip": workerHost, "url": workerUrl}
    # Optional self-described compute strength (0..1); read by the phase-aware
    # term in the scheduler. Defaults are 0.5 in scheduler if not provided.
    compute_strength_raw = os.getenv("RELAY_COMPUTE_STRENGTH")
    if compute_strength_raw:
        try:
            metadata["compute_strength"] = max(0.0, min(1.0, float(compute_strength_raw)))
        except ValueError:
            logger.warning("Invalid RELAY_COMPUTE_STRENGTH | raw={}", compute_strength_raw)
    # Self-described model quality (0..1); consumed by the quality-aware (nu)
    # cost term so the scheduler can route complex prompts to higher-quality
    # workers in the RouteLLM sense. Defaults to 0.5 in the scheduler if unset.
    model_quality_raw = os.getenv("RELAY_MODEL_QUALITY")
    if model_quality_raw:
        try:
            metadata["model_quality"] = max(0.0, min(1.0, float(model_quality_raw)))
        except ValueError:
            logger.warning("Invalid RELAY_MODEL_QUALITY | raw={}", model_quality_raw)
    await membershipClient.register(NODE_ID, metadata)

    asyncio.create_task(_watchCoordinator())
    logger.info(
        "Worker started | nodeId={} url={} engineModel={}",
        NODE_ID,
        workerUrl,
        inferenceEngine._model_path or "<unset>",
    )
    yield
    try:
        await membershipClient.deregister(NODE_ID)
    except Exception:
        logger.exception("Failed to deregister worker | nodeId={}", NODE_ID)
    if inferenceEngine is not None:
        await inferenceEngine.stop()
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


# ---------------------------------------------------------------------------
# Data-plane endpoints (called BY the coordinator's scheduler, not by clients)
# ---------------------------------------------------------------------------


@app.get("/v1/telemetry")
async def telemetry() -> dict:
    """Return the current inference engine telemetry snapshot for the scheduler."""
    if inferenceEngine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    snapshot = await inferenceEngine.get_telemetry()
    return snapshot.model_dump()


@app.post("/v1/generate")
async def generate(request: Request) -> StreamingResponse:
    """Stream a chat completion from this worker's inference engine (OpenAI-compatible SSE)."""
    if inferenceEngine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return StreamingResponse(
        inferenceEngine.generate(body),
        media_type="text/event-stream",
    )
