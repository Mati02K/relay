import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

import logger as loggerSetup
from coordinator.scheduler import (
    CostWeights,
    estimate_ttft_ms,
    messages_to_prompt_text,
    pick_worker,
)
from coordinator.worker_registry import WorkerRegistry
from membership.etcd import EtcdMembership
from network.tailscale import TailscaleNetwork
from worker.inference.base import estimate_tokens_from_text

_TTFT_SLO_MS: float = float(os.getenv("RELAY_TTFT_SLO_MS", "0"))

NODE_ID: str = os.getenv("NODE_ID", "coordinator")
COORDINATOR_PORT: int = int(os.getenv("COORDINATOR_PORT", "8080"))
ACTIVE_COORDINATOR_KEY: str = "/relay/active-coordinator"

membershipClient: EtcdMembership | None = None
workerRegistry: WorkerRegistry | None = None
_isActive: bool = False


class Message(BaseModel):
    """A single chat message with a role and content."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Incoming chat request containing a list of messages."""

    messages: list[Message]


class ChatResponse(BaseModel):
    """Response returned after a chat request is stored in etcd."""

    id: str
    stored: bool


class HealthResponse(BaseModel):
    """Health check response with node identity and leader status."""

    status: str
    nodeId: str
    isLeader: bool


async def _leadershipLoop(membership: EtcdMembership, myAddr: str) -> None:
    """Campaign for leadership in a loop; write address to etcd on win, retry on loss."""
    global _isActive
    while True:
        try:
            logger.info("Campaigning for leadership | nodeId={}", NODE_ID)
            async for isLeader in membership.holdLeadership():
                if isLeader:
                    _isActive = True
                    await membership.put(ACTIVE_COORDINATOR_KEY, myAddr)
                    logger.info("Won election — now active coordinator | addr={}", myAddr)
            # Stream ended = lost leadership (shouldn't happen in normal flow)
            _isActive = False
            await membership.put(ACTIVE_COORDINATOR_KEY, "")
            logger.warning("Lost leadership, re-campaigning | nodeId={}", NODE_ID)
        except Exception:
            logger.exception("Leadership loop error, retrying in 3s")
            _isActive = False
            await asyncio.sleep(3)


def _resolveCoordinatorAddress() -> tuple[str, str]:
    """Pick this coordinator's reachable address; falls back to localhost if Tailscale is absent."""
    explicit = os.getenv("COORDINATOR_HOST")
    if explicit:
        return explicit, f"http://{explicit}:{COORDINATOR_PORT}"
    try:
        ip = TailscaleNetwork().getMyAddress()
        return ip, f"http://{ip}:{COORDINATOR_PORT}"
    except Exception:
        logger.warning("Could not resolve Tailscale address, falling back to 127.0.0.1")
        return "127.0.0.1", f"http://127.0.0.1:{COORDINATOR_PORT}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Set up logging, membership, leadership, and worker telemetry registry on startup."""
    global membershipClient, workerRegistry

    loggerSetup.setup()
    log = logger.bind(nodeId=NODE_ID)

    membershipClient = EtcdMembership(
        host=os.getenv("MEMBERSHIP_HOST", "localhost"),
        port=int(os.getenv("MEMBERSHIP_PORT", "50051")),
    )

    myAddress, myAddr = _resolveCoordinatorAddress()
    log.info("Coordinator starting | addr={}", myAddr)

    await membershipClient.register(NODE_ID, {"role": "coordinator", "ip": myAddress})
    log.info("Registered in membership layer | nodeId={}", NODE_ID)

    workerRegistry = WorkerRegistry(membershipClient)
    workerRegistry.start()

    asyncio.create_task(_leadershipLoop(membershipClient, myAddr))

    yield

    log.info("Coordinator shutting down | nodeId={}", NODE_ID)
    if workerRegistry is not None:
        await workerRegistry.stop()
    await membershipClient.put(ACTIVE_COORDINATOR_KEY, "")
    await membershipClient.deregister(NODE_ID)
    await membershipClient.close()


app = FastAPI(title="Relay Coordinator", lifespan=lifespan)


@app.middleware("http")
async def logRequests(request: Request, callNext: object) -> Response:
    """Log every incoming HTTP request with response status and duration."""
    start = time.perf_counter()
    response: Response = await callNext(request)  # type: ignore[operator]
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "{} {} -> {} | {:.1f}ms",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


def _requireActive() -> None:
    """Raise 503 if this coordinator is not the current leader."""
    if not _isActive:
        raise HTTPException(status_code=503, detail="Not the active coordinator")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return health status, node ID, and whether this node is the active leader."""
    return HealthResponse(status="ok", nodeId=NODE_ID, isLeader=_isActive)


@app.get("/v1/cluster")
async def clusterView() -> dict:
    """Debug view of the live worker telemetry registry (paper §3.2 inputs)."""
    if workerRegistry is None:
        raise HTTPException(status_code=503, detail="Worker registry not initialized")
    snapshot = workerRegistry.snapshot()
    return {
        "workers": [
            {
                "nodeId": state.node_id,
                "url": state.url,
                "online": state.online,
                "rttMsEma": round(state.rtt_ms_ema, 2),
                "jitterMsEma": round(state.jitter_ms_ema, 2),
                "consecutiveFailures": state.consecutive_failures,
                "lastSeenMsAgo": round(
                    (time.monotonic() * 1000 - state.last_seen_ms), 1
                ) if state.last_seen_ms > 0 else None,
                "telemetry": state.telemetry.model_dump(),
                "metadata": state.metadata,
            }
            for state in snapshot.values()
        ]
    }


@app.post("/v1/chat")
async def chat(request: Request) -> StreamingResponse:
    """Schedule a chat request to the best worker and stream the response back to the client."""
    _requireActive()
    assert workerRegistry is not None

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

    workers = workerRegistry.online_workers()
    if not workers:
        raise HTTPException(status_code=503, detail="No healthy workers available")

    prompt_text = messages_to_prompt_text(messages)
    weights = CostWeights.from_env()
    winner_id, breakdowns = pick_worker(prompt_text, workers, weights=weights)
    if winner_id is None:
        raise HTTPException(status_code=503, detail="Scheduler could not select a worker")

    winner = workers[winner_id]

    if _TTFT_SLO_MS > 0:
        prompt_tokens = estimate_tokens_from_text(prompt_text)
        predicted_ttft_ms = estimate_ttft_ms(winner, prompt_tokens)
        if predicted_ttft_ms > _TTFT_SLO_MS:
            logger.warning(
                "Shedding request | nodeId={} predictedTtftMs={:.0f} sloMs={:.0f}",
                winner_id,
                predicted_ttft_ms,
                _TTFT_SLO_MS,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Predicted TTFT {predicted_ttft_ms:.0f}ms exceeds SLO {_TTFT_SLO_MS:.0f}ms",
            )

    request_id = str(uuid.uuid4())
    winner_breakdown = next(b for b in breakdowns if b.node_id == winner_id)
    logger.info(
        "Routing chat | reqId={} winner={} overlap={:.2f} cost={:.4f} candidates={}",
        request_id,
        winner_id,
        winner_breakdown.overlap,
        winner_breakdown.total,
        len(workers),
    )

    async def _stream_from_worker() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream(
                "POST",
                f"{winner.url}/v1/generate",
                json=body,
                headers={"Accept": "text/event-stream"},
            ) as upstream:
                if upstream.status_code >= 400:
                    err = (await upstream.aread()).decode(errors="replace")
                    logger.error(
                        "Upstream worker error | reqId={} nodeId={} status={} body={}",
                        request_id,
                        winner_id,
                        upstream.status_code,
                        err[:500],
                    )
                    yield f"data: {{\"error\": \"upstream {upstream.status_code}\"}}\n\n".encode()
                    return
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk

    return StreamingResponse(
        _stream_from_worker(),
        media_type="text/event-stream",
        headers={"X-Relay-Worker": winner_id, "X-Relay-Request-Id": request_id},
    )


@app.get("/v1/messages", response_model=dict)
async def listMessages() -> dict:
    """Return all stored chat requests from etcd; only the active coordinator serves this."""
    _requireActive()
    assert membershipClient is not None
    raw = await membershipClient.getByPrefix("/relay/messages/")
    result = {
        key.removeprefix("/relay/messages/"): json.loads(value)
        for key, value in raw.items()
    }
    logger.debug("Listed messages | count={}", len(result))
    return result


@app.get("/v1/messages/{msgId}", response_model=dict)
async def getMessage(msgId: str) -> dict:
    """Retrieve a single stored chat request by ID; only the active coordinator serves this."""
    _requireActive()
    assert membershipClient is not None
    value = await membershipClient.get(f"/relay/messages/{msgId}")
    if value is None:
        raise HTTPException(status_code=404, detail=f"Message {msgId} not found")
    return json.loads(value)
