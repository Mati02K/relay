import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel

import logger as loggerSetup
from membership.etcd import EtcdMembership
from network.tailscale import TailscaleNetwork

NODE_ID: str = os.getenv("NODE_ID", "coordinator")
COORDINATOR_PORT: int = int(os.getenv("COORDINATOR_PORT", "8080"))
ACTIVE_COORDINATOR_KEY: str = "/relay/active-coordinator"

membershipClient: EtcdMembership | None = None
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Set up logging, membership, and leadership campaign on startup."""
    global membershipClient

    loggerSetup.setup()
    log = logger.bind(nodeId=NODE_ID)

    membershipClient = EtcdMembership(
        host=os.getenv("MEMBERSHIP_HOST", "localhost"),
        port=int(os.getenv("MEMBERSHIP_PORT", "50051")),
    )
    networkLayer = TailscaleNetwork()

    myAddress = networkLayer.getMyAddress()
    myAddr = f"http://{os.getenv('COORDINATOR_HOST', myAddress)}:{COORDINATOR_PORT}"
    log.info("Coordinator starting | addr={}", myAddr)

    await membershipClient.register(NODE_ID, {"role": "coordinator", "ip": myAddress})
    log.info("Registered in membership layer | nodeId={}", NODE_ID)

    # Start leader election in background — does not block startup.
    asyncio.create_task(_leadershipLoop(membershipClient, myAddr))

    yield

    log.info("Coordinator shutting down | nodeId={}", NODE_ID)
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


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Store a chat request in etcd; only the active coordinator accepts this."""
    _requireActive()
    assert membershipClient is not None
    msgId = str(uuid.uuid4())
    payload = {"messages": [m.model_dump() for m in request.messages]}
    await membershipClient.put(f"/relay/messages/{msgId}", json.dumps(payload))
    logger.info("Message stored | id={} messageCount={}", msgId, len(request.messages))
    return ChatResponse(id=msgId, stored=True)


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
