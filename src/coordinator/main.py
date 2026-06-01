import asyncio
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator, cast

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

import logger as loggerSetup
from coordinator import inflight
from coordinator import scheduler as scheduler_module
from coordinator.scheduler import SchedulingError, WorkerChoice, choose_worker
from coordinator.worker_registry import WorkerSnapshot, fetch_worker_snapshots
from membership.etcd import EtcdMembership
from network import NetworkLayer, build_network
from telemetry.jitter import JitterProbe

NODE_ID: str = os.getenv("NODE_ID", "coordinator")
COORDINATOR_PORT: int = int(os.getenv("COORDINATOR_PORT", "8080"))
ACTIVE_COORDINATOR_KEY: str = "/relay/active-coordinator"
SCHEDULE_MAX_ATTEMPTS: int = max(1, int(os.getenv("RELAY_SCHEDULE_MAX_ATTEMPTS", "3")))
WORKER_HEALTH_TIMEOUT_SECONDS: float = float(os.getenv("RELAY_WORKER_HEALTH_TIMEOUT", "2.0"))
# How often the background monitor probes every worker's /health, and how long a
# cached result stays usable. Requests read the cache instead of probing inline,
# so the /health probe rate is fixed (workers / interval) regardless of request
# rate — a burst no longer fans out into a storm of probes that drops workers.
HEALTH_PROBE_INTERVAL_SECONDS: float = float(os.getenv("RELAY_HEALTH_PROBE_INTERVAL", "1.0"))
HEALTH_STALE_AFTER_SECONDS: float = float(os.getenv("RELAY_HEALTH_STALE_AFTER", "5.0"))

membershipClient: EtcdMembership | None = None
# Background-refreshed worker health, keyed by node id: (monotonic_ts, state).
_health_cache: dict[str, tuple[float, dict[str, Any]]] = {}
jitterProbe: JitterProbe | None = None
networkLayer: NetworkLayer | None = None
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
    """Set up logging, membership, jitter probe, and leadership campaign on startup."""
    global membershipClient, jitterProbe, networkLayer

    loggerSetup.setup()
    log = logger.bind(nodeId=NODE_ID)

    membershipClient = EtcdMembership(
        host=os.getenv("MEMBERSHIP_HOST", "localhost"),
        port=int(os.getenv("MEMBERSHIP_PORT", "50051")),
    )
    networkLayer = build_network()
    configuredHost = os.getenv("COORDINATOR_HOST")
    myAddress = configuredHost or networkLayer.getMyAddress()
    myAddr = f"http://{myAddress}:{COORDINATOR_PORT}"
    log.info("Coordinator starting | addr={}", myAddr)

    await membershipClient.register(NODE_ID, {"role": "coordinator", "ip": myAddress})
    log.info("Registered in membership layer | nodeId={}", NODE_ID)

    jitterProbe = JitterProbe(membershipClient)
    await jitterProbe.start()

    # Probe worker health on a fixed cadence so requests read a cache instead of
    # probing inline (avoids the per-request health-probe storm under load).
    asyncio.create_task(_health_monitor_loop())

    # Start leader election in background — does not block startup.
    asyncio.create_task(_leadershipLoop(membershipClient, myAddr))

    yield

    log.info("Coordinator shutting down | nodeId={}", NODE_ID)
    if jitterProbe is not None:
        await jitterProbe.stop()
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


@app.post("/v1/chat/completions")
async def chatCompletions(request: Request) -> StreamingResponse:
    """Schedule and stream an OpenAI-compatible chat completion from a worker."""
    _requireActive()
    assert membershipClient is not None

    raw_body = await request.json()
    if not isinstance(raw_body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    body: dict[str, object] = {}
    for key, value in raw_body.items():
        if not isinstance(key, str):
            raise HTTPException(status_code=400, detail="Request body keys must be strings")
        body[key] = value

    workers = await fetch_worker_snapshots(membershipClient)
    if jitterProbe is not None:
        for worker in workers:
            measured = jitterProbe.get_jitter_ms(worker.node_id)
            if measured is not None:
                worker.telemetry.jw = measured
    ready_workers = await _ready_worker_snapshots(workers)

    try:
        choice, opened, attempts = await _open_stream_with_retry(
            body,
            ready_workers,
            lambda worker: _openWorkerStream(worker, body),
            SCHEDULE_MAX_ATTEMPTS,
        )
    except SchedulingError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except WorkerUnreachable as e:
        raise HTTPException(
            status_code=503,
            detail=f"All eligible workers unreachable; last error: {e}",
        ) from e

    logger.info(
        "Scheduled chat completion | worker={} matchedTokens={} uncachedTokens={} "
        "cost={:.3f} attempts={}",
        choice.worker.node_id,
        choice.matched_tokens,
        choice.uncached_prompt_tokens,
        choice.cost,
        attempts,
    )

    routing_decision = json.dumps(
        {
            "worker": choice.worker.node_id,
            "model": choice.model_id,
            "skills_needed": sorted(choice.skills_needed),
            "skills_available": sorted(choice.skills_available),
            "skill_match": choice.skill_match,
            "complexity": round(choice.complexity, 4),
            "model_quality": round(choice.model_quality, 4),
            "quality_term": round(choice.quality_term, 4),
            "cost": round(choice.cost, 4),
        },
        separators=(",", ":"),
    )

    return StreamingResponse(
        _streamWorkerResponse(
            opened.client, opened.stream_ctx, opened.upstream, choice.worker.node_id
        ),
        media_type="text/event-stream",
        headers={
            "X-Relay-Worker": choice.worker.node_id,
            "X-Relay-Model": choice.model_id,
            "X-Relay-Cost": f"{choice.cost:.4f}",
            "X-Relay-Matched-Tokens": str(choice.matched_tokens),
            "X-Relay-Prompt-Tokens": str(choice.prompt_tokens),
            "X-Relay-Overlap": f"{choice.overlap:.4f}",
            "X-Relay-Complexity": f"{choice.complexity:.4f}",
            "X-Relay-Model-Quality": f"{choice.model_quality:.4f}",
            "X-Relay-Quality-Term": f"{choice.quality_term:.4f}",
            "X-Relay-Skills-Needed": ",".join(sorted(choice.skills_needed)),
            "X-Relay-Skills-Available": ",".join(sorted(choice.skills_available)),
            "X-Relay-Skill-Match": "true" if choice.skill_match else "false",
            "X-Relay-Routing-Decision": routing_decision,
            "X-Relay-Attempts": str(attempts),
            "Access-Control-Expose-Headers": (
                "X-Relay-Worker, X-Relay-Model, X-Relay-Cost, "
                "X-Relay-Matched-Tokens, X-Relay-Prompt-Tokens, "
                "X-Relay-Overlap, X-Relay-Complexity, "
                "X-Relay-Model-Quality, X-Relay-Quality-Term, "
                "X-Relay-Skills-Needed, X-Relay-Skills-Available, "
                "X-Relay-Skill-Match, X-Relay-Routing-Decision, "
                "X-Relay-Attempts"
            ),
        },
    )


@app.get("/v1/models")
async def listModels() -> dict[str, Any]:
    """OpenAI-compatible model list: union of loaded models across all workers.

    Used by OpenAI-style clients (Open WebUI etc.) to populate the model
    selector when they connect. Duplicates across workers are folded into a
    single entry — the scheduler decides at request time which worker actually
    serves it.
    """
    _requireActive()
    assert membershipClient is not None
    workers = await _ready_worker_snapshots(await fetch_worker_snapshots(membershipClient))
    return {"object": "list", "data": _collect_model_entries(workers)}


def _collect_model_entries(workers: Sequence[WorkerSnapshot]) -> list[dict[str, Any]]:
    """Return a deduplicated OpenAI-shaped list of models advertised by workers."""
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for worker in workers:
        models = worker.metadata.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id or model_id in seen:
                continue
            if not model.get("loaded", True):
                continue
            seen.add(model_id)
            entries.append({"id": model_id, "object": "model", "owned_by": "relay"})
    return entries


@app.get("/v1/workers")
async def listWorkers() -> list[dict[str, Any]]:
    """Return registered workers with their latest telemetry for dashboards/tools."""
    _requireActive()
    assert membershipClient is not None
    workers = await fetch_worker_snapshots(membershipClient)
    for worker in workers:
        # Show the coordinator's live in-flight count as qw — the same signal the
        # scheduler routes on — rather than the now-unused worker-published value.
        worker.telemetry.qw = inflight.depth(worker.node_id)
        if jitterProbe is not None:
            measured = jitterProbe.get_jitter_ms(worker.node_id)
            if measured is not None:
                worker.telemetry.jw = measured
    health_by_node = {worker.node_id: _cached_health(worker.node_id) for worker in workers}
    return [
        {
            "node_id": worker.node_id,
            "address": worker.address,
            "models": worker.metadata.get("models", []),
            "engines": worker.metadata.get("engines", []),
            "weight": worker.metadata.get("weight", 0.0),
            "weight_override": scheduler_module.WORKER_WEIGHT_OVERRIDES.get(worker.node_id),
            "healthy": health_by_node.get(worker.node_id, {}).get("healthy", False),
            "health": health_by_node.get(worker.node_id, _unknown_worker_health()),
            "telemetry": worker.telemetry.model_dump(),
        }
        for worker in workers
    ]


@app.get("/v1/scheduler/mode")
async def getSchedulerMode() -> dict[str, str]:
    """Return the current scheduler mode (``cost`` or ``round_robin``)."""
    _requireActive()
    return {"mode": scheduler_module.get_mode()}


@app.post("/v1/scheduler/mode")
async def setSchedulerMode(payload: dict[str, str]) -> dict[str, str]:
    """Switch the scheduler mode at runtime.

    Body: ``{"mode": "cost"}`` or ``{"mode": "round_robin"}``. Cost mode runs
    the full paper cost formula plus the chart-driven router; round-robin
    bypasses scoring and rotates across model + input-modality eligible
    workers in stable node-id order. The change takes effect on the next
    request and is not persisted across coordinator restarts.
    """
    _requireActive()
    if not isinstance(payload, dict) or "mode" not in payload:
        raise HTTPException(
            status_code=400,
            detail='Body must be a JSON object with a "mode" field',
        )
    raw = payload["mode"]
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="'mode' must be a string")
    try:
        new_mode = scheduler_module.set_mode(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    logger.info("Scheduler mode set | mode={}", new_mode)
    return {"mode": new_mode}


@app.get("/v1/scheduler/weights")
async def getSchedulerWeights() -> dict[str, float]:
    """Return the live cost-function weights the scheduler is using right now."""
    _requireActive()
    return scheduler_module.WEIGHTS.as_dict()


# Allowed ranges for each weight in POST /v1/scheduler/weights. The base
# 5 stay in [0, 1] to match the paper-style cost terms; ``nu`` is the
# RouteLLM quality knob. We cap nu at 5.0 — below the paper's nu=20
# operating point — because our complexity classifier is a heuristic and
# a smaller ceiling stops misclassifications from dominating the cost.
_WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {
    "queue": (0.0, 1.0),
    "prefix_miss": (0.0, 1.0),
    "memory": (0.0, 1.0),
    "jitter": (0.0, 1.0),
    "thermal": (0.0, 1.0),
    "nu": (0.0, 5.0),
}


@app.post("/v1/scheduler/weights")
async def setSchedulerWeights(payload: dict[str, float]) -> dict[str, float]:
    """Hot-swap one or more cost-function weights without a process restart.

    Accepts a partial JSON body — only the keys present are updated. The base
    5 weights must be floats in ``[0.0, 1.0]``; ``nu`` (RouteLLM quality knob)
    is allowed up to ``5.0``. Returns the post-update snapshot.
    """
    _requireActive()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object of weights")
    allowed = set(scheduler_module.WEIGHTS.as_dict().keys())
    updates: dict[str, float] = {}
    for key, raw_value in payload.items():
        if key not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown weight '{key}'. Allowed: {sorted(allowed)}",
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Weight '{key}' must be a number (got {raw_value!r})",
            ) from e
        lo, hi = _WEIGHT_BOUNDS.get(key, (0.0, 1.0))
        if not lo <= value <= hi:
            raise HTTPException(
                status_code=400,
                detail=f"Weight '{key}' must be between {lo} and {hi} (got {value})",
            )
        updates[key] = value
    if updates:
        scheduler_module.WEIGHTS.update(**updates)
        logger.info("Scheduler weights updated | {}", updates)
    return scheduler_module.WEIGHTS.as_dict()


@app.get("/v1/scheduler/worker_weights")
async def getWorkerWeightOverrides() -> dict[str, float]:
    """Return the active per-worker preference overrides keyed by node id."""
    _requireActive()
    return dict(scheduler_module.WORKER_WEIGHT_OVERRIDES)


@app.post("/v1/scheduler/worker_weights")
async def setWorkerWeightOverrides(payload: dict[str, Any]) -> dict[str, float]:
    """Replace one or more per-worker preference overrides without a restart.

    Accepts a partial JSON object keyed by node id. Each value must be a number
    in ``[-1.0, 1.0]`` or ``null`` to clear the override for that worker.
    Returns the post-update override map.
    """
    _requireActive()
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Body must be a JSON object of node_id -> weight",
        )
    for node_id, raw_value in payload.items():
        if not isinstance(node_id, str) or not node_id:
            raise HTTPException(status_code=400, detail=f"Invalid node id: {node_id!r}")
        if raw_value is None:
            scheduler_module.WORKER_WEIGHT_OVERRIDES.pop(node_id, None)
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Weight for '{node_id}' must be a number (got {raw_value!r})",
            ) from e
        if not -1.0 <= value <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"Weight for '{node_id}' must be between -1.0 and 1.0 (got {value})",
            )
        scheduler_module.WORKER_WEIGHT_OVERRIDES[node_id] = value
    overrides = dict(scheduler_module.WORKER_WEIGHT_OVERRIDES)
    logger.info("Worker weight overrides updated | {}", overrides)
    return overrides


async def _health_monitor_loop() -> None:
    """Probe every worker's /health on a fixed cadence and cache the result.

    Requests read this cache (see :func:`_cached_health`) instead of probing
    /health inline, so a burst of N requests no longer fans out into
    ``workers * N`` concurrent probes — which under load timed workers out and
    dropped them from scheduling. The probe rate is fixed at
    ``workers / HEALTH_PROBE_INTERVAL_SECONDS`` regardless of request rate.
    """
    global _health_cache
    while True:
        try:
            if membershipClient is not None:
                workers = await fetch_worker_snapshots(membershipClient)
                states = await _worker_health_states(workers)
                now = time.monotonic()
                _health_cache = {node_id: (now, state) for node_id, state in states.items()}
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Health monitor loop error")
        await asyncio.sleep(HEALTH_PROBE_INTERVAL_SECONDS)


def _cached_health(node_id: str) -> dict[str, Any]:
    """Return the cached health for a worker, or unknown if missing/stale."""
    entry = _health_cache.get(node_id)
    if entry is None:
        return _unknown_worker_health("not yet probed")
    ts, state = entry
    if time.monotonic() - ts > HEALTH_STALE_AFTER_SECONDS:
        return _unknown_worker_health("health stale")
    return state


async def _ready_worker_snapshots(workers: Sequence[WorkerSnapshot]) -> list[WorkerSnapshot]:
    """Return workers whose cached health reports an active inference engine.

    Reads the background health cache, not an inline probe — so this is cheap and
    constant regardless of how many requests are in flight.
    """
    return [worker for worker in workers if _cached_health(worker.node_id).get("healthy") is True]


async def _worker_health_states(workers: Sequence[WorkerSnapshot]) -> dict[str, dict[str, Any]]:
    """Fetch worker health and normalize it for scheduling and dashboards."""
    if not workers:
        return {}
    timeout = httpx.Timeout(WORKER_HEALTH_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(
            *(_worker_health_state(client, worker) for worker in workers),
            return_exceptions=True,
        )
    health_by_node: dict[str, dict[str, Any]] = {}
    for worker, result in zip(workers, results, strict=False):
        if isinstance(result, Exception):
            logger.warning(
                "Skipping unhealthy worker | worker={} address={} error={}",
                worker.node_id,
                worker.address,
                result,
            )
            health_by_node[worker.node_id] = _unknown_worker_health(str(result))
        else:
            health_by_node[worker.node_id] = cast(dict[str, Any], result)
    return health_by_node


async def _worker_health_state(
    client: httpx.AsyncClient,
    worker: WorkerSnapshot,
) -> dict[str, Any]:
    """Return normalized health for one worker."""
    try:
        response = await client.get(f"{worker.address}/health")
    except httpx.RequestError as e:
        logger.warning(
            "Skipping unreachable worker | worker={} address={} error={}",
            worker.node_id,
            worker.address,
            e,
        )
        return _unknown_worker_health(str(e))

    try:
        payload = response.json()
    except ValueError:
        logger.warning("Skipping worker with invalid health JSON | worker={}", worker.node_id)
        return _unknown_worker_health("invalid health JSON", status_code=response.status_code)

    if not isinstance(payload, dict):
        return _unknown_worker_health("invalid health payload", status_code=response.status_code)

    body = payload.get("detail") if response.status_code >= 400 else payload
    if not isinstance(body, dict):
        body = payload
    engine = body.get("engine") if isinstance(body, dict) else None
    if not isinstance(engine, dict) or engine.get("status") is not True:
        detail = engine.get("detail") if isinstance(engine, dict) else "missing engine health"
        detail_text = str(detail)
        logger.warning(
            "Skipping worker with inactive engine | worker={} detail={}",
            worker.node_id,
            detail_text,
        )
        return {
            "healthy": False,
            "status_code": response.status_code,
            "detail": detail_text,
            "engine": engine if isinstance(engine, dict) else _unknown_engine_health(detail_text),
            "body": body,
        }
    return {
        "healthy": response.status_code == 200,
        "status_code": response.status_code,
        "detail": "ok" if response.status_code == 200 else response.text[:200],
        "engine": engine,
        "body": body,
    }


def _unknown_worker_health(
    detail: str = "unknown",
    status_code: int | None = None,
) -> dict[str, Any]:
    """Return a normalized unhealthy worker health payload."""
    return {
        "healthy": False,
        "status_code": status_code,
        "detail": detail,
        "engine": _unknown_engine_health(detail),
        "body": {},
    }


def _unknown_engine_health(detail: str) -> dict[str, Any]:
    return {"status": False, "detail": detail, "engine": "unknown"}


@app.get("/v1/messages", response_model=dict[str, Any])
async def listMessages() -> dict[str, Any]:
    """Return all stored chat requests from etcd; only the active coordinator serves this."""
    _requireActive()
    assert membershipClient is not None
    raw = await membershipClient.getByPrefix("/relay/messages/")
    result: dict[str, Any] = {}
    for key, value in raw.items():
        result[key.removeprefix("/relay/messages/")] = json.loads(value)
    logger.debug("Listed messages | count={}", len(result))
    return result


@app.get("/v1/messages/{msgId}", response_model=dict[str, Any])
async def getMessage(msgId: str) -> dict[str, Any]:
    """Retrieve a single stored chat request by ID; only the active coordinator serves this."""
    _requireActive()
    assert membershipClient is not None
    value = await membershipClient.get(f"/relay/messages/{msgId}")
    if value is None:
        raise HTTPException(status_code=404, detail=f"Message {msgId} not found")
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise HTTPException(status_code=500, detail=f"Message {msgId} is not a JSON object")
    return {str(key): item for key, item in obj.items()}


class WorkerUnreachable(Exception):
    """Raised when a worker cannot be reached at the network level.

    Distinct from an HTTP error response: this means the worker either refused
    the TCP connection, timed out, was unresolvable, etc. Callers should treat
    this as "try a different worker" — not a client-visible error in itself.
    """

    def __init__(self, node_id: str, address: str, cause: Exception) -> None:
        super().__init__(f"{node_id} ({address}): {cause}")
        self.node_id = node_id
        self.address = address
        self.cause = cause


@dataclass(frozen=True)
class OpenedStream:
    """Successfully-opened upstream stream ready to forward to the client."""

    client: httpx.AsyncClient
    stream_ctx: AbstractAsyncContextManager[httpx.Response]
    upstream: httpx.Response


async def _openWorkerStream(worker: WorkerSnapshot, body: dict[str, object]) -> OpenedStream:
    """Open an SSE stream to ``worker``; raise WorkerUnreachable on connect failure."""
    url = f"{worker.address}/v1/chat/completions"
    timeout = httpx.Timeout(timeout=None, connect=10.0)
    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = client.stream("POST", url, json=body)
    try:
        upstream = await stream_ctx.__aenter__()
    except httpx.RequestError as e:
        await client.aclose()
        raise WorkerUnreachable(worker.node_id, worker.address, e) from e

    if upstream.status_code >= 400:
        # The worker responded — this is its considered answer, not a transport
        # failure. Pass it through as-is; do NOT retry on another worker.
        error_body = (await upstream.aread()).decode("utf-8", errors="replace")
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()
        raise HTTPException(
            status_code=upstream.status_code,
            detail=error_body[:500],
        )
    return OpenedStream(client=client, stream_ctx=stream_ctx, upstream=upstream)


async def _open_stream_with_retry(
    body: dict[str, object],
    workers: Sequence[WorkerSnapshot],
    open_fn: Callable[[WorkerSnapshot], Awaitable[OpenedStream]],
    max_attempts: int,
) -> tuple[WorkerChoice, OpenedStream, int]:
    """Schedule a worker and open its stream; retry on connect-level failure.

    Failed workers are excluded from subsequent ``choose_worker`` calls so the
    scheduler picks the next-best from the survivors. Stops when a stream opens,
    when no eligible workers remain, or when ``max_attempts`` is reached.

    Returns ``(choice, opened, attempt_count)`` where ``attempt_count`` is 1 for
    success on the first try. Raises :class:`WorkerUnreachable` if every tried
    worker failed, or :class:`SchedulingError` if no candidate was eligible to
    begin with.
    """
    tried: set[str] = set()
    last_error: WorkerUnreachable | None = None
    for attempt in range(1, max_attempts + 1):
        candidates = [worker for worker in workers if worker.node_id not in tried]
        # Queue signal is coordinator-owned: overwrite the worker-published qw
        # with our live in-flight count. Read synchronously here so concurrent
        # requests observe each other's reservations and don't stampede one
        # worker; keep read -> choose -> reserve free of awaits.
        for worker in candidates:
            worker.telemetry.qw = inflight.depth(worker.node_id)
        try:
            choice = choose_worker(body, candidates)
        except SchedulingError:
            if last_error is not None:
                raise last_error
            raise
        inflight.reserve(choice.worker.node_id)
        try:
            opened = await open_fn(choice.worker)
        except WorkerUnreachable as e:
            # Release the reservation for the worker that failed to open, then
            # trust the worker we just picked (not the exception payload) — the
            # set must reflect what we actually tried.
            inflight.release(choice.worker.node_id)
            tried.add(choice.worker.node_id)
            last_error = e
            logger.warning(
                "Worker unreachable, retrying | worker={} attempt={} cause={}",
                choice.worker.node_id,
                attempt,
                e.cause,
            )
            continue
        return choice, opened, attempt

    assert last_error is not None
    raise last_error


async def _streamWorkerResponse(
    client: httpx.AsyncClient,
    stream_ctx: AbstractAsyncContextManager[httpx.Response],
    upstream: httpx.Response,
    nodeId: str,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        # Release the in-flight reservation only once the request has fully
        # completed — including client disconnect or a mid-stream error, both of
        # which run this finally — never at first byte.
        inflight.release(nodeId)
        await stream_ctx.__aexit__(
            None,
            None,
            None,
        )
        await client.aclose()
