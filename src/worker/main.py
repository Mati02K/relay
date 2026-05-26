import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

import logger as loggerSetup
from worker.daemon import WorkerDaemon


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create the worker daemon singleton for this process."""
    loggerSetup.setup()
    worker = WorkerDaemon.from_env()
    app.state.worker = worker
    await worker.start()
    yield
    await worker.stop()


app = FastAPI(title="Relay Worker", lifespan=lifespan)


def _worker(request: Request) -> WorkerDaemon:
    worker = getattr(request.app.state, "worker", None)
    if not isinstance(worker, WorkerDaemon):
        raise HTTPException(status_code=503, detail="Worker daemon not started")
    return worker


@app.middleware("http")
async def logRequests(request: Request, callNext: object) -> Response:
    """Log every incoming request with response status and duration."""
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


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Return worker daemon and inference engine health."""
    payload = await _worker(request).health()
    engine = payload.get("engine")
    if not isinstance(engine, dict) or engine.get("status") is not True:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/telemetry")
async def telemetry(request: Request) -> dict[str, Any]:
    """Return the latest worker telemetry snapshot."""
    return await _worker(request).telemetry_snapshot()


@app.post("/v1/chat/completions")
async def chatCompletions(request: Request) -> StreamingResponse:
    """Run an OpenAI-compatible chat completion request scheduled by the coordinator."""
    worker = _worker(request)
    body = await request.json()
    try:
        await worker.ensure_ready()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return StreamingResponse(
        worker.generate(body),
        media_type="text/event-stream",
    )
