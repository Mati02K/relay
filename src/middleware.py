"""Shared HTTP middleware for Relay services."""

import time

from fastapi import Request, Response
from loguru import logger


async def log_requests(request: Request, call_next: object) -> Response:
    """Log every incoming HTTP request with response status and duration."""
    start = time.perf_counter()
    response: Response = await call_next(request)  # type: ignore[operator]
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "{} {} -> {} | {:.1f}ms",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response
