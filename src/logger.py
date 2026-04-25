import logging
import os
import sys
from pathlib import Path

from loguru import logger

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: str = os.getenv("LOG_FILE", "logs/relay.log")


class _InterceptHandler(logging.Handler):
    """Route all stdlib logging (uvicorn, grpc, etc.) through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib log record into the loguru pipeline."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup() -> None:
    """Configure loguru with a human-readable stdout handler and a JSON rotating file handler."""
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        LOG_FILE,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        serialize=True,  # JSON Lines — one JSON object per log line
        backtrace=True,
        diagnose=True,
        enqueue=True,  # async-safe: writes happen in a background thread
    )

    # Redirect all stdlib loggers (uvicorn, grpc, asyncio) into loguru.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "grpc"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

    logger.info("Logging initialised | level={} file={}", LOG_LEVEL, LOG_FILE)
