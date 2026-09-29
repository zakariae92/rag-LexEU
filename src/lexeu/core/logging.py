"""Structured logging: JSON in prod (machine-parsable), pretty console locally."""

import logging
import sys

import structlog

from lexeu.core.config import Settings


def add_severity(
    _logger: object, _method: str, event: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Cloud Logging reads the level from `severity` (it would show every JSON line as DEFAULT)."""
    event["severity"] = str(event.get("level", "info")).upper()
    return event


def configure_logging(settings: Settings) -> None:
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,  # carries request_id etc.
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if settings.log_json:
        shared.append(add_severity)
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logs (uvicorn, sqlalchemy, ...) through the same pipeline.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())

    # Our middleware already logs one line per request.
    logging.getLogger("uvicorn.access").disabled = True
    # Outgoing HTTP calls (probes, later LLM/embedding APIs) are traced, not logged.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # LiteLLM logs every call at INFO; usage and cost are reported by our own answer log.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
