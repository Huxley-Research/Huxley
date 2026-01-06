"""
Structured logging for Huxley.

Provides consistent, structured logging across all components with:
- JSON and console output formats
- Automatic context binding
- Request/execution tracing
- Performance metrics
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import Processor

from huxley.core.config import get_config

# Track if we've configured logging
_logging_configured = False

# Log level mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _get_default_log_level() -> int:
    """Get log level from environment, defaulting to INFO."""
    env_level = os.environ.get("HUXLEY_LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(env_level, logging.INFO)


def _ensure_default_config() -> None:
    """Ensure structlog has a minimal default configuration."""
    global _logging_configured
    if _logging_configured:
        return
    
    # Set up minimal config that respects HUXLEY_LOG_LEVEL
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_get_default_log_level()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,  # Allow reconfiguration
    )
    _logging_configured = True


def configure_logging() -> None:
    """Configure structured logging based on current configuration."""
    global _logging_configured
    config = get_config()

    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.logging.format == "json":
        # JSON output for production
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console output for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            LOG_LEVELS.get(config.logging.level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _logging_configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically module name)
        **initial_context: Initial context to bind to all log entries

    Returns:
        A bound structured logger
    """
    # Ensure default configuration respects log level
    _ensure_default_config()
    
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def bind_context(**context: Any) -> None:
    """
    Bind context variables to all subsequent log entries in this context.

    Useful for request-scoped logging (e.g., execution_id, user_id).
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
