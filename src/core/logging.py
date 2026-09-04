"""Centralised logging configuration.

Uses :mod:`structlog` on top of the stdlib so that:

* every log line is structured (key/value) and consistently timestamped;
* an ``execution_id`` / ``node_id`` can be bound once and appear on every
  subsequent line (see :func:`bind_execution`);
* secrets are never rendered -- callers must pass already-sanitised values.

Call :func:`configure_logging` exactly once at process start.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", *, json_output: bool | None = None) -> None:
    """Configure stdlib + structlog.

    Parameters
    ----------
    level:
        Standard logging level name (``"INFO"``, ``"DEBUG"`` ...).
    json_output:
        Force JSON (``True``) or console (``False``) rendering.  When
        ``None`` (default) JSON is used only when stdout is not a TTY.
    """
    global _CONFIGURED

    level_no = logging.getLevelName(level.upper())
    if not isinstance(level_no, int):
        level_no = logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level_no,
    )

    if json_output is None:
        json_output = not sys.stdout.isatty()

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structured logger, configuring logging lazily if needed."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


def bind_execution(**kwargs: Any) -> None:
    """Bind key/value pairs (e.g. ``execution_id=...``) to the current context.

    Every log line emitted afterwards on this task/thread carries them.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_execution() -> None:
    """Drop everything previously bound via :func:`bind_execution`."""
    structlog.contextvars.clear_contextvars()
