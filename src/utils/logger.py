"""Structured logging setup for command-line pipeline runs."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.utils.config import resolve_project_path

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "run_id=%(run_id)s | dataset=%(dataset)s | stage=%(stage)s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
LOGGER_ROOT_NAME = "data_validation"
STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}
LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("mlops_log_context", default={})


class ContextDefaultsFilter(logging.Filter):
    """Ensure contextual fields exist for every formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = LOG_CONTEXT.get()
        for key, default in {"run_id": "-", "dataset": "-", "stage": "-"}.items():
            if not hasattr(record, key):
                setattr(record, key, context.get(key, default))
        return True


class JsonLogFormatter(logging.Formatter):
    """JSON Lines formatter for machine-readable CI and audit logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
            "run_id": getattr(record, "run_id", "-"),
            "dataset": getattr(record, "dataset", "-"),
            "stage": getattr(record, "stage", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_RECORD_FIELDS and key not in payload:
                payload[key] = _json_safe(value)
        return json.dumps(payload, sort_keys=True, default=str)


def setup_logger(
    name: str = LOGGER_ROOT_NAME,
    log_dir: str | Path = "logs",
    level: str = "INFO",
    log_file: str = "data_validation.log",
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
    json_log_file: str | None = "data_validation.jsonl",
    enable_console: bool = True,
) -> logging.Logger:
    """Create rotating text and JSON log handlers without duplicate handlers."""

    resolved_log_dir = resolve_project_path(log_dir)
    configured_level = getattr(logging, level.upper(), logging.INFO)
    max_bytes = _positive_int(max_bytes, default=2_000_000)
    backup_count = _non_negative_int(backup_count, default=5)
    signature = (
        str(resolved_log_dir),
        configured_level,
        log_file,
        max_bytes,
        backup_count,
        json_log_file,
        enable_console,
    )
    logger = logging.getLogger(name)
    logger.setLevel(configured_level)
    logger.propagate = False

    if getattr(logger, "_mlops_signature", None) == signature:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    context_filter = ContextDefaultsFilter()

    if enable_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(configured_level)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(context_filter)
        logger.addHandler(stream_handler)

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        resolved_log_dir / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(configured_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)
    logger.addHandler(file_handler)

    if json_log_file:
        json_handler = RotatingFileHandler(
            resolved_log_dir / json_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        json_handler.setLevel(configured_level)
        json_handler.setFormatter(JsonLogFormatter(datefmt=LOG_DATE_FORMAT))
        json_handler.addFilter(context_filter)
        logger.addHandler(json_handler)

    logger._mlops_signature = signature  # type: ignore[attr-defined]
    return logger


def get_logger(component: str) -> logging.Logger:
    """Return a child logger that propagates into the configured validation logger."""

    normalized = component.removeprefix("src.validation.").removeprefix("src.utils.")
    normalized = normalized.replace("__main__", "cli").strip(".")
    if not normalized or normalized == LOGGER_ROOT_NAME:
        return logging.getLogger(LOGGER_ROOT_NAME)
    if normalized.startswith(f"{LOGGER_ROOT_NAME}."):
        logger = logging.getLogger(normalized)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
        return logger
    logger = logging.getLogger(f"{LOGGER_ROOT_NAME}.{normalized}")
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Temporarily attach structured values to all validation log records."""

    current = LOG_CONTEXT.get()
    merged = {**current, **{key: value for key, value in values.items() if value is not None}}
    token: Token[dict[str, Any]] = LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        LOG_CONTEXT.reset(token)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _positive_int(value: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default
