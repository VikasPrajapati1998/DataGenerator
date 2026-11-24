"""Centralised logging for the whole project.

Log files are organised as ``logs/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS.log``. A new
file is created per run (never overwritten). Worker processes append to the
parent's file by calling :func:`configure_logging` with ``log_file``.

Secrets (API keys, bearer tokens) are redacted from every record.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "sde"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(processName)s/%(threadName)s | %(name)s | %(message)s"
_lock = threading.Lock()
_configured_file: Path | None = None

_SECRET_PATTERNS = (
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|secret|authorization)\s*[=:]\s*['\"]?)[^\s'\",}]+", re.IGNORECASE),
    re.compile(r"\b(nvapi-|sk-)[A-Za-z0-9_\-]{8,}"),
)


class SecretRedactingFilter(logging.Filter):
    """Masks credentials in log messages before they are emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets in ``record`` in place.

        Args:
            record: The log record about to be emitted.

        Returns:
            Always ``True`` so the record is still logged.
        """
        message = record.getMessage()
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class ConsoleFormatter(logging.Formatter):
    """Console format without stack traces: one readable line per event.

    Tracebacks (and notes libraries attach to exceptions, such as LangGraph's
    "During task with name ...") still go to the log file in full. Set
    ``SDE_DEBUG=1`` (CLI: ``--debug``) to see them on the console too.
    """

    def __init__(self, fmt: str, show_tracebacks: bool = False) -> None:
        super().__init__(fmt)
        self.show_tracebacks = show_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record``, leaving out exception details unless tracebacks are enabled."""
        if self.show_tracebacks or not (record.exc_info or record.stack_info):
            return super().format(record)
        saved = record.exc_info, record.exc_text, record.stack_info
        record.exc_info = record.exc_text = record.stack_info = None
        try:
            line = super().format(record)
        finally:
            record.exc_info, record.exc_text, record.stack_info = saved
        return f"{line} (details in {current_log_file()})" if record.levelno >= logging.ERROR else line


class _CurrentStderrHandler(logging.StreamHandler):
    """Console handler that always writes to the *current* ``sys.stderr``.

    A plain ``StreamHandler`` keeps the stream object it was created with. When
    that object is later closed or replaced (pytest output capture, redirected
    consoles, background threads outliving a test), logging fails with
    "I/O operation on closed file". Looking ``sys.stderr`` up on every write
    avoids that.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):  # noqa: ANN201 - matches logging.StreamHandler.stream
        """The current ``sys.stderr``."""
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # noqa: ANN001
        """Ignore assignments; the stream is always ``sys.stderr``."""


def redact(text: str) -> str:
    """Return ``text`` with anything that looks like a credential masked.

    Args:
        text: Arbitrary text that may contain secrets.

    Returns:
        The text with secret values replaced by ``***``.
    """
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + "***", text)
        else:
            text = pattern.sub("***", text)
    return text


def _new_log_file(log_root: Path) -> Path:
    now = datetime.now()
    day_dir = log_root / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    base = now.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = day_dir / f"{base}.log"
    counter = 1
    while candidate.exists():
        candidate = day_dir / f"{base}_{counter}.log"
        counter += 1
    return candidate


def configure_logging(
    log_root: Path | None = None,
    level: str | int | None = None,
    log_file: Path | None = None,
    console: bool = True,
) -> Path:
    """Configure the project logger once per process.

    Args:
        log_root: Root ``logs`` directory; defaults to ``$SDE_LOG_DIR`` or ``./logs``.
        level: Log level name or number; defaults to ``$SDE_LOG_LEVEL`` or INFO.
        log_file: Existing file to append to (used by worker processes).
        console: Whether to also log to stderr.

    Returns:
        The path of the log file in use.
    """
    global _configured_file  # pylint: disable=global-statement
    with _lock:
        if _configured_file is not None:
            return _configured_file
        root = Path(log_root or os.getenv("SDE_LOG_DIR", "logs"))
        target = log_file or _new_log_file(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_level = level or os.getenv("SDE_LOG_LEVEL", "INFO")

        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(resolved_level)
        logger.propagate = False
        logger.handlers.clear()
        formatter = logging.Formatter(_FORMAT)
        redactor = SecretRedactingFilter()

        file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        logger.addHandler(file_handler)

        if console:
            console_handler = _CurrentStderrHandler()
            console_handler.setFormatter(ConsoleFormatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                show_tracebacks=os.getenv("SDE_DEBUG", "").lower() in ("1", "true", "yes")))
            console_handler.addFilter(redactor)
            logger.addHandler(console_handler)

        # Third-party libraries log through the root logger; keep them quiet.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("faker").setLevel(logging.WARNING)
        _configured_file = target
        return target


def current_log_file() -> Path | None:
    """Return the log file configured in this process, if any."""
    return _configured_file


def get_logger(name: str) -> logging.Logger:
    """Return a child of the project logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A logger that inherits the central configuration.
    """
    short = name.removeprefix("app.")
    return logging.getLogger(f"{LOGGER_NAME}.{short}")
