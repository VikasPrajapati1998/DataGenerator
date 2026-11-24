"""Generic helpers with no project-specific dependencies."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import TypeVar

T = TypeVar("T")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: str | None, default: str = "dataset", max_length: int = 40) -> str:
    """Return ``text`` as a file-system-safe name (letters, digits, ``_`` and ``-``).

    Args:
        text: Raw name such as a schema file stem.
        default: Used when nothing usable remains.
        max_length: Maximum length of the result.

    Returns:
        The sanitised name.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text or "").strip("_-")[:max_length].strip("_-")
    return cleaned or default


def new_job_id(prefix: str | None = None) -> str:
    """Return a sortable, collision-resistant job id, ``<prefix>_YYYYMMDD_HHMMSS_<hex6>``."""
    stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    return f"{safe_name(prefix)}_{stamp}" if prefix else stamp


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of ``items`` of at most ``size`` elements.

    Args:
        items: Sequence to split.
        size: Maximum slice length (must be positive).

    Yields:
        Slices of ``items``.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def format_duration(seconds: float) -> str:
    """Readable duration: ``45s``, ``12m05s``, ``3h20m`` or ``4d07h``."""
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h"


def format_bytes(size: float) -> str:
    """Return a human-readable byte size such as ``12.3 MB``."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
