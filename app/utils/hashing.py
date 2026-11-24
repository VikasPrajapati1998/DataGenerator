"""Stable, process-independent hashing used for deduplication and integrity.

Python's built-in :func:`hash` is salted per process, so it must never be used
for anything that crosses process boundaries or survives a restart.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from app.core.exceptions import StorageError


def normalize_value(value: Any) -> str:
    """Return a canonical text form of ``value`` for comparison.

    Strings are case-folded and whitespace-collapsed so that trivially
    different spellings hash identically.

    Args:
        value: Any JSON-like scalar.

    Returns:
        A canonical string.
    """
    if value is None:
        return "\x00null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(round(value, 6))
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return str(value)


def value_hash(value: Any) -> int:
    """Return a 63-bit stable hash of a single field value.

    Args:
        value: The value to hash.

    Returns:
        A non-negative integer suitable for SQLite INTEGER columns.
    """
    digest = hashlib.blake2b(normalize_value(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def record_hash(record: Mapping[str, Any], fields: Iterable[str]) -> int:
    """Return a stable hash of ``record`` restricted to ``fields``.

    Args:
        record: The record to hash.
        fields: Field names that participate (order-independent).

    Returns:
        A non-negative 63-bit integer.
    """
    canonical = "\x1f".join(f"{name}={normalize_value(record.get(name))}" for name in sorted(fields))
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def stable_seed(*parts: Any) -> int:
    """Derive a deterministic 32-bit seed from arbitrary parts.

    Args:
        *parts: Values identifying the random stream (job seed, task id...).

    Returns:
        An integer in ``[0, 2**32)``.
    """
    digest = hashlib.blake2b(json.dumps(parts, default=str).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, streamed in chunks.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration.

    Returns:
        Hex digest string.

    Raises:
        StorageError: If the file cannot be read.
    """
    sha = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                sha.update(chunk)
    except OSError as exc:
        raise StorageError(f"Cannot read {path} for checksumming: {exc}") from exc
    return sha.hexdigest()
