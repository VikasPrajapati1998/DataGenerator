"""Reading and writing records as CSV, JSON (array) or JSONL."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from app.core.exceptions import StorageError

SUPPORTED_FORMATS = ("csv", "json", "jsonl")


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_records(handle: IO[str], records: Iterable[dict[str, Any]], columns: list[str], fmt: str) -> int:
    """Write ``records`` to an open text handle in ``fmt``.

    Args:
        handle: Text handle (opened with ``newline=""`` for CSV).
        records: Records to write.
        columns: Column order.
        fmt: ``csv``, ``json`` or ``jsonl``.

    Returns:
        Number of records written.

    Raises:
        StorageError: If ``fmt`` is unsupported or a record cannot be serialised.
    """
    try:
        return _write_records(handle, records, columns, fmt)
    except (csv.Error, TypeError, ValueError) as exc:
        raise StorageError(f"Cannot serialise records as {fmt}: {exc}") from exc


def _write_records(handle: IO[str], records: Iterable[dict[str, Any]], columns: list[str], fmt: str) -> int:
    count = 0
    if fmt == "csv":
        writer = csv.writer(handle)
        writer.writerow(columns)
        for record in records:
            writer.writerow([_csv_value(record.get(c)) for c in columns])
            count += 1
    elif fmt == "jsonl":
        for record in records:
            handle.write(json.dumps({c: record.get(c) for c in columns}, ensure_ascii=False, default=str))
            handle.write("\n")
            count += 1
    elif fmt == "json":
        handle.write("[")
        for record in records:
            handle.write(",\n" if count else "\n")
            handle.write(json.dumps({c: record.get(c) for c in columns}, ensure_ascii=False, default=str))
            count += 1
        handle.write("\n]\n" if count else "]\n")
    else:
        raise StorageError(f"Unsupported output format {fmt!r}")
    return count


def read_header(path: Path, fmt: str) -> list[str]:
    """Return the column names of a batch file.

    Raises:
        StorageError: If the file is unreadable or empty.
    """
    try:
        if fmt == "csv":
            with path.open(encoding="utf-8", newline="") as handle:
                return next(csv.reader(handle))
        first = next(iter_records(path, fmt), None)
        return list(first) if first else []
    except (OSError, StopIteration, ValueError) as exc:
        raise StorageError(f"Cannot read header of {path}: {exc}") from exc


def iter_records(path: Path, fmt: str) -> Iterator[dict[str, Any]]:
    """Stream records from a file. CSV empty strings are returned as ``None``.

    Args:
        path: File to read.
        fmt: Its format.

    Yields:
        Records as dictionaries.

    Raises:
        StorageError: If the file is malformed.
    """
    try:
        if fmt == "csv":
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    yield {k: (None if v == "" else v) for k, v in row.items()}
        elif fmt == "jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        elif fmt == "json":
            with path.open(encoding="utf-8") as handle:
                yield from json.load(handle)
        else:
            raise StorageError(f"Unsupported format {fmt!r}")
    except (OSError, ValueError, csv.Error) as exc:
        raise StorageError(f"Cannot read records from {path}: {exc}") from exc


def count_records(path: Path, fmt: str) -> int:
    """Count the records in a file without materialising them."""
    if fmt == "jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return sum(1 for _ in iter_records(path, fmt))
