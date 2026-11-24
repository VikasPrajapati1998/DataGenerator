"""Independent check of a generated dataset against its request file (standard library only).

Usage::

    python scripts/check_output.py samples/schemas/healthcare_appointments.json path/to/generated_dataset.csv

Errors (exit code 1): duplicate unique values, whole-record duplicates, nulls in required fields,
values outside allowed values / min-max / profile ranges / regex patterns, unparseable numbers or
dates, dependent dates outside their ``offset_days``, dangling references and failed schema rules
written as simple comparisons. Notes (never fail): repeated LLM text, dates in the future.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

_EMPTY = {"", "None", "null", "nan", "NaN"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    def text(value: Any) -> str:
        return "" if value is None else str(value)

    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [{k: text(v) for k, v in json.loads(line).items()} for line in handle if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("records", data) if isinstance(data, dict) else data
    return [{k: text(v) for k, v in row.items()} for row in rows]


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _day(value: str) -> date | None:
    try:
        return datetime.fromisoformat(value).date() if "T" in value or " " in value else date.fromisoformat(value)
    except ValueError:
        return None


def _bool_text(value: Any) -> str:
    return str(value)


def check(schema_file: Path, data_file: Path) -> tuple[list[str], list[str]]:
    """Return ``(errors, notes)`` for one dataset."""
    request = json.loads(schema_file.read_text(encoding="utf-8"))
    fields: dict[str, dict[str, Any]] = request["schema"].get("fields") or {}
    rows = _load_rows(data_file)
    errors: list[str] = []
    notes: list[str] = []

    def err(message: str, count: int) -> None:
        if count:
            errors.append(f"{message}: {count}")

    if not rows:
        return ["dataset is empty"], notes
    columns = list(rows[0])
    id_fields = {k for k, f in fields.items() if f.get("generator") in ("sequence", "uuid")}
    err("whole-row duplicates", len(rows) - len({tuple(r.values()) for r in rows}))
    if id_fields:
        rest = [c for c in columns if c not in id_fields]
        dupes = len(rows) - len({tuple(r[c] for c in rest) for r in rows})
        notes.append(f"rows identical apart from {sorted(id_fields)}: {dupes}") if dupes else None

    by_field = {name: [r.get(name, "") for r in rows] for name in columns}
    for name, spec in fields.items():
        if name not in by_field:
            errors.append(f"column {name} missing from the output")
            continue
        values = by_field[name]
        present = [(i, v) for i, v in enumerate(values) if v not in _EMPTY]
        if spec.get("unique"):
            err(f"{name}: duplicate values", len(present) - len({v for _, v in present}))
        if not spec.get("nullable") and spec.get("required", True):
            err(f"{name}: empty values in a required field", len(values) - len(present))
        kind = spec.get("type", "string")
        if spec.get("allowed_values") and kind != "boolean":
            allowed = {_bool_text(a) for a in spec["allowed_values"]} | {
                str(float(a)) for a in spec["allowed_values"] if isinstance(a, (int, float)) and not isinstance(a, bool)}
            err(f"{name}: values outside allowed_values", sum(v not in allowed for _, v in present))
        if spec.get("pattern"):
            pattern = re.compile(spec["pattern"])
            err(f"{name}: values not matching the pattern", sum(not pattern.fullmatch(v) for _, v in present))
        if kind in ("integer", "float"):
            nums = [(i, _num(v)) for i, v in present]
            err(f"{name}: not a number", sum(n is None for _, n in nums))
            nums = [(i, n) for i, n in nums if n is not None]
            if kind == "integer":
                err(f"{name}: not a whole number", sum(n % 1 != 0 for _, n in nums))
            if spec.get("min") is not None:
                err(f"{name}: below min {spec['min']}", sum(n < float(spec["min"]) for _, n in nums))
            if spec.get("max") is not None:
                err(f"{name}: above max {spec['max']}", sum(n > float(spec["max"]) for _, n in nums))
            if spec.get("profile") and spec.get("depends_on"):
                prof, dep = spec["profile"], by_field[spec["depends_on"]]
                err(f"{name}: outside the profile range of its {spec['depends_on']}",
                    sum(dep[i] in prof and not prof[dep[i]]["min"] <= n <= prof[dep[i]]["max"] for i, n in nums))
        if kind in ("date", "datetime"):
            days = [(i, _day(v)) for i, v in present]
            err(f"{name}: unparseable date", sum(d is None for _, d in days))
            days = [(i, d) for i, d in days if d is not None]
            if spec.get("min"):
                err(f"{name}: before min {spec['min']}", sum(d < _day(str(spec["min"])) for _, d in days))
            if spec.get("max"):
                err(f"{name}: after max {spec['max']}", sum(d > _day(str(spec["max"])) for _, d in days))
            future = sum(d > date.today() for _, d in days)
            notes.append(f"{name}: {future} date(s) after today") if future else None
            if spec.get("offset_days") and spec.get("depends_on"):
                low, high = spec["offset_days"]
                base = by_field[spec["depends_on"]]
                bad = 0
                for i, d in days:
                    b = _day(base[i]) if base[i] not in _EMPTY else None
                    if b is not None and not low <= (d - b).days <= high and not (spec.get("max") and d == _day(spec["max"])):
                        bad += 1
                err(f"{name}: offset from {spec['depends_on']} outside {low}..{high} days", bad)
        if spec.get("generator") == "reference" and spec.get("reference") in by_field:
            targets = set(by_field[spec["reference"]])
            normalise = lambda v: v[:-2] if v.endswith(".0") else v  # noqa: E731
            err(f"{name}: dangling reference to {spec['reference']}", sum(normalise(v) not in targets for _, v in present))
        if spec.get("generator") == "llm" and kind == "string" and present:
            counts = Counter(v for _, v in present)
            top = counts.most_common(1)[0][1]
            notes.append(f"{name}: {len(counts) / len(present):.1%} distinct text, most repeated x{top}")
    for column in columns:
        if column.endswith("email") or fields.get(column, {}).get("generator") == "email":
            bad = sum(v not in _EMPTY and not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", v)
                      for v in by_field[column])
            err(f"{column}: malformed email", bad)
    return errors, notes


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    errors, notes = check(Path(argv[1]), Path(argv[2]))
    for line in errors:
        print(f"  ERROR {line}")
    for line in notes:
        print(f"  note  {line}")
    print("  OK" if not errors else f"  FAILED ({len(errors)} problem(s))")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
