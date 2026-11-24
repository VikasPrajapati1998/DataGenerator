"""Record-level realism checks that go beyond schema validity.

Checks: category-dependent numeric ranges (profiles), city/state
consistency, email format, age vs. birth-date consistency, reference sanity,
and free-text plausibility (not echoing field names, not too short).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from app.generation import geo_data
from app.schemas.input_schema import DatasetSchema, ProfileRange

_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_MIN_TEXT_WORDS = 3


def _as_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        text = str(value)
        return datetime.fromisoformat(text).date() if "T" in text else date.fromisoformat(text)
    except ValueError:
        return None


class RealismValidator:
    """Applies realism heuristics derived from the schema.

    Args:
        schema: The normalised schema.
        profiles: Calibrated ranges for profile fields.
    """

    def __init__(self, schema: DatasetSchema, profiles: dict[str, dict[str, ProfileRange]] | None = None) -> None:
        self.schema = schema
        self.profiles = {**{n: s.profile for n, s in schema.fields.items() if s.profile}, **(profiles or {})}
        fields = schema.fields
        self.city = next((n for n, s in fields.items() if s.generator == "geo_city"), None)
        self.state = next((n for n, s in fields.items() if s.generator == "geo_state"), None)
        self.emails = [n for n, s in fields.items() if s.generator == "email"]
        self.texts = [n for n, s in fields.items() if s.generator == "llm" and s.type == "string"]
        self.age = next((n for n in fields if n.lower() == "age"), None)
        self.birth = next((n for n, s in fields.items() if s.type == "date" and "birth" in n.lower()), None)
        self.references = {n: s.reference for n, s in fields.items() if s.generator == "reference"}

    def check(self, record: dict[str, Any]) -> list[str]:  # noqa: PLR0912
        """Return realism problems for ``record`` (empty when plausible).

        Args:
            record: The record to check.

        Returns:
            Problem descriptions prefixed with ``realism:``.
        """
        issues: list[str] = []
        for name, ranges in self.profiles.items():
            spec = self.schema.fields[name]
            value, category = record.get(name), record.get(spec.depends_on or "")
            if value in (None, "") or category is None:
                continue
            bounds = ranges.get(str(category))
            if bounds is None:
                issues.append(f"realism: {name} has no profile for {spec.depends_on}={category!r}")
            elif not bounds.min - 1e-6 <= float(value) <= bounds.max + 1e-6:
                issues.append(f"realism: {name}={value} outside {bounds.min}..{bounds.max} for {category}")
        if self.city and self.state and record.get(self.city) and record.get(self.state):
            expected = geo_data.state_of(self.schema.locale).get(str(record[self.city]))
            if expected and expected != record[self.state]:
                issues.append(f"realism: city {record[self.city]} is not in {record[self.state]}")
        for name in self.emails:
            value = record.get(name)
            if value not in (None, "") and not _EMAIL.match(str(value)):
                issues.append(f"realism: {name} is not a valid email address")
        if self.age and self.birth and record.get(self.age) not in (None, "") and record.get(self.birth):
            born = _as_date(record[self.birth])
            if born is not None:
                years = (date.today() - born).days / 365.2425
                if abs(years - float(record[self.age])) > 1.01:
                    issues.append(f"realism: {self.age} does not match {self.birth}")
        for name in self.texts:
            value = record.get(name)
            if value in (None, ""):
                continue
            text = str(value)
            if len(text.split()) < _MIN_TEXT_WORDS and (self.schema.fields[name].min_length or 0) == 0:
                issues.append(f"realism: {name} is too short to be informative")
            if text.strip().lower() in (name.lower(), name.replace("_", " ").lower()):
                issues.append(f"realism: {name} just repeats the field name")
        for name, target in self.references.items():
            value = record.get(name)
            if value not in (None, "") and target and str(value) == str(record.get(target)):
                issues.append(f"realism: {name} references the record itself")
        return issues
