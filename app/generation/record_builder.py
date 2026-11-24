"""Deterministic (non-LLM) field generation.

IDs come from deterministic partitioning, names/addresses from Faker,
numbers from explicit statistical distributions and categories from weighted
sampling. Fields marked ``llm`` are left as ``None`` for the LLM enricher.
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
import uuid
from datetime import date, datetime, timedelta
from statistics import NormalDist
from typing import Any

from faker import Faker

from app.core.exceptions import SchemaError
from app.generation import geo_data
from app.generation.links import ExactAssigner, LinkResolver, mix64, resolve_counts, text_key
from app.generation.regex_generator import RegexGenerator
from app.schemas.input_schema import DatasetSchema, FieldSpec, ProfileRange
from app.schemas.schema_parser import case_key, case_specs, dependency_order
from app.utils.safe_eval import RuleExpression

_EMPTY = object()

_TITLES = re.compile(r"^(mr|mrs|ms|miss|dr|prof|shri|smt|sri|kumari)\.?$", re.IGNORECASE)
_DEFAULT_DATE_SPAN_DAYS = 365 * 30


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    return datetime.fromisoformat(text).date() if "T" in text else date.fromisoformat(text)


def _ascii_token(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", normalized.lower())


def _inverse_triangular(p: float, low: float, high: float, mode: float) -> float:
    split = (mode - low) / (high - low)
    if p < split:
        return low + math.sqrt(p * (high - low) * (mode - low))
    return high - math.sqrt((1 - p) * (high - low) * (high - mode))


def _sequence_start(spec: FieldSpec) -> int:
    return int(spec.min) if spec.min is not None else 1


class RecordBuilder:
    """Builds the deterministic part of records for one schema.

    The builder is stateless across records: every call receives its own
    random source and Faker instance, so tasks are independent and
    reproducible from their seed.

    Args:
        schema: The normalised schema.
        total_records: Requested dataset size (bounds for references, exact counts).
        profiles: Calibrated category ranges for ``profile`` fields.
        links: Multi-table context (parent key stores, per-parent counts); ``None`` for a single table.
        seed: Seed of the dataset (exact-count permutations and parent choices).

    Raises:
        SchemaError: If a field cannot be generated as specified.
    """

    def __init__(self, schema: DatasetSchema, total_records: int,
                 profiles: dict[str, dict[str, ProfileRange]] | None = None,
                 links: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.schema = schema
        self.total_records = total_records
        self.seed = seed
        self.order = dependency_order(schema)
        self.profiles = {**{n: s.profile for n, s in schema.fields.items() if s.profile}, **(profiles or {})}
        self._cases = {n: case_specs(n, s) for n, s in schema.fields.items() if s.cases is not None}
        specs = [s for s in schema.fields.values()] + [c for cases in self._cases.values() for c in cases.values()
                                                        if c is not None]
        self._regex = {s.pattern: RegexGenerator(s.pattern) for s in specs if s.generator == "regex" and s.pattern}
        self._formulas = {s.formula: RuleExpression(s.formula) for s in specs if s.formula}
        self._exact: dict[int, ExactAssigner] = {}
        self.links = LinkResolver(links, seed, total_records) if links else None
        self._name_field = next((n for n, s in schema.fields.items()
                                 if s.generator == "faker" and s.faker_method == "name"), None)
        self._state_field = next((n for n, s in schema.fields.items() if s.generator == "geo_state"), None)
        probe = Faker(schema.locale)
        for name, spec in schema.fields.items():
            if spec.generator == "faker" and not callable(getattr(probe, spec.faker_method or "", None)):
                raise SchemaError(f"Field {name!r}: Faker has no provider {spec.faker_method!r}")
            if spec.generator == "range" and spec.distribution is None and (spec.min is None or spec.max is None):
                raise SchemaError(f"Field {name!r}: numeric range needs min and max or a distribution")
            if spec.generator == "profile" and name not in self.profiles:
                continue  # calibrated later by the LLM; checked by require_profiles()

    def require_profiles(self) -> None:
        """Ensure every profile field has calibrated ranges.

        Raises:
            SchemaError: If a profile field has no ranges.
        """
        missing = [n for n in self.schema.profile_fields if n not in self.profiles]
        if missing:
            raise SchemaError(f"Profile fields {missing} have no calibrated ranges")

    def build(self, index: int, rng: random.Random, faker: Faker, attempt: int = 0) -> dict[str, Any]:
        """Build one record's deterministic fields.

        Args:
            index: Zero-based global position of the record in the dataset.
            rng: Random source dedicated to this record attempt.
            faker: Faker instance seeded for this task.
            attempt: Regeneration attempt (0 for the first try); used to
                diversify values that collided with existing ones.

        Returns:
            Record in column order; ``llm`` fields are ``None``.
        """
        values: dict[str, Any] = {}
        for name in self.order:
            spec = self.schema.fields[name]
            if name in self._cases:
                case = self._cases[name].get(case_key(values.get(spec.cases_by)), spec)
                if case is None:
                    values[name] = None
                    continue
                spec = case
            if spec.nullable and spec.null_ratio > 0 and spec.generator not in ("sequence",) \
                    and rng.random() < spec.null_ratio:
                values[name] = None
                continue
            values[name] = self._generate(name, spec, index, values, rng, faker, attempt)
        return {name: values.get(name) for name in self.schema.columns}

    def close(self) -> None:
        """Release the parent key stores (memory maps)."""
        if self.links is not None:
            self.links.close()

    def _linked(self) -> LinkResolver:
        if self.links is None:
            raise SchemaError("parent and child_index fields only work inside a multi-table schema "
                              "(\"tables\": {...} with links)")
        return self.links

    def _exact_value(self, spec: FieldSpec, index: int) -> Any:
        assigner = self._exact.get(id(spec))
        if assigner is None:
            counts = resolve_counts("enum", list(spec.allowed_values or []), spec.counts, spec.weights,
                                    self.total_records)
            key = mix64(self.seed, text_key(json.dumps(spec.allowed_values, default=str)), text_key(str(counts)))
            assigner = self._exact[id(spec)] = ExactAssigner(list(spec.allowed_values or []), counts, key)
        return assigner.value(index)

    def _formula(self, spec: FieldSpec, values: dict[str, Any]) -> Any:
        value = self._formulas[spec.formula].compute(values)  # type: ignore[index]
        if value is None:
            return None
        try:
            if spec.type == "integer":
                return int(round(float(value)))
            if spec.type == "float":
                return round(float(value), spec.decimals if spec.decimals is not None else 2)
            if spec.type == "boolean":
                return bool(value)
            if spec.type in ("date", "datetime"):
                return value.isoformat() if hasattr(value, "isoformat") else str(value)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, str) else str(value)

    # -- per-generator logic -------------------------------------------------------
    def _generate(self, name: str, spec: FieldSpec, index: int, values: dict[str, Any],  # noqa: PLR0911, PLR0912
                  rng: random.Random, faker: Faker, attempt: int) -> Any:
        gen = spec.generator
        if gen == "sequence":
            return _sequence_start(spec) + index
        if gen == "uuid":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        if gen == "faker":
            value = getattr(faker, spec.faker_method)(**spec.faker_args)  # type: ignore[arg-type]
            if isinstance(value, str):
                value = value.replace("\n", ", ")
            if attempt and spec.unique and isinstance(value, str):
                value = f"{value} {rng.randint(2, 99)}"
            return value
        if gen == "enum":
            if spec.counts is not None or spec.exact:
                return self._exact_value(spec, index)
            return rng.choices(spec.allowed_values, weights=spec.weights)[0]  # type: ignore[arg-type]
        if gen == "parent":
            return self._linked().value(spec.link, spec.column, index, values)  # type: ignore[arg-type]
        if gen == "child_index":
            return self._linked().child_position(index) + (int(spec.min) if spec.min is not None else 1)
        if gen == "formula":
            return self._formula(spec, values)
        if gen == "boolean":
            if spec.allowed_values and spec.weights:
                return bool(rng.choices(spec.allowed_values, weights=spec.weights)[0])
            return rng.random() < 0.5
        if gen == "constant":
            return spec.value
        if gen == "range":
            return self._numeric(spec, values, rng)
        if gen == "profile":
            return self._profile(name, spec, values, rng)
        if gen == "date_range":
            return self._temporal(spec, values, rng)
        if gen == "email":
            return self._email(values, rng, faker, attempt)
        if gen == "geo_state":
            weights = geo_data.states(self.schema.locale)
            return rng.choices(list(weights), weights=list(weights.values()))[0]
        if gen == "geo_city":
            rows = geo_data.cities(self.schema.locale)
            state = values.get(self._state_field) if self._state_field else None
            if state is not None:
                rows = tuple(r for r in rows if r[1] == state) or rows
            return rng.choices([r[0] for r in rows], weights=[r[2] for r in rows])[0]
        if gen == "geo_country":
            return faker.country()
        if gen == "reference":
            return self._reference(spec, index, rng)
        if gen == "regex":
            return self._regex[spec.pattern].generate(rng)  # type: ignore[index]
        return None  # llm fields are filled by the enricher

    def _percentile(self, spec: FieldSpec, values: dict[str, Any], rng: random.Random) -> float:
        """Return a [0,1) percentile, blended with a correlated driver field."""
        u = rng.random()
        corr = spec.correlate_with
        if corr is None or values.get(corr.field) is None:
            return u
        driver = self.schema.fields[corr.field]
        if driver.generator == "profile" and corr.field in self.profiles:
            ranges = self.profiles[corr.field].values()
            low, high = min(r.min for r in ranges), max(r.max for r in ranges)
        else:
            low, high = self._bounds(driver)
        if high <= low:
            return u
        p_driver = min(max((float(values[corr.field]) - low) / (high - low), 0.0), 1.0)
        noise = rng.gauss(0, 0.08)
        return min(max(corr.strength * p_driver + (1 - corr.strength) * u + noise, 0.001), 0.999)

    @staticmethod
    def _bounds(spec: FieldSpec) -> tuple[float, float]:
        dist = spec.distribution
        if spec.min is not None and spec.max is not None:
            return float(spec.min), float(spec.max)
        if dist and dist.mean is not None and dist.std:
            return dist.mean - 3 * dist.std, dist.mean + 3 * dist.std
        return 0.0, 1.0

    def _finish_number(self, spec: FieldSpec, value: float) -> int | float:
        if spec.min is not None:
            value = max(value, float(spec.min))
        if spec.max is not None:
            value = min(value, float(spec.max))
        if spec.type == "integer":
            return int(round(value))
        return round(value, spec.decimals if spec.decimals is not None else 2)

    def _numeric(self, spec: FieldSpec, values: dict[str, Any], rng: random.Random) -> int | float:
        dist = spec.distribution
        low, high = self._bounds(spec)
        p = self._percentile(spec, values, rng)
        if dist is None or dist.kind == "uniform":
            if spec.type == "integer":
                return self._finish_number(spec, math.floor(low + p * (high - low + 1)))
            return self._finish_number(spec, low + p * (high - low))
        if dist.kind == "normal":
            mean = dist.mean if dist.mean is not None else (low + high) / 2
            std = dist.std or (high - low) / 6
            return self._finish_number(spec, NormalDist(mean, std).inv_cdf(p))
        if dist.kind == "lognormal":
            mu = dist.mean if dist.mean is not None else 0.0
            sigma = dist.std or 0.5
            return self._finish_number(spec, math.exp(NormalDist(mu, sigma).inv_cdf(p)))
        mode = dist.mode if dist.mode is not None else (low + high) / 2
        return self._finish_number(spec, _inverse_triangular(p, low, high, mode))

    def _profile(self, name: str, spec: FieldSpec, values: dict[str, Any], rng: random.Random) -> int | float | None:
        category = values.get(spec.depends_on or "")
        if category is None:
            return None
        ranges = self.profiles[name][str(category)]
        mode = ranges.typical if ranges.typical is not None else (ranges.min + ranges.max) / 2
        value = _inverse_triangular(self._percentile(spec, values, rng), ranges.min, ranges.max, mode)
        if spec.type == "integer":
            step = 100 if ranges.max >= 10_000 else 1
            rounded = int(round(value / step) * step)
            return min(max(rounded, int(math.ceil(ranges.min))), int(ranges.max))
        return round(value, spec.decimals if spec.decimals is not None else 2)

    def _temporal(self, spec: FieldSpec, values: dict[str, Any], rng: random.Random) -> str | None:
        if spec.depends_on and spec.offset_days:
            base = values.get(spec.depends_on)
            if base is None:
                return None
            low, high = spec.offset_days
            result = _to_date(base) + timedelta(days=rng.randint(low, high))
            if spec.max is not None:
                result = min(result, _to_date(spec.max))
            return result.isoformat()
        today = date.today()
        start = _to_date(spec.min) if spec.min is not None else today - timedelta(days=_DEFAULT_DATE_SPAN_DAYS)
        end = _to_date(spec.max) if spec.max is not None else today
        span = max((end - start).days, 0)
        day = start + timedelta(days=rng.randint(0, span))
        if spec.type == "datetime":
            moment = datetime(day.year, day.month, day.day) + timedelta(seconds=rng.randint(0, 86_399))
            return moment.isoformat(timespec="seconds")
        return day.isoformat()

    def _email(self, values: dict[str, Any], rng: random.Random, faker: Faker, attempt: int) -> str:
        domain = faker.free_email_domain()
        full_name = values.get(self._name_field) if self._name_field else None
        tokens = [_ascii_token(t) for t in str(full_name or "").split() if not _TITLES.match(t)]
        tokens = [t for t in tokens if t]
        digits = str(rng.choice((rng.randint(1, 99), rng.randint(70, 99) + 1900, rng.randint(100, 9999))))
        if len(tokens) >= 2:
            first, last = tokens[0], tokens[-1]
            # Full-name forms are common but collide often at scale; short forms always carry digits.
            local = rng.choices(
                (f"{first}.{last}", f"{first}{last}", f"{first}_{last}", f"{last}.{first}",
                 f"{first}.{last}{digits}", f"{first}{last}{digits}", f"{first[0]}{last}{digits}",
                 f"{first}.{last[0]}{digits}", f"{first}{digits}"),
                weights=(18, 8, 5, 4, 25, 12, 12, 8, 8))[0]
        else:
            local = f"{_ascii_token(faker.user_name()) or 'user'}{digits}"
        if attempt:
            local = f"{local}{rng.randint(10, 9999)}"
        return f"{local}@{domain}"

    def _reference(self, spec: FieldSpec, index: int, rng: random.Random) -> int | None:
        target = self.schema.fields[spec.reference]  # type: ignore[index]
        start = _sequence_start(target)
        own = start + index
        if spec.nullable:
            if index == 0:
                return None
            return rng.randint(start, own - 1)
        if self.total_records < 2:
            raise SchemaError("A non-nullable reference needs at least two records")
        candidate = rng.randint(start, start + self.total_records - 2)
        return candidate + 1 if candidate >= own else candidate
