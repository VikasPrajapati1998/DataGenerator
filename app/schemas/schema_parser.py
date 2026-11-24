"""Convert shorthand or rich user schemas into a :class:`DatasetSchema`.

Shorthand descriptions are interpreted with conservative keyword rules. Any
field whose meaning cannot be derived deterministically becomes an ``llm``
field, so the model only generates what genuinely needs context.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from graphlib import CycleError, TopologicalSorter
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import SchemaError
from app.generation.geo_data import supported_geo_locales
from app.generation.regex_generator import RegexGenerator
from app.monitoring.logger import get_logger
from app.schemas.input_schema import CorrelationSpec, DatasetSchema, FieldSpec, RuleSpec
from app.utils.safe_eval import RuleExpression

logger = get_logger(__name__)

_LOCALE_HINTS: tuple[tuple[str, str], ...] = (
    (r"\bindia|indian\b", "en_IN"),
    (r"\b(us|usa|united states|american)\b", "en_US"),
    (r"\b(uk|united kingdom|british)\b", "en_GB"),
    (r"\b(germany|german)\b", "de_DE"),
    (r"\b(france|french)\b", "fr_FR"),
    (r"\b(canada|canadian)\b", "en_CA"),
    (r"\b(australia|australian)\b", "en_AU"),
)
_RANGE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:and|to|-|–)\s*(-?\d+(?:\.\d+)?)")
_DATE_RANGE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:and|to|–)\s*(\d{4}-\d{2}-\d{2})")
_BASED_ON = re.compile(r"\b(?:based on|depends on|according to|by)\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_NUMERIC_NAMES = ("salary", "price", "amount", "income", "cost", "revenue", "balance", "fee", "wage", "budget")
_DRIVER_NAMES = ("experience_years", "years_of_experience", "experience", "age", "tenure_years")

_FAKER_KEYWORDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("first name", "first_name", "given name"), "first_name", "string"),
    (("last name", "last_name", "surname", "family name"), "last_name", "string"),
    (("full name", "realistic name", "person name", "name"), "name", "string"),
    (("phone", "mobile", "contact number"), "phone_number", "string"),
    (("street address", "address"), "address", "string"),
    (("postcode", "postal code", "zip"), "postcode", "string"),
    (("company", "organisation", "organization", "employer"), "company", "string"),
    (("url", "website"), "url", "string"),
    (("username", "user name", "user_name"), "user_name", "string"),
    (("ipv4", "ip address"), "ipv4", "string"),
)


def detect_locale(*texts: str | None) -> str | None:
    """Return a Faker locale hinted at by any of ``texts``.

    Args:
        *texts: Descriptions or instructions to scan.

    Returns:
        A locale such as ``en_IN``, or ``None`` when nothing matches.
    """
    joined = " ".join(t for t in texts if t).lower()
    for pattern, locale in _LOCALE_HINTS:
        if re.search(pattern, joined):
            return locale
    return None


def _infer_from_text(name: str, text: str, all_names: list[str]) -> dict[str, Any]:  # noqa: PLR0911, PLR0912
    """Infer a partial FieldSpec from a shorthand description."""
    lowered = text.lower().strip()
    lname = name.lower()
    spec: dict[str, Any] = {"description": text}
    if "optional" in lowered:
        spec["required"] = False
        spec["nullable"] = True
    if "nullable" in lowered:
        spec["nullable"] = True
    unique = "unique" in lowered

    if "uuid" in lowered:
        return {**spec, "type": "string", "generator": "uuid", "unique": True}
    if unique and ("integer" in lowered or lowered.endswith(" id") or lname.endswith("_id")):
        return {**spec, "type": "integer", "generator": "sequence", "unique": True}
    if "email" in lowered:
        return {**spec, "type": "string", "generator": "email", "unique": unique or "email" in lname}
    if "boolean" in lowered or "true or false" in lowered or "yes or no" in lowered:
        return {**spec, "type": "boolean", "generator": "boolean"}
    if match := _DATE_RANGE.search(lowered):
        return {**spec, "type": "date", "generator": "date_range", "min": match.group(1), "max": match.group(2)}
    if "datetime" in lowered or "timestamp" in lowered:
        return {**spec, "type": "datetime", "generator": "date_range"}
    if re.search(r"\bdate\b", lowered) or lname.endswith("_date"):
        return {**spec, "type": "date", "generator": "date_range"}

    based_on = _BASED_ON.search(text)
    if based_on and based_on.group(1) in all_names and based_on.group(1) != name:
        numeric_type = "integer" if any(k in lname for k in ("salary", "wage", "income")) else "float"
        result = {**spec, "type": numeric_type, "generator": "profile", "depends_on": based_on.group(1)}
        driver = next((d for d in _DRIVER_NAMES if d in all_names and d != name), None)
        if driver:
            result["correlate_with"] = CorrelationSpec(field=driver, strength=0.45)
        return result

    if "city" in lowered or lname == "city":
        return {**spec, "type": "string", "generator": "geo_city"}
    if re.search(r"\bstate\b", lowered) or lname == "state":
        return {**spec, "type": "string", "generator": "geo_state"}
    if "country" in lowered:
        return {**spec, "type": "string", "generator": "geo_country"}

    numeric_hint = any(k in lowered for k in ("integer", "number", "decimal", "float", "between"))
    if numeric_hint and (match := _RANGE.search(lowered)):
        low, high = match.group(1), match.group(2)
        is_float = "decimal" in lowered or "float" in lowered or "." in low or "." in high
        return {**spec, "type": "float" if is_float else "integer", "generator": "range",
                "min": float(low) if is_float else int(low), "max": float(high) if is_float else int(high),
                "unique": unique}
    if "integer" in lowered:
        return {**spec, "type": "integer", "generator": "range", "unique": unique}
    if any(k in lowered for k in ("decimal", "float")):
        return {**spec, "type": "float", "generator": "range", "unique": unique}

    for keywords, method, ftype in _FAKER_KEYWORDS:
        if any(k in lowered for k in keywords) or lname in {k.replace(" ", "_") for k in keywords}:
            return {**spec, "type": ftype, "generator": "faker", "faker_method": method, "unique": unique}
    if any(k in lname for k in _NUMERIC_NAMES):
        return {**spec, "type": "float", "generator": "llm", "unique": unique}
    return {**spec, "type": "string", "generator": "llm", "unique": unique, "instructions": text}


def _field_from_raw(name: str, raw: Any, all_names: list[str]) -> FieldSpec:
    try:
        if isinstance(raw, str):
            return FieldSpec(**_infer_from_text(name, raw, all_names))
        if isinstance(raw, list):
            if not raw:
                raise SchemaError(f"Field {name!r}: an enumeration needs at least one value")
            value_type = "integer" if all(isinstance(v, int) and not isinstance(v, bool) for v in raw) else (
                "float" if all(isinstance(v, (int, float)) for v in raw) else "string")
            return FieldSpec(type=value_type, generator="enum", allowed_values=raw)
        if isinstance(raw, dict):
            data = dict(raw)
            if data.get("generator") is None:
                inferred = _infer_from_text(name, str(data.get("description", "")), all_names) \
                    if data.get("description") else {}
                if data.get("link"):
                    inferred = {"generator": "parent"}
                elif data.get("formula"):
                    inferred = {"generator": "formula"}
                elif data.get("allowed_values"):
                    inferred = {"generator": "enum"}
                elif data.get("profile") or data.get("depends_on") and data.get("type") in ("integer", "float"):
                    inferred = {"generator": "profile"}
                elif data.get("reference"):
                    inferred = {"generator": "reference"}
                elif data.get("pattern"):
                    inferred = {"generator": "regex"}
                elif data.get("faker_method"):
                    inferred = {"generator": "faker"}
                elif data.get("value") is not None:
                    inferred = {"generator": "constant"}
                elif not inferred:
                    inferred = {"generator": _default_generator(data.get("type", "string"), data)}
                for key, value in inferred.items():
                    data.setdefault(key, value)
                if "type" in raw:
                    data["type"] = raw["type"]
            return FieldSpec(**data)
    except PydanticValidationError as exc:
        raise SchemaError(f"Field {name!r} is invalid: {exc.errors()[0]['msg']}") from exc
    raise SchemaError(f"Field {name!r}: unsupported definition type {type(raw).__name__}")


def _default_generator(ftype: str, data: dict[str, Any]) -> str:
    if data.get("unique") and ftype == "integer":
        return "sequence"
    return {"integer": "range", "float": "range", "boolean": "boolean", "date": "date_range",
            "datetime": "date_range"}.get(ftype, "llm")


def _parse_temporal(value: Any, field_name: str) -> None:
    if value is None or isinstance(value, (int, float)):
        return
    try:
        (datetime.fromisoformat if "T" in str(value) else date.fromisoformat)(str(value))
    except ValueError as exc:
        raise SchemaError(f"Field {field_name!r}: {value!r} is not an ISO date") from exc


def _validate_field(name: str, spec: FieldSpec, schema_fields: dict[str, FieldSpec]) -> None:  # noqa: PLR0912
    gen = spec.generator
    if gen == "sequence" and spec.type != "integer":
        raise SchemaError(f"Field {name!r}: sequence generator requires type integer")
    if gen == "enum" and not spec.allowed_values:
        raise SchemaError(f"Field {name!r}: enum generator requires allowed_values")
    if gen == "range" and spec.type not in ("integer", "float"):
        raise SchemaError(f"Field {name!r}: range generator requires a numeric type")
    if gen == "date_range":
        if spec.type not in ("date", "datetime"):
            raise SchemaError(f"Field {name!r}: date_range generator requires type date/datetime")
        _parse_temporal(spec.min, name)
        _parse_temporal(spec.max, name)
    if gen == "faker" and not spec.faker_method:
        raise SchemaError(f"Field {name!r}: faker generator requires faker_method")
    if gen == "regex":
        if not spec.pattern:
            raise SchemaError(f"Field {name!r}: regex generator requires pattern")
        RegexGenerator(spec.pattern)  # raises SchemaError when unsupported
    if spec.pattern:
        try:
            re.compile(spec.pattern)
        except re.error as exc:
            raise SchemaError(f"Field {name!r}: invalid regex pattern: {exc}") from exc
    if gen == "profile":
        if spec.type not in ("integer", "float"):
            raise SchemaError(f"Field {name!r}: profile generator requires a numeric type")
        dep = schema_fields.get(spec.depends_on or "")
        if dep is None or not dep.allowed_values:
            raise SchemaError(f"Field {name!r}: profile fields must depend on an enumerated field")
        if spec.profile:
            missing = {str(v) for v in dep.allowed_values} - set(spec.profile)
            if missing:
                raise SchemaError(f"Field {name!r}: profile is missing categories {sorted(missing)}")
    if gen == "reference":
        target = schema_fields.get(spec.reference or "")
        if target is None or target.generator != "sequence":
            raise SchemaError(f"Field {name!r}: reference must point to a sequence field")
    if spec.depends_on and spec.depends_on not in schema_fields:
        raise SchemaError(f"Field {name!r}: depends_on refers to unknown field {spec.depends_on!r}")
    if spec.correlate_with:
        driver = schema_fields.get(spec.correlate_with.field)
        if driver is None or driver.type not in ("integer", "float"):
            raise SchemaError(f"Field {name!r}: correlate_with must reference a numeric field")
    if spec.offset_days and (not spec.depends_on or schema_fields[spec.depends_on].type not in ("date", "datetime")):
        raise SchemaError(f"Field {name!r}: offset_days requires depends_on a date field")
    if spec.min is not None and spec.max is not None and spec.type in ("integer", "float") \
            and float(spec.min) > float(spec.max):
        raise SchemaError(f"Field {name!r}: min is greater than max")
    _validate_extras(name, spec, schema_fields)


def _validate_extras(name: str, spec: FieldSpec, schema_fields: dict[str, FieldSpec]) -> None:  # noqa: PLR0912
    """Checks of the multi-table and planted-data options."""
    gen = spec.generator
    if gen == "parent" and not (spec.link and spec.column):
        raise SchemaError(f"Field {name!r}: generator parent needs link (a link of the table) and column "
                          "(the parent column to copy)")
    if spec.link and gen != "parent":
        raise SchemaError(f"Field {name!r}: link is only used with generator parent")
    if gen == "child_index" and spec.type != "integer":
        raise SchemaError(f"Field {name!r}: child_index requires type integer")
    if gen == "formula":
        if not spec.formula:
            raise SchemaError(f"Field {name!r}: generator formula needs a formula")
        unknown = set(RuleExpression(spec.formula).names) - set(schema_fields)
        if unknown:
            raise SchemaError(f"Field {name!r}: formula uses unknown fields {sorted(unknown)}")
    if (spec.counts is not None or spec.exact) and gen != "enum":
        raise SchemaError(f"Field {name!r}: counts and exact need an enumerated field (allowed_values)")
    if (spec.counts is not None or spec.exact) and spec.null_ratio:
        raise SchemaError(f"Field {name!r}: counts/exact cannot be combined with null_ratio; add null as a value")
    for other in spec.after:
        if other not in schema_fields or other == name:
            raise SchemaError(f"Field {name!r}: after refers to unknown field {other!r}")
    if spec.cases is not None:
        if spec.cases_by not in schema_fields or spec.cases_by == name:
            raise SchemaError(f"Field {name!r}: cases_by refers to unknown field {spec.cases_by!r}")
        if gen == "llm":
            raise SchemaError(f"Field {name!r}: LLM fields cannot have cases (use cases on the fields they read)")
        for key, case in case_specs(name, spec).items():
            if case is None:
                continue
            if case.generator == "llm":
                raise SchemaError(f"Field {name!r}: case {key!r} cannot switch to the LLM generator")
            _validate_field(f"{name}[{key}]", case, schema_fields)
            _check_case_bounds(name, key, spec, case)


def _check_case_bounds(name: str, key: str, base: FieldSpec, case: FieldSpec) -> None:
    """A case must stay inside the field's own bounds and values (records are validated against those)."""
    if base.type in ("integer", "float"):
        for bound, inside in (("min", lambda c, b: c >= b), ("max", lambda c, b: c <= b)):
            limit, value = getattr(base, bound), getattr(case, bound)
            if limit is not None and (value is None or not inside(float(value), float(limit))):
                raise SchemaError(f"Field {name!r}: case {key!r} must keep {bound} within the field's "
                                  f"{bound} ({limit}); widen the field's {bound} or remove it")
    if base.allowed_values and case.allowed_values:
        extra = {str(v) for v in case.allowed_values} - {str(v) for v in base.allowed_values}
        if extra:
            raise SchemaError(f"Field {name!r}: case {key!r} uses values {sorted(extra)} that are not in "
                              "the field's allowed_values")


def case_key(value: Any) -> str:
    """The key a driver value selects in ``cases`` (``true``/``false`` for booleans, ``null`` for empty)."""
    if value is None or value == "":
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def case_specs(name: str, spec: FieldSpec) -> dict[str, FieldSpec | None]:
    """Expand ``cases`` into full field specs (each case overrides the field's own settings).

    Raises:
        SchemaError: If a case is not a valid field definition.
    """
    base = spec.model_dump(exclude={"cases", "cases_by"}, exclude_none=True)
    out: dict[str, FieldSpec | None] = {}
    for key, override in (spec.cases or {}).items():
        if override is None:
            out[key] = None
            continue
        data = {**base, **override}
        if "generator" not in override:
            if "value" in override:
                data["generator"] = "constant"
            elif "allowed_values" in override:
                data["generator"] = "enum"
                if "weights" not in override:
                    data.pop("weights", None)
                    data.pop("counts", None)
            elif "formula" in override:
                data["generator"] = "formula"
            elif "link" in override:
                data["generator"] = "parent"
            elif ("min" in override or "max" in override or "distribution" in override) and \
                    base.get("generator") not in ("range", "date_range"):
                data["generator"] = "date_range" if spec.type in ("date", "datetime") else "range"
        if "counts" in override or "exact" in override:
            raise SchemaError(f"Field {name!r}: case {key!r}: exact counts apply to the whole table and cannot "
                              "be set inside cases (use weights there)")
        data.pop("counts", None)
        data["exact"] = False
        try:
            out[key] = FieldSpec(**data)
        except PydanticValidationError as exc:
            raise SchemaError(f"Field {name!r}: case {key!r} is invalid: {exc.errors()[0]['msg']}") from exc
    return out


def dependency_order(schema: DatasetSchema) -> list[str]:
    """Return field names ordered so every dependency is generated first.

    Args:
        schema: The normalised schema.

    Returns:
        Field names in a valid generation order.

    Raises:
        SchemaError: If the dependencies contain a cycle.
    """
    graph: dict[str, set[str]] = {}
    name_fields = [n for n, s in schema.fields.items() if s.generator == "faker" and s.faker_method == "name"]
    for name, spec in schema.fields.items():
        deps: set[str] = set()
        if spec.depends_on:
            deps.add(spec.depends_on)
        if spec.correlate_with:
            deps.add(spec.correlate_with.field)
        if spec.generator == "email" and name_fields:
            deps.add(name_fields[0])
        if spec.generator in ("geo_city",):
            deps.update(n for n, s in schema.fields.items() if s.generator == "geo_state")
        if spec.generator == "llm":
            deps.update(n for n, s in schema.fields.items() if s.generator != "llm")
        if spec.cases_by:
            deps.add(spec.cases_by)
            for case in case_specs(name, spec).values():
                if case is not None:
                    deps.update(_own_deps(case))
        deps.update(_own_deps(spec))
        graph[name] = deps - {name}
    try:
        return list(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise SchemaError(f"Field dependencies contain a cycle: {exc.args[1]}") from exc


def _own_deps(spec: FieldSpec) -> set[str]:
    deps = set(spec.after)
    if spec.depends_on:
        deps.add(spec.depends_on)
    if spec.correlate_with:
        deps.add(spec.correlate_with.field)
    if spec.formula:
        deps.update(RuleExpression(spec.formula).names)
    return deps


def check_unique_capacity(schema: DatasetSchema, records: int) -> dict[str, float]:
    """Check that every unique regex field can hold ``records`` distinct values.

    By the birthday effect, drawing ``n`` random values from ``N`` possibilities
    gives about ``n^2 / 2N`` collisions. Each collision is rejected and regenerated,
    so a small pattern wastes work, and a pattern smaller than ``n`` can never finish.

    Args:
        schema: The normalised schema.
        records: Total records of the dataset (the whole split run, if split).

    Returns:
        Expected collisions per field.

    Raises:
        SchemaError: If a unique pattern has fewer possible values than ``records``.
    """
    expected: dict[str, float] = {}
    for name, spec in schema.fields.items():
        if not spec.unique or spec.generator != "regex" or not spec.pattern:
            continue
        capacity = RegexGenerator(spec.pattern).cardinality()
        if capacity < records * 2:
            raise SchemaError(f"Field {name!r}: pattern {spec.pattern!r} allows only {capacity:,} distinct values, "
                              f"too few for {records:,} unique records; make it longer")
        expected[name] = records * records / (2 * capacity)
        if expected[name] > 100:
            logger.warning("Field %r: pattern %r allows %s values; ~%.0f collisions expected for %s records "
                           "(each one is regenerated). A longer pattern avoids the extra work.", name,
                           spec.pattern, f"{capacity:,}", expected[name], f"{records:,}")
    return expected


def _field_cardinality(spec: FieldSpec) -> int | None:
    """Distinct values a field can take, or ``None`` when practically unlimited."""
    extra = 1 if spec.nullable else 0
    if spec.allowed_values:
        return len(spec.allowed_values) + extra
    if spec.type == "boolean":
        return 2 + extra
    if spec.type in ("integer", "number"):
        if spec.profile:
            low, high = min(r.min for r in spec.profile.values()), max(r.max for r in spec.profile.values())
        elif isinstance(spec.min, (int, float)) and isinstance(spec.max, (int, float)):
            low, high = float(spec.min), float(spec.max)
        else:
            return None
        steps = 1 if spec.type == "integer" else 10 ** (spec.decimals if spec.decimals is not None else 2)
        return int((high - low) * steps) + 1 + extra
    if spec.type == "date" and spec.offset_days:
        return spec.offset_days[1] - spec.offset_days[0] + 1 + extra
    if spec.type == "date" and isinstance(spec.min, str) and isinstance(spec.max, str):
        try:
            return (date.fromisoformat(spec.max[:10]) - date.fromisoformat(spec.min[:10])).days + 1 + extra
        except ValueError:
            return None
    return None


def _cardinality(schema: DatasetSchema, name: str, links: dict[str, int] | None = None) -> int | None:
    """Distinct values of a field, counting formulas as derived and cases as the union of their values.

    ``links`` gives the parent rows behind each link (a ``parent`` field has at most that many values).
    """
    spec = schema.fields[name]
    if spec.generator == "formula":
        return 1  # determined by other fields: adds no new combinations
    if spec.generator == "parent":
        return (links or {}).get(spec.link or "")
    if spec.generator == "child_index":
        return (links or {}).get("__child_max__")
    if spec.cases is None:
        return _field_cardinality(spec)
    total = 0
    cases = case_specs(name, spec)
    for case in cases.values():
        count = 1 if case is None else (1 if case.generator == "formula" else _field_cardinality(case))
        if count is None:
            return None
        total += count
    driver = schema.fields[spec.cases_by]  # type: ignore[index]
    covered = driver.allowed_values is not None and {case_key(v) for v in driver.allowed_values} <= set(cases)
    if not covered:
        base = _field_cardinality(spec)
        if base is None:
            return None
        total += base
    return total


def record_space(schema: DatasetSchema, links: dict[str, int] | None = None) -> int | None:
    """Distinct combinations of the fields used for whole-record duplicate detection (``None`` = unlimited)."""
    total = 1
    for name in schema.columns:
        if name in schema.dedup_exclude:
            continue
        count = _cardinality(schema, name, links)
        if count is None:
            return None
        total *= count
        if total > 10**15:
            return None
    return total


def check_record_space(schema: DatasetSchema, records: int, links: dict[str, int] | None = None) -> DatasetSchema:
    """Turn off whole-record duplicate detection when the non-ID fields cannot make enough distinct records.

    Example: an ID plus a two-value category and a 1-20 score allow only ~40
    combinations. Rejecting "identical" records there (identical apart from the
    ID) could never finish for 3,000 records, and such repeats are legitimate.
    Unique fields are still enforced.

    Returns:
        The schema, with ``record_dedup`` off when needed.
    """
    space = record_space(schema, links)
    if schema.record_dedup and space is not None and space < records * 2:
        logger.info("Whole-record duplicate check off: the non-ID fields allow only %s distinct combinations "
                    "for %s records (unique fields are still enforced)", f"{space:,}", f"{records:,}")
        return schema.model_copy(update={"record_dedup": False})
    return schema


def parse_schema(raw: dict[str, Any], instructions: str | None = None) -> DatasetSchema:
    """Parse a shorthand or rich schema into a validated :class:`DatasetSchema`.

    Args:
        raw: The ``schema`` object from the request.
        instructions: Generation instructions (used for locale detection).

    Returns:
        The normalised schema.

    Raises:
        SchemaError: If the schema is empty, inconsistent or unsupported.
    """
    if not isinstance(raw, dict) or not raw:
        raise SchemaError("Schema must be a non-empty object")
    rich = isinstance(raw.get("fields"), dict)
    field_defs: dict[str, Any] = raw["fields"] if rich else raw
    if not field_defs:
        raise SchemaError("Schema defines no fields")
    for name in field_defs:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
            raise SchemaError(f"Invalid field name {name!r}: use letters, digits and underscores")
    names = list(field_defs)
    fields = {name: _field_from_raw(name, field_defs[name], names) for name in names}
    for name, spec in fields.items():  # a case that empties the field makes it nullable
        if spec.cases and any(case is None for case in spec.cases.values()) and not spec.nullable:
            fields[name] = spec.model_copy(update={"nullable": True})

    descriptions = " ".join(spec.description for spec in fields.values())
    locale = (raw.get("locale") if rich else None) or detect_locale(instructions, descriptions) or "en_US"
    try:
        rules = [RuleSpec(**rule) for rule in (raw.get("rules", []) if rich else [])]
    except PydanticValidationError as exc:
        raise SchemaError(f"Invalid rule definition: {exc.errors()[0]['msg']}") from exc
    for rule in rules:
        expression = RuleExpression(rule.expression)
        unknown = set(expression.names) - set(fields)
        if unknown:
            raise SchemaError(f"Rule {rule.name!r} references unknown fields {sorted(unknown)}")

    semantic_fields = list(raw.get("semantic_fields", [])) if rich else []
    if not semantic_fields:
        # Semantic similarity is only meaningful for free text; structured fields are covered by exact dedup.
        semantic_fields = [n for n, s in fields.items() if s.generator == "llm" and s.type == "string"]
    dedup_exclude = list(raw.get("dedup_exclude", [])) if rich else []
    if not dedup_exclude:
        # IDs, and values fixed by the record's position or other fields (parent links, child numbers, formulas):
        # regenerating a record cannot change those, and repeats of them are legitimate.
        dedup_exclude = [n for n, s in fields.items()
                         if s.generator in ("sequence", "uuid", "parent", "child_index", "formula")]

    try:
        schema = DatasetSchema(
            entity=str(raw.get("entity", "record")) if rich else "record",
            fields=fields, rules=rules, locale=locale,
            semantic_fields=semantic_fields, dedup_exclude=dedup_exclude,
        )
    except PydanticValidationError as exc:
        raise SchemaError(f"Invalid schema: {exc.errors()[0]['msg']}") from exc
    for field_name in (*schema.semantic_fields, *schema.dedup_exclude):
        if field_name not in fields:
            raise SchemaError(f"Unknown field {field_name!r} in semantic_fields/dedup_exclude")
    for name, spec in fields.items():
        _validate_field(name, spec, fields)
        if spec.generator in ("geo_city", "geo_state") and schema.locale not in supported_geo_locales():
            raise SchemaError(
                f"Field {name!r}: geographic data is available for {supported_geo_locales()}, not {schema.locale}"
            )
    dependency_order(schema)
    return schema
