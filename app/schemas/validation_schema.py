"""Dynamic Pydantic record models and the record validator built on them.

A Pydantic model is generated from the :class:`DatasetSchema` at runtime
(``pydantic.create_model``) and enforces required fields, types,
nullability, numeric/date bounds, enumerations, lengths and regex patterns.
Cross-field business rules are evaluated afterwards with the safe rule
engine. Validation runs in lax mode so CSV strings are coerced to the
declared types when batch files are re-validated.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, create_model
from pydantic import ValidationError as PydanticValidationError

from app.schemas.input_schema import DatasetSchema, FieldSpec
from app.utils.safe_eval import RuleExpression

_PY_TYPES: dict[str, type] = {"integer": int, "float": float, "string": str, "boolean": bool,
                              "date": date, "datetime": datetime}


def _empty_to_none(value: Any) -> Any:
    return None if value == "" else value


def _membership(allowed: list[Any], ftype: str):  # noqa: ANN202
    normalised = {str(v) for v in allowed} if ftype == "string" else set(allowed)

    def check(value: Any) -> Any:
        if value is None:
            return value
        probe = str(value) if ftype == "string" else value
        if probe not in normalised:
            raise ValueError(f"value {value!r} is not one of the allowed values")
        return value
    return check


def _temporal_bounds(spec: FieldSpec):  # noqa: ANN202
    def to_date(raw: Any) -> date:
        text = str(raw)
        return datetime.fromisoformat(text).date() if "T" in text else date.fromisoformat(text)

    low = to_date(spec.min) if spec.min is not None else None
    high = to_date(spec.max) if spec.max is not None else None

    def check(value: Any) -> Any:
        if value is None:
            return value
        day = value.date() if isinstance(value, datetime) else value
        if low is not None and day < low:
            raise ValueError(f"date {day} is before {low}")
        if high is not None and day > high:
            raise ValueError(f"date {day} is after {high}")
        return value
    return check


def _field_definition(spec: FieldSpec) -> tuple[Any, Any]:
    py_type: Any = _PY_TYPES[spec.type]
    constraints: dict[str, Any] = {}
    validators: list[Any] = [BeforeValidator(_empty_to_none)]
    if spec.type in ("integer", "float"):
        if spec.min is not None:
            constraints["ge"] = float(spec.min)
        if spec.max is not None:
            constraints["le"] = float(spec.max)
    if spec.type == "string":
        if spec.pattern:
            constraints["pattern"] = spec.pattern
        if spec.max_length:
            constraints["max_length"] = spec.max_length
        if spec.min_length is not None:
            constraints["min_length"] = spec.min_length
        elif spec.required and not spec.nullable:
            constraints["min_length"] = 1
    if spec.type in ("date", "datetime") and (spec.min is not None or spec.max is not None):
        validators.append(AfterValidator(_temporal_bounds(spec)))
    if spec.allowed_values and spec.generator != "boolean":
        validators.append(AfterValidator(_membership(spec.allowed_values, spec.type)))
    annotated = Annotated[py_type, Field(**constraints), *validators]
    if spec.nullable or not spec.required:
        return annotated | None, None if not spec.required else ...
    return annotated, ...


def build_record_model(schema: DatasetSchema) -> type[BaseModel]:
    """Create a Pydantic model class that validates one record.

    Args:
        schema: The normalised schema.

    Returns:
        A model class named after the schema entity.
    """
    definitions = {name: _field_definition(spec) for name, spec in schema.fields.items()}
    model_name = "".join(part.capitalize() for part in schema.entity.split("_")) or "Record"
    return create_model(  # type: ignore[call-overload]
        f"{model_name}Model",
        __config__=ConfigDict(extra="forbid", str_strip_whitespace=False),
        **definitions,
    )


class RecordValidator:
    """Validates records against the schema model and the business rules.

    Args:
        schema: The normalised schema.
    """

    def __init__(self, schema: DatasetSchema) -> None:
        self.schema = schema
        self.model = build_record_model(schema)
        self.rules = [(rule, RuleExpression(rule.expression)) for rule in schema.rules]

    def validate(self, record: dict[str, Any]) -> list[str]:
        """Return the list of violations for ``record`` (empty when valid).

        Args:
            record: Field values by name (Python types or CSV strings).

        Returns:
            Human-readable error messages; ``warning`` rules are prefixed with
            ``warning:`` and do not count as failures.
        """
        try:
            parsed = self.model.model_validate(record).model_dump()
        except PydanticValidationError as exc:
            return [f"schema: {'.'.join(str(p) for p in err['loc']) or 'record'}: {err['msg']}"
                    for err in exc.errors()]
        return self.rule_errors(parsed)

    def rule_errors(self, parsed: dict[str, Any]) -> list[str]:
        """Evaluate business rules on an already type-validated record.

        Args:
            parsed: Record with Python-typed values.

        Returns:
            Rule violations (``warning:``-prefixed for warning rules).
        """
        errors: list[str] = []
        for rule, expression in self.rules:
            if not expression.evaluate(parsed):
                prefix = "warning: " if rule.severity == "warning" else ""
                errors.append(f"{prefix}rule: {rule.name}: {rule.description or rule.expression}")
        return errors

    @staticmethod
    def is_failure(errors: list[str]) -> bool:
        """Return ``True`` when ``errors`` contains at least one blocking error."""
        return any(not e.startswith("warning:") for e in errors)
