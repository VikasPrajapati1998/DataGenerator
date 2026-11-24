"""Turning user input (a request file, a bare schema, overrides) into a checked engine request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.config.settings import PROJECT_ROOT
from app.core.exceptions import SchemaError
from app.generation.links import resolve_counts
from app.multi_table import is_multi_table, parse_tables
from app.schemas.input_schema import GenerationRequest
from app.schemas.schema_parser import check_unique_capacity, parse_schema
from app.service.config import ServiceSettings

SAMPLES_DIR = PROJECT_ROOT / "samples" / "schemas"

# Keys a remote caller must not set: they choose where files are written or are engine-internal.
_SERVER_OWNED = ("output_dir", "sub_job", "job_id")
_OVERRIDE_KEYS = {"records": "total_records", "output_format": "output_format", "format": "output_format",
                  "provider": "model_provider", "model": "model_name", "seed": "seed", "llm_mode": "llm_mode",
                  "allow_no_llm": "allow_no_llm", "schema_name": "schema_name", "batch_size": "batch_size",
                  "force": "force"}


class RequestError(SchemaError):
    """The request or schema is invalid; ``problems`` lists every issue found."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or [message]


def load_payload(payload: dict[str, Any] | str | bytes) -> dict[str, Any]:
    """Parse JSON text (or accept a dict) and return the request object.

    A payload without a ``schema`` key is treated as a bare schema: ``{"schema": payload}``.

    Raises:
        RequestError: If the text is not a JSON object.
    """
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RequestError(f"invalid JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict) or not payload:
        raise RequestError("the request must be a non-empty JSON object")
    if "schema" in payload:
        return dict(payload)
    if isinstance(payload.get("tables"), dict):  # a multi-table file: tables plus request settings
        rest = {k: v for k, v in payload.items() if k != "tables"}
        return {"schema": {"tables": payload["tables"]}, **rest}
    return {"schema": payload}


def build_request(payload: dict[str, Any] | str | bytes, overrides: dict[str, Any] | None = None,
                  default_records: int | None = None) -> dict[str, Any]:
    """Merge a request with call-time overrides and drop the keys only the server may set.

    Args:
        payload: Request object (``{"schema": ..., "total_records": ...}``), bare schema, or JSON text.
        overrides: ``records``, ``format``, ``provider``, ``model``, ``seed``, ``llm_mode``, ... (``None`` = unset).
        default_records: Used when neither the request nor the overrides give a record count.

    Returns:
        A request dictionary for :class:`GenerationRequest`.
    """
    request = load_payload(payload)
    for key in _SERVER_OWNED:
        request.pop(key, None)
    for key, value in (overrides or {}).items():
        if value is not None and key in _OVERRIDE_KEYS:
            request[_OVERRIDE_KEYS[key]] = value
    if "total_records" not in request and default_records:
        request["total_records"] = default_records
    return request


# Model and tuning keys in an uploaded file are ignored: the mode chooses mock or the .env model pool.
_MODEL_KEYS = ("model_provider", "model_name", "fallback_models", "model_pool", "model_assignment", "model_weights",
               "allow_no_llm")
MODES = ("mock", "llm")


def prepare_job(payload: dict[str, Any] | str | bytes, records: int | None = None, output_format: str | None = None,
                mode: str = "llm", schema_name: str | None = None) -> tuple[dict[str, Any], str]:
    """Build an engine request from a schema file and the few choices the simple API offers.

    Args:
        payload: The uploaded schema / request JSON (object or text).
        records: Number of records (overrides ``total_records`` in the file).
        output_format: ``csv``, ``json``, ``jsonl`` or ``excel`` (default: the file's, else ``csv``).
        mode: ``mock`` (offline placeholder model) or ``llm`` (the models configured in ``.env``).
        schema_name: Job name prefix (default: the file's ``schema_name``, else its ``entity``).

    Returns:
        The engine request and the format delivered to the user.

    Raises:
        RequestError: For an unknown mode or format, or an invalid file.
    """
    if mode not in MODES:
        raise RequestError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    if is_multi_table(load_payload(payload).get("schema")):
        if records is not None:
            raise RequestError("records is set per table in a multi-table file (\"records\" or \"per_parent\" "
                               "of each table); leave the records field empty")
        request = build_request(payload, {"schema_name": schema_name})
        request.pop("total_records", None)  # the sum of the tables
    else:
        request = build_request(payload, {"records": records, "schema_name": schema_name})
    for key in _MODEL_KEYS:
        request.pop(key, None)
    if mode == "mock":
        request.update(model_provider="mock", model_name="mock-llm")
    deliver = (output_format or request.get("output_format") or "csv").lower()
    if deliver in ("xlsx", "xls"):
        deliver = "excel"
    if deliver not in ("csv", "json", "jsonl", "excel"):
        raise RequestError(f"output format must be csv, json, jsonl or excel, got {deliver!r}")
    request["output_format"] = "jsonl" if deliver == "excel" else deliver  # Excel is converted from JSONL
    return request, deliver


def validate_request(request: dict[str, Any], limits: ServiceSettings, max_records: int | None = None
                     ) -> tuple[GenerationRequest, dict[str, Any]]:
    """Check a request completely without generating anything.

    Args:
        request: Output of :func:`build_request`.
        limits: Service limits.
        max_records: Size cap for this call (default ``limits.max_records``).

    Returns:
        The parsed request and a JSON-friendly report (fields, generators, LLM fields, warnings).

    Raises:
        RequestError: With every problem found.
    """
    if is_multi_table(request.get("schema")):
        return _validate_multi(request, limits, max_records)
    if "total_records" not in request:
        raise RequestError("the number of records is missing: send records, or put total_records in the file")
    try:
        model = GenerationRequest(**request)
    except PydanticValidationError as exc:
        problems = [f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors()[:20]]
        raise RequestError("invalid request: " + "; ".join(problems), problems) from exc
    cap = max_records or limits.max_records
    if model.total_records > cap:
        raise RequestError(f"total_records={model.total_records:,} is above the limit of {cap:,} for this call")
    try:
        schema = parse_schema(model.data_schema, model.generation_instructions)
        collisions = check_unique_capacity(schema, model.total_records)
    except SchemaError as exc:
        raise RequestError(str(exc)) from exc
    warnings = [f"field {name!r}: about {count:,.0f} unique-value collisions expected (each is regenerated); "
                "a longer pattern is faster" for name, count in collisions.items() if count > 100]
    if schema.llm_fields:
        warnings.append(f"LLM fields {schema.llm_fields} need a reachable model (or allow_no_llm=true / mock "
                        "provider); large runs use pooled mode")
    report = {
        "valid": True, "total_records": model.total_records, "output_format": model.output_format,
        "entity": schema.entity, "locale": schema.locale, "columns": schema.columns,
        "fields": {name: {"type": spec.type, "generator": spec.generator, "unique": spec.unique}
                   for name, spec in schema.fields.items()},
        "unique_fields": schema.unique_fields, "llm_fields": schema.llm_fields,
        "rules": [rule.name for rule in schema.rules], "warnings": warnings,
        "model": model.model_name or (model.model_pool or None) or "default (SDE_MODEL_POOL / SDE_DEFAULT_MODEL)",
    }
    return model, report


def _validate_multi(request: dict[str, Any], limits: ServiceSettings, max_records: int | None
                    ) -> tuple[GenerationRequest, dict[str, Any]]:
    """Check a multi-table request: every table schema, the links, fixed exact counts and unique capacity."""
    request = {k: v for k, v in request.items() if k != "total_records"}
    try:
        model = GenerationRequest(**request)
        tables = parse_tables(model.data_schema, model.generation_instructions)
    except PydanticValidationError as exc:
        problems = [f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors()[:20]]
        raise RequestError("invalid request: " + "; ".join(problems), problems) from exc
    except SchemaError as exc:
        raise RequestError(str(exc)) from exc
    cap = max_records or limits.max_records
    warnings: list[str] = []
    described = []
    for table in tables:
        if table.estimate > cap:
            raise RequestError(f"table {table.name!r}: about {table.estimate:,} records, above the limit of {cap:,}")
        try:
            collisions = check_unique_capacity(table.schema, table.estimate)
            if table.records is not None:
                for name, spec in table.schema.fields.items():
                    if spec.counts is not None:
                        resolve_counts(name, list(spec.allowed_values or []), spec.counts, spec.weights,
                                       table.records)
        except SchemaError as exc:
            raise RequestError(f"table {table.name!r}: {exc}") from exc
        warnings += [f"table {table.name!r}, field {name!r}: about {count:,.0f} unique-value collisions expected"
                     for name, count in collisions.items() if count > 100]
        described.append({
            "name": table.name,
            "records": table.records if table.records is not None else f"about {table.estimate:,} "
                       f"({table.per_parent['min']}-{table.per_parent['max']} per {table.per_parent['table']} row)",
            "columns": [c for c in table.schema.columns if c not in table.hidden], "hidden": table.hidden,
            "links": {name: {"table": link["table"], "mode": link["mode"], "columns": table.link_fields(name)}
                      for name, link in table.links.items()},
            "primary_key": table.primary_key, "llm_fields": table.schema.llm_fields,
        })
    if any(t.schema.llm_fields for t in tables):
        warnings.append("LLM fields need a reachable model in llm mode (mock mode fills placeholder text)")
    report = {"valid": True, "multi_table": True, "order": [t.name for t in tables],
              "total_records": model.total_records, "output_format": model.output_format, "tables": described,
              "warnings": warnings}
    return model, report


def list_samples() -> list[dict[str, Any]]:
    """Describe the sample request files in ``samples/schemas``."""
    samples = []
    for path in sorted(SAMPLES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        schema = load_payload(data).get("schema", {})
        if is_multi_table(schema):
            samples.append({"name": path.stem, "tables": sorted(schema["tables"]), "fields": 0,
                            "default_records": None, "output_format": data.get("output_format"),
                            "pins_models": False, "path": str(path)})
            continue
        fields = schema.get("fields", schema) if isinstance(schema, dict) else {}
        samples.append({"name": path.stem, "entity": schema.get("entity") if isinstance(schema, dict) else None,
                        "fields": len(fields) if isinstance(fields, dict) else 0,
                        "default_records": data.get("total_records"), "output_format": data.get("output_format"),
                        "pins_models": bool(data.get("model_pool") or data.get("model_name")),
                        "path": str(path)})
    return samples


def load_sample(name: str) -> dict[str, Any]:
    """Return the request object of sample ``name`` (file stem, with or without ``.json``).

    Raises:
        RequestError: If no such sample exists.
    """
    stem = Path(name).stem
    path = SAMPLES_DIR / f"{stem}.json"
    if not stem or path.parent != SAMPLES_DIR or not path.is_file():
        known = ", ".join(s["name"] for s in list_samples())
        raise RequestError(f"unknown sample {name!r}; available: {known}")
    return json.loads(path.read_text(encoding="utf-8"))
