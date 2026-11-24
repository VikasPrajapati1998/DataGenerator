"""User-facing request and dataset-schema models.

A request can describe fields in shorthand (``"age": "integer between 22 and
60"``) or in the rich form (``"age": {"type": "integer", "min": 22, ...}``).
:mod:`app.schemas.schema_parser` converts both into :class:`DatasetSchema`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FieldType = Literal["integer", "float", "string", "boolean", "date", "datetime"]
GeneratorName = Literal[
    "sequence", "uuid", "faker", "enum", "range", "date_range", "email", "geo_city", "geo_state",
    "geo_country", "profile", "reference", "regex", "llm", "boolean", "constant", "parent", "child_index", "formula",
]
OutputFormat = Literal["csv", "json", "jsonl"]
ExecutionMode = Literal["auto", "thread", "process"]


class ProfileRange(BaseModel):
    """Numeric range of a profile field for one category value.

    Attributes:
        min: Lowest plausible value.
        max: Highest plausible value.
        typical: Most common value (mode of the triangular distribution).
    """

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float
    typical: float | None = None

    @model_validator(mode="after")
    def _check_order(self) -> ProfileRange:
        if self.min >= self.max:
            raise ValueError(f"profile min ({self.min}) must be below max ({self.max})")
        if self.typical is not None and not self.min <= self.typical <= self.max:
            raise ValueError("profile typical must lie between min and max")
        return self


class DistributionSpec(BaseModel):
    """Statistical distribution for a numeric field.

    Attributes:
        kind: Distribution family.
        mean: Mean (normal) or log-space mean (lognormal).
        std: Standard deviation (normal) or log-space sigma (lognormal).
        mode: Mode of a triangular distribution.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["uniform", "normal", "lognormal", "triangular"] = "uniform"
    mean: float | None = None
    std: float | None = Field(default=None, gt=0)
    mode: float | None = None


class CorrelationSpec(BaseModel):
    """Positive correlation of a numeric field with another numeric field.

    Attributes:
        field: The driving numeric field (must be generated earlier).
        strength: 0 (independent) .. 1 (fully determined by the driver's percentile).
    """

    model_config = ConfigDict(extra="forbid")

    field: str
    strength: float = Field(default=0.5, ge=0, le=1)


class FieldSpec(BaseModel):
    """Complete specification of one dataset field.

    Attributes mirror the supported schema features: types, required/nullable
    flags, uniqueness, numeric/date bounds, enumerations with weights, regex
    patterns, dependencies, cross-record references, statistical
    distributions and custom generation instructions.

    Multi-table and planted-data options: ``counts`` / ``exact`` assign enum values in exact numbers;
    ``cases_by`` + ``cases`` override the field per value of another field (``null`` = empty);
    ``formula`` computes the value from other fields; ``link`` + ``column`` copy a column of the linked
    parent row (generator ``parent``); generator ``child_index`` numbers rows within their parent;
    ``after`` adds generation-order dependencies; ``hidden`` drops a helper column from multi-table output.
    """

    model_config = ConfigDict(extra="forbid")

    type: FieldType = "string"
    description: str = ""
    generator: GeneratorName | None = None
    required: bool = True
    nullable: bool = False
    null_ratio: float = Field(default=0.0, ge=0, le=1)
    unique: bool = False
    min: float | str | None = None
    max: float | str | None = None
    allowed_values: list[Any] | None = None
    weights: list[float] | None = None
    pattern: str | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=1)
    decimals: int | None = Field(default=None, ge=0, le=10)
    faker_method: str | None = None
    faker_args: dict[str, Any] = Field(default_factory=dict)
    depends_on: str | None = None
    profile: dict[str, ProfileRange] | None = None
    distribution: DistributionSpec | None = None
    correlate_with: CorrelationSpec | None = None
    reference: str | None = None
    offset_days: tuple[int, int] | None = None
    value: Any = None
    instructions: str | None = None
    semantic: bool | None = None
    counts: list[int | None] | None = None
    exact: bool = False
    cases_by: str | None = None
    cases: dict[str, dict[str, Any] | None] | None = None
    formula: str | None = None
    link: str | None = None
    column: str | None = None
    after: list[str] = Field(default_factory=list)
    hidden: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> FieldSpec:
        if self.weights is not None:
            if not self.allowed_values or len(self.weights) != len(self.allowed_values):
                raise ValueError("weights must have one entry per allowed value")
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError("weights must be non-negative and not all zero")
        if self.null_ratio > 0 and not self.nullable:
            raise ValueError("null_ratio > 0 requires nullable=true")
        if self.unique and self.nullable:
            raise ValueError("unique fields cannot be nullable")
        if self.counts is not None:
            if not self.allowed_values or len(self.counts) != len(self.allowed_values):
                raise ValueError("counts must have one entry per allowed value")
            if any(c is not None and c < 0 for c in self.counts) or sum(c is None for c in self.counts) > 1:
                raise ValueError("counts must be non-negative numbers, with at most one null (= the rest)")
        if self.exact and self.weights is None and self.counts is None:
            raise ValueError("exact needs weights (exact proportions) or counts")
        if (self.cases is None) != (self.cases_by is None):
            raise ValueError("cases and cases_by go together: cases_by names the field whose value picks the case")
        return self


class RuleSpec(BaseModel):
    """Cross-field business rule evaluated on every record.

    Attributes:
        name: Short identifier used in reports.
        expression: Expression in the safe rule language (see ``safe_eval``).
        description: Human-readable explanation, also given to the LLM.
        severity: ``error`` rejects the record, ``warning`` only reports.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    expression: str
    description: str = ""
    severity: Literal["error", "warning"] = "error"


class DatasetSchema(BaseModel):
    """Normalised schema used by every component of the engine.

    Attributes:
        entity: Singular name of what one record represents.
        fields: Ordered field specifications (order defines column order).
        rules: Cross-field business rules.
        locale: Faker locale.
        semantic_fields: Fields compared for near-duplicate detection.
        dedup_exclude: Fields ignored by exact record-duplicate detection.
        record_dedup: Reject records identical in every non-excluded field. Turned off by
            ``check_record_space`` when those fields allow too few combinations.
    """

    model_config = ConfigDict(extra="forbid")

    entity: str = "record"
    fields: dict[str, FieldSpec]
    rules: list[RuleSpec] = Field(default_factory=list)
    locale: str = "en_US"
    semantic_fields: list[str] = Field(default_factory=list)
    dedup_exclude: list[str] = Field(default_factory=list)
    record_dedup: bool = True

    @property
    def columns(self) -> list[str]:
        """Field names in output column order."""
        return list(self.fields)

    @property
    def record_fields(self) -> list[str]:
        """Fields of the whole-record duplicate key (empty when ``record_dedup`` is off)."""
        return [c for c in self.fields if c not in self.dedup_exclude] if self.record_dedup else []

    @property
    def unique_fields(self) -> list[str]:
        """Fields declared unique."""
        return [name for name, spec in self.fields.items() if spec.unique]

    @property
    def llm_fields(self) -> list[str]:
        """Fields generated per record by the LLM."""
        return [name for name, spec in self.fields.items() if spec.generator == "llm"]

    @property
    def profile_fields(self) -> list[str]:
        """Fields whose ranges depend on a category (LLM-calibrated if unspecified)."""
        return [name for name, spec in self.fields.items() if spec.generator == "profile"]


class SubJobContext(BaseModel):
    """Set by the split runner on each sub-job of a large request (not written by users).

    Attributes:
        part: 1-based sub-job number (also the uniqueness-registry namespace).
        record_offset: Global index of the sub-job's first record (IDs continue across sub-jobs).
        dataset_total_records: Size of the whole split dataset.
        job_dir: Directory of this sub-job.
        registry_path: Uniqueness registry shared by all sub-jobs.
        profiles: Category ranges calibrated once for the whole split run.
        dataset_seed: Seed of the whole split run (exact counts and parent links must not change per sub-job).
    """

    model_config = ConfigDict(extra="forbid")

    part: int = Field(ge=1, le=9999)
    record_offset: int = Field(ge=0)
    dataset_total_records: int = Field(gt=0)
    job_dir: str
    registry_path: str
    profiles: dict[str, dict[str, dict[str, float | None]]] = Field(default_factory=dict)
    dataset_seed: int | None = None


class GenerationRequest(BaseModel):
    """Input accepted by the engine.

    Attributes:
        data_schema: Shorthand or rich schema (JSON key ``schema``).
        schema_name: Name used as the job-id prefix and output folder. Defaults to
            the request file name (CLI), else the rich schema's ``entity``, else ``dataset``.
        total_records: Exact number of records to deliver.
        output_format: ``csv``, ``json`` or ``jsonl``.
        generation_instructions: Optional free-text guidance.
        model_provider: Preferred provider (``ollama`` maps to ``ollama_local``).
        model_name: Preferred model.
        fallback_models: Registry keys tried when the preferred model fails.
        model_pool: Registry keys (``provider/model``) spread over the generation
            tasks; each task uses exactly one pool model, the others act as its
            fallbacks. Empty means "use model_provider/model_name for every task".
        model_assignment: How pool models are assigned to tasks.
        model_weights: Relative share of tasks per pool model (``weighted`` only).
        batch_size: Target records per task; drives the power-of-two task count.
        quality_requirements: Overrides for :class:`GuardrailConfig`.
        max_workers: Optional cap on logical core queues.
        work_stealing: Let idle queues take pending tasks from busy queues.
        execution_mode: ``thread``, ``process`` or ``auto``.
        cpu_affinity: Pin queue threads/processes to logical CPUs where supported.
        seed: Seed for reproducible generation.
        job_id: Optional explicit job id.
        output_dir: Optional output root overriding the settings.
        split_size: Max records per sub-job. Larger requests run as several
            sequential sub-jobs that are merged at the end (``0`` disables splitting;
            ``None`` uses ``SDE_MAX_RECORDS_PER_JOB``).
        allow_no_llm: When no model is reachable, fill LLM text fields with Faker
            placeholder text (and LLM numbers from their min/max) instead of failing.
        llm_mode: ``per_record``, ``pooled`` or ``auto``; ``None`` uses ``SDE_LLM_MODE``.
        force: Start even when the projected run time exceeds ``SDE_MAX_ETA_HOURS``.
        sub_job: Internal context the split runner gives each sub-job.
        links: Internal context the multi-table runner gives each table (parent tables, per-parent counts).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    data_schema: dict[str, Any] = Field(alias="schema")
    schema_name: str | None = Field(default=None, max_length=80)
    total_records: int = Field(gt=0, le=50_000_000)
    output_format: OutputFormat = "csv"
    generation_instructions: str | None = Field(default=None, max_length=8000)
    model_provider: str | None = None
    model_name: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    model_pool: list[str] = Field(default_factory=list, max_length=64)
    model_assignment: Literal["round_robin", "weighted"] = "round_robin"
    model_weights: list[float] | None = None
    batch_size: int | None = Field(default=None, gt=0, le=100_000)
    quality_requirements: dict[str, Any] = Field(default_factory=dict)
    max_workers: int | None = Field(default=None, ge=1, le=1024)
    work_stealing: bool = False
    execution_mode: ExecutionMode = "auto"
    cpu_affinity: bool = False
    seed: int | None = None
    job_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,80}$")
    output_dir: str | None = None
    split_size: int | None = Field(default=None, ge=0)
    allow_no_llm: bool = False
    llm_mode: Literal["auto", "per_record", "pooled"] | None = None
    force: bool = False
    sub_job: SubJobContext | None = None
    links: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _multi_table_size(cls, data: Any) -> Any:
        """A multi-table schema sets its sizes per table; ``total_records`` becomes their (estimated) sum."""
        if isinstance(data, dict) and not data.get("total_records"):
            schema = data.get("schema", data.get("data_schema"))
            if isinstance(schema, dict) and isinstance(schema.get("tables"), dict):
                from app.multi_table import estimate_records  # pylint: disable=import-outside-toplevel

                data = {**data, "total_records": min(max(estimate_records(schema), 1), 50_000_000)}
        return data

    @field_validator("output_format", mode="before")
    @classmethod
    def _lower_format(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_pool(self) -> GenerationRequest:
        if len(set(self.model_pool)) != len(self.model_pool):
            raise ValueError("model_pool contains duplicate models")
        if self.model_weights is not None:
            if self.model_assignment != "weighted":
                raise ValueError("model_weights requires model_assignment='weighted'")
            if len(self.model_weights) != len(self.model_pool) or any(w < 0 for w in self.model_weights) \
                    or sum(self.model_weights) <= 0:
                raise ValueError("model_weights needs one non-negative weight per model_pool entry")
        return self
