"""Pooled LLM mode: build a pool of LLM texts once, then reuse it for every record.

Per-record LLM generation costs one model answer per record, which a small
CPU-only model delivers at a few seconds each: days for millions of records.
Pooled mode instead asks the models for a limited number of answers (as many
as fit in ``SDE_POOL_BUILD_MINUTES``) and stores them in ``llm_pool.json``.

Each answer is turned into a **template** before it is stored: values of the
sample record the model saw (name, amounts, dates, city...) are replaced by
placeholders such as ``{insured_name}`` or ``{claim_amount_inr:in}``. A record
then gets a text assembled from one or two templates that match its categories,
filled with **its own** values. So the text agrees with the record, and records
differ from each other even with a small pool.

Answers are matched on the record's categorical fields. When an answer names
a different category than the one it was written for (a "health" claim text
for a "motor" record), it is filed under the category it names.

Every composed text is checked against the same rules as LLM output (length,
minimum words, forbidden content) before it is used: invalid combinations are
never handed to the guardrails.

While records are generated, :class:`PoolGrower` keeps adding answers in the
background up to ``SDE_POOL_MAX_VALUES``; the larger pool is saved when the job
(or sub-job) ends, so later sub-jobs use it.
"""

from __future__ import annotations

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from faker import Faker

from app.core.exceptions import EngineError, LLMError, StorageError
from app.generation.llm_enricher import LLMEnricher
from app.guardrails.prompt_safety import check_generated_text
from app.models.model_router import LLMUsage
from app.monitoring.logger import get_logger
from app.schemas.input_schema import DatasetSchema
from app.storage.file_manager import read_json, write_json_atomic
from app.utils.hashing import stable_seed

logger = get_logger(__name__)
POOL_FORMAT = 2
_MAX_CONTEXT_VALUES = 20
_MIN_WORDS = 3  # same floor as the realism validator's "too short to be informative"
_COMPOSE_TRIES = 6
MIN_POOL_VALUES = 30
_ABBREVIATIONS = {"rs", "mr", "mrs", "ms", "dr", "no", "st", "ltd", "co", "vs", "inc", "jr", "sr", "e.g", "i.e",
                  "approx", "dept", "govt", "pvt", "sec"}
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9{])")
_DOLLAR = re.compile(r"(?:\$|\bUSD\s?)\s*(?=\{)")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::([a-z]+))?\}")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


# -- schema helpers ---------------------------------------------------------------------------------
def context_fields(schema: DatasetSchema) -> list[str]:
    """Categorical fields whose values pooled answers are matched on."""
    return [n for n, s in schema.fields.items()
            if s.allowed_values and len(s.allowed_values) <= _MAX_CONTEXT_VALUES and s.generator != "llm"]


def _template_fields(schema: DatasetSchema) -> list[str]:
    """Fields whose values are replaced by placeholders (everything but LLM, categorical and boolean fields)."""
    contexts = set(context_fields(schema))
    return [n for n, s in schema.fields.items()
            if s.generator != "llm" and n not in contexts and s.type != "boolean"]


# -- value formats ----------------------------------------------------------------------------------
def _indian(number: int) -> str:
    """``1234567`` -> ``12,34,567`` (Indian digit grouping)."""
    sign, digits = ("-", str(-number)) if number < 0 else ("", str(number))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return sign + ",".join([*groups, tail])


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _format(value: Any, fmt: str | None) -> str:
    """Render ``value`` in the format a placeholder asks for."""
    if fmt in ("comma", "in") and isinstance(value, (int, float)) and not isinstance(value, bool):
        number = int(value)
        return f"{number:,}" if fmt == "comma" else _indian(number)
    if fmt in ("long", "dmy"):
        day = _as_date(value)
        if day is not None:
            month = _MONTHS[day.month - 1]
            return f"{month} {day.day}, {day.year}" if fmt == "long" else f"{day.day} {month} {day.year}"
    return str(value)


def _variants(value: Any) -> list[tuple[str | None, str]]:
    """``(format, text)`` pairs under which ``value`` may appear in a model answer."""
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        number = int(value)
        if number != value or abs(number) < 10:
            return []  # small numbers ("2 cars") are too likely to be ordinary words of the text
        return list({text: (fmt, text) for fmt, text in
                     ((None, str(number)), ("comma", f"{number:,}"), ("in", _indian(number)))}.values())
    day = _as_date(value) if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", str(value)) else None
    if day is not None:
        return [(None, day.isoformat()), ("long", _format(day, "long")), ("dmy", _format(day, "dmy"))]
    text = str(value).strip()
    return [(None, text)] if len(text) >= 3 else []


def templatize(text: str, record: dict[str, Any], fields: list[str]) -> str:
    """Replace the record's own values in ``text`` by placeholders.

    Longer values are replaced first, so ``2025-06-14`` wins over ``2025``.
    Matching is whole-word and case-insensitive for text values.
    """
    found: list[tuple[str, str, str | None]] = []
    for name in fields:
        for fmt, variant in _variants(record.get(name)):
            found.append((variant, name, fmt))
    marks: dict[str, str] = {}
    for index, (variant, name, fmt) in enumerate(sorted(found, key=lambda f: -len(f[0]))):
        pattern = re.compile(rf"(?<![\w,]){re.escape(variant)}(?![\w,]|\.\d)", re.IGNORECASE)
        mark = f"\x00{chr(0xE000 + index)}\x00"  # no digits or letters a later value could match
        text, count = pattern.subn(mark, text)
        if count:
            marks[mark] = "{" + name + (f":{fmt}" if fmt else "") + "}"
    for mark, placeholder in marks.items():
        text = text.replace(mark, placeholder)
    return text


def render(template: str, record: dict[str, Any]) -> str | None:
    """Fill a template with ``record``'s values (``None`` if a needed value is missing)."""
    missing = False

    def value(match: re.Match[str]) -> str:
        nonlocal missing
        current = record.get(match.group(1))
        if current is None:
            missing = True
            return ""
        return _format(current, match.group(2))

    text = _PLACEHOLDER.sub(value, template)
    return None if missing else " ".join(text.split())


def sentences(text: str) -> list[str]:
    """Split into sentences, ignoring the dot of abbreviations such as ``Rs.`` or ``Dr.``."""
    parts: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        words = text[start:match.start()].split()
        word = words[-1].lower().rstrip(".") if words else ""
        if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
            continue
        parts.append(text[start:match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def relabel(context: dict[str, Any], text: str, schema: DatasetSchema) -> dict[str, Any]:
    """File an answer under the category it actually names (when it names exactly one)."""
    lowered = " " + " ".join(re.findall(r"[a-z0-9]+", text.casefold())) + " "
    updated = dict(context)
    for name in context:
        values = schema.fields[name].allowed_values or []
        if not all(isinstance(v, str) for v in values):
            continue
        named = [v for v in values if f" {' '.join(re.findall(r'[a-z0-9]+', v.casefold()))} " in lowered]
        if len(named) == 1:
            updated[name] = named[0]
    return updated


def _pool_item(schema: DatasetSchema, record: dict[str, Any], contexts: list[str], fields: list[str]
               ) -> dict[str, Any] | None:
    values: dict[str, str] = {}
    for name in schema.llm_fields:
        text = record.get(name)
        if not isinstance(text, str) or not text.strip() or "{" in text or "}" in text:
            return None
        template = templatize(" ".join(text.split()), record, fields)
        if schema.locale == "en_IN":
            # Small models often write dollar amounts for Indian data; the amounts here are rupees.
            template = _DOLLAR.sub("Rs. ", template)
        values[name] = template
    context = {c: record.get(c) for c in contexts}
    first = values[schema.llm_fields[0]] if schema.llm_fields else ""
    return {"context": relabel(context, first, schema), "values": values}


# -- building -----------------------------------------------------------------------------------------
def _collect(schema: DatasetSchema, builder: Any, enricher: LLMEnricher, rng: random.Random, faker: Faker,
             usage: LLMUsage) -> list[dict[str, Any]]:
    """One LLM request: sample records, their answers, turned into pool items."""
    contexts, fields = context_fields(schema), _template_fields(schema)
    size = enricher.chunk_size
    records = [builder.build(rng.randrange(1_000_000), rng, faker) for _ in range(size)]
    failures = enricher.enrich(records, list(range(size)), rng, usage)
    items = [_pool_item(schema, r, contexts, fields) for p, r in enumerate(records) if p not in failures]
    return [i for i in items if i is not None]


def build_value_pool(schema: DatasetSchema, builder: Any, enrichers: list[LLMEnricher], target: int,
                     budget_seconds: float, seed: int, usage: LLMUsage) -> list[dict[str, Any]]:
    """Ask the models for up to ``target`` answers within ``budget_seconds``.

    Each enricher (one per usable model) works in its own thread on freshly
    generated sample records, so several models fill the pool in parallel.

    Args:
        schema: The normalised schema.
        builder: The job's ``RecordBuilder`` (provides realistic contexts).
        enrichers: One LLM enricher per model to use.
        target: Desired number of pooled answers.
        budget_seconds: Time budget; building stops when it is used up.
        seed: Seed for sample contexts.
        usage: LLM usage counters (merged across threads).

    Returns:
        Pool items ``{"context": {...}, "values": {field: template}}``.
    """
    items: list[dict[str, Any]] = []
    lock = threading.Lock()
    deadline = time.monotonic() + budget_seconds

    def work(worker: int, enricher: LLMEnricher) -> None:
        rng = random.Random(stable_seed(seed, "pool", worker))
        faker = Faker(schema.locale)
        faker.seed_instance(stable_seed(seed, "pool-faker", worker))
        local = LLMUsage(listener=usage.listener)
        errors = 0
        while time.monotonic() < deadline:
            with lock:
                if len(items) >= target:
                    break
            try:
                fresh = _collect(schema, builder, enricher, rng, faker, local)
            except LLMError as exc:
                logger.warning("Text pool: a request to %s failed (%s)", enricher.router.primary.key,
                               type(exc).__name__)
                errors += 1
                if errors > 3 and not items:
                    break
                continue
            with lock:
                items.extend(fresh)
                count = len(items)
            logger.info("Text pool: %d/%d answers", min(count, target), target)
        with lock:
            usage.merge(local)

    with ThreadPoolExecutor(max_workers=max(1, len(enrichers)), thread_name_prefix="pool-build") as pool:
        for future in [pool.submit(work, i, e) for i, e in enumerate(enrichers)]:
            future.result()
    return items[:target]


def save_pool(path: Path, schema: DatasetSchema, items: list[dict[str, Any]], plan: dict[str, Any] | None = None
              ) -> None:
    """Persist the pool atomically (reused by resumed jobs and by every sub-job of a split run)."""
    previous = read_json(path, {}) or {}
    write_json_atomic(path, {"format": POOL_FORMAT, "fields": schema.llm_fields,
                             "context_fields": context_fields(schema),
                             "plan": plan if plan is not None else previous.get("plan"), "items": items})


def load_pool(path: Path) -> dict[str, Any]:
    """Read ``llm_pool.json`` (``{}`` when absent)."""
    return read_json(path, {}) or {}


# -- filling ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Entry:
    """A pooled answer, pre-split into sentences (filling runs for millions of records)."""

    context: dict[str, Any]
    texts: dict[str, str]
    parts: dict[str, list[str]]


class PooledFiller:
    """Fills LLM fields of records from a stored pool.

    Args:
        schema: The normalised schema.
        pool_path: ``llm_pool.json``.
        max_text_length: Upper bound for generated text.

    Raises:
        StorageError: If the pool is missing, unreadable or empty.
    """

    def __init__(self, schema: DatasetSchema, pool_path: Path, max_text_length: int) -> None:
        data = load_pool(pool_path)
        if not data.get("items"):
            raise StorageError(f"LLM text pool {pool_path} is missing or empty")
        self.schema = schema
        self.contexts: list[str] = list(data.get("context_fields") or [])
        self.templated = int(data.get("format", 1)) >= POOL_FORMAT
        if not self.templated:
            logger.warning("Text pool %s was built by an older version: its texts are used as they are (they may "
                           "name other people or amounts than the record). Delete it to build a new pool.",
                           pool_path)
        self.limits = {f: min(schema.fields[f].max_length or max_text_length, max_text_length)
                       for f in schema.llm_fields}
        # The realism validator rejects texts under 3 words unless the field sets its own min_length.
        self.min_words = {f: 1 if schema.fields[f].min_length else _MIN_WORDS for f in schema.llm_fields}
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._by_context: dict[tuple[Any, ...], list[_Entry]] = {}
        self._by_first: dict[Any, list[_Entry]] = {}
        self.extend(data["items"])

    @property
    def size(self) -> int:
        """Number of pooled answers."""
        return len(self._entries)

    def extend(self, items: list[dict[str, Any]]) -> None:
        """Add answers (the background grower calls this while tasks run)."""
        entries = []
        for item in items:
            texts = {k: str(v) for k, v in (item.get("values") or {}).items() if v}
            entries.append(_Entry(item.get("context") or {}, texts, {k: sentences(v) for k, v in texts.items()}))
        with self._lock:
            for entry in entries:
                self._entries.append(entry)
                self._by_context.setdefault(tuple(entry.context.get(c) for c in self.contexts), []).append(entry)
                if self.contexts:
                    self._by_first.setdefault(entry.context.get(self.contexts[0]), []).append(entry)

    def _candidates(self, record: dict[str, Any], widen: bool) -> list[_Entry]:
        # Lists are only ever appended to, so reading them without the lock is safe.
        first = self._by_first.get(record.get(self.contexts[0])) if self.contexts else None
        if widen:
            return first or self._entries
        exact = self._by_context.get(tuple(record.get(c) for c in self.contexts))
        if exact and len(exact) >= 2:
            return exact
        return first if first and len(first) >= 2 else self._entries

    def _valid(self, name: str, text: str | None) -> bool:
        return (text is not None and len(text.split()) >= self.min_words[name]
                and check_generated_text(text, self.limits[name]) is None)

    def _render(self, template: str, record: dict[str, Any]) -> str | None:
        return render(template, record) if self.templated else " ".join(template.split())

    def _compose(self, name: str, record: dict[str, Any], candidates: list[_Entry], rng: random.Random
                 ) -> str | None:
        """A valid text for ``record``: two answers recombined, else one answer, else ``None``."""
        for _ in range(_COMPOSE_TRIES):
            first, second = rng.choice(candidates), rng.choice(candidates)
            if name not in first.texts:
                continue
            opening = first.parts[name] or [first.texts[name]]
            closing = second.parts.get(name) or opening
            options = []
            if len(closing) > 1:
                options.append(f"{opening[0]} {rng.choice(closing[1:])}")
            if closing[0] != opening[0]:
                options.append(f"{opening[0]} {closing[0]}")
            options += [first.texts[name], opening[0]]
            for option in options:
                text = self._render(option, record)
                if self._valid(name, text):
                    return text
        return None

    def fill(self, records: list[dict[str, Any]], positions: list[int], rng: random.Random,
             widen: bool = False) -> dict[int, str]:
        """Fill the LLM fields of ``records[p]`` for every ``p`` in ``positions``.

        Args:
            records: The batch (modified in place).
            positions: Indices to fill.
            rng: Random source.
            widen: Rescue rounds: match on the first category only, so more answers qualify.

        Returns:
            ``position -> reason`` for records no valid text could be composed for.
        """
        failures: dict[int, str] = {}
        for pos in positions:
            candidates = self._candidates(records[pos], widen)
            for name in self.schema.llm_fields:
                text = self._compose(name, records[pos], candidates, rng)
                if text is None:
                    failures[pos] = f"llm: no valid pooled text for {name}"
                    break
                records[pos][name] = text
        return failures

class PoolGrower:
    """Adds answers to the pool in a background thread while records are generated.

    Pooled tasks make no LLM calls, so the model would otherwise sit idle. New
    answers go straight into the in-process :class:`PooledFiller` (thread mode)
    and are written to ``llm_pool.json`` by :meth:`stop`, so the next sub-job (or
    a resumed job) starts with the larger pool.

    Args:
        schema: The normalised schema.
        builder: The job's ``RecordBuilder``.
        enrichers: One enricher per model.
        filler: The in-process filler to extend.
        pool_path: ``llm_pool.json``.
        target: Stop when the pool reaches this size.
        seed: Seed for sample records.
        usage: LLM usage counters.
    """

    def __init__(self, schema: DatasetSchema, builder: Any, enrichers: list[LLMEnricher], filler: PooledFiller,
                 pool_path: Path, target: int, seed: int, usage: LLMUsage) -> None:
        self.schema, self.builder, self.enrichers, self.filler = schema, builder, enrichers, filler
        self.pool_path, self.target, self.seed, self.usage = pool_path, target, seed, usage
        self.new_items: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start one background thread per model."""
        logger.info("Text pool: growing in the background from %d towards %d answers (SDE_POOL_GROW)",
                    self.filler.size, self.target)
        for worker, enricher in enumerate(self.enrichers):
            thread = threading.Thread(target=self._run, args=(worker, enricher), name=f"pool-grow-{worker}",
                                      daemon=True)
            thread.start()
            self._threads.append(thread)

    def _run(self, worker: int, enricher: LLMEnricher) -> None:
        rng = random.Random(stable_seed(self.seed, "pool-grow", worker, self.filler.size))
        faker = Faker(self.schema.locale)
        faker.seed_instance(stable_seed(self.seed, "pool-grow-faker", worker, self.filler.size))
        errors = 0
        while not self._stop.is_set() and self.filler.size < self.target and errors < 3:
            try:
                fresh = _collect(self.schema, self.builder, enricher, rng, faker, self.usage)
            except LLMError as exc:
                errors += 1
                logger.info("Text pool growth: a request failed (%s)", type(exc).__name__)
                continue
            except (EngineError, RuntimeError, OSError) as exc:
                # Typically the job ended and closed the HTTP client under an in-flight request.
                if not self._stop.is_set():
                    logger.warning("Text pool growth stopped: %s: %s", type(exc).__name__, exc)
                return
            if self._stop.is_set():
                break  # the job is ending: this answer would arrive after the pool was saved
            with self._lock:
                self.new_items.extend(fresh)
            self.filler.extend(fresh)

    def stop(self) -> int:
        """Stop growing and save the new answers; returns how many were added.

        A request still in flight is not awaited (its thread is a daemon and its
        answer is discarded).
        """
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=0.5)
        with self._lock:
            fresh, self.new_items = self.new_items, []
        if not fresh:
            return 0
        data = load_pool(self.pool_path)
        items = [*data.get("items", []), *fresh][:max(self.target, len(data.get("items", [])))]
        save_pool(self.pool_path, self.schema, items)
        logger.info("Text pool: %d answer(s) added in the background, %d in total", len(fresh), len(items))
        return len(fresh)
