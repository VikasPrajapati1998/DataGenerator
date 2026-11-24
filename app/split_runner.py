"""Run very large requests as several sequential sub-jobs and merge them.

A request larger than ``split_size`` (``--split-size`` / ``SDE_MAX_RECORDS_PER_JOB``)
becomes a *split run*:

* the records are divided into balanced sub-jobs; each sub-job is a normal job
  (own directory, plan, batches, guardrails, merge, recovery) whose record
  range continues the global index, so sequence IDs never overlap;
* all sub-jobs share one uniqueness registry (each in its own namespace), so
  unique fields such as e-mails stay unique across the whole dataset;
* category ranges are calibrated once (by the first sub-job) and reused, so all
  sub-jobs follow the same distributions;
* sub-jobs run one after another, which bounds RAM and disk use per sub-job;
  a ``split_run.json`` manifest records progress, so ``resume`` re-runs only
  unfinished sub-jobs;
* finally the sub-job outputs are streamed into one file while counts, column
  order, cross-sub-job duplicates and references are verified.

Layout::

    output/Jobs_<date>/<schema>/<run_id>/
        split_run.json            manifest (parts, status, shared settings)
        uniqueness.sqlite         registry shared by all sub-jobs
        parts/<run_id>_part001/   an ordinary job directory per sub-job
        final/generated_dataset.<fmt>
        metadata/split_summary.json
"""

from __future__ import annotations

import csv
import json
import random
import time
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.exceptions import ConfigurationError, MergeError, StorageError
from app.guardrails.uniqueness import RECORD_KEY
from app.monitoring.logger import get_logger
from app.schemas.input_schema import DatasetSchema, GenerationRequest
from app.storage.file_manager import atomic_writer, read_json, write_json_atomic
from app.storage.formats import count_records, iter_records, read_header
from app.utils.hashing import file_sha256, record_hash, stable_seed, value_hash
from app.utils.helpers import new_job_id, safe_name, utc_now_iso

if TYPE_CHECKING:
    from app.engine import JobResult, SyntheticDataEngine

logger = get_logger(__name__)
MANIFEST = "split_run.json"


def plan_parts(total: int, split_size: int) -> list[dict[str, int]]:
    """Divide ``total`` records into balanced sub-jobs of at most ``split_size``.

    Args:
        total: Requested records.
        split_size: Maximum records per sub-job (positive).

    Returns:
        ``[{"part", "record_offset", "record_count"}, ...]`` covering ``0..total-1``.

    Raises:
        ConfigurationError: If ``split_size`` is not positive.
    """
    if split_size <= 0:
        raise ConfigurationError("split_size must be positive")
    count = -(-total // split_size)
    base, remainder = divmod(total, count)
    parts, offset = [], 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        parts.append({"part": index + 1, "record_offset": offset, "record_count": size})
        offset += size
    return parts


class SplitRunner:
    """Executes and resumes split runs with an existing engine.

    Args:
        engine: The engine that runs each sub-job.
    """

    def __init__(self, engine: SyntheticDataEngine) -> None:
        self.engine = engine

    # -- entry points ---------------------------------------------------------------
    def start(self, request: GenerationRequest, split_size: int) -> JobResult:
        """Create a split run for ``request`` and execute it."""
        entity = request.data_schema.get("entity") if isinstance(request.data_schema.get("fields"), dict) else None
        schema_name = safe_name(request.schema_name or entity)
        run_id = request.job_id or new_job_id(schema_name)
        output_dir = (Path(request.output_dir) if request.output_dir else self.engine.settings.output_dir).resolve()
        run_dir = output_dir / f"Jobs_{time.strftime('%Y-%m-%d')}" / schema_name / run_id
        if (run_dir / MANIFEST).exists():
            raise ConfigurationError(f"Split run {run_id} already exists; use resume to continue it")
        base = request.model_dump(by_alias=True)
        base.update(split_size=0, job_id=None, schema_name=schema_name,
                    seed=request.seed if request.seed is not None else random.SystemRandom().randrange(2 ** 31))
        parts = plan_parts(request.total_records, split_size)
        manifest = {
            "run_id": run_id, "schema_name": schema_name, "created_at": utc_now_iso(), "status": "running",
            "total_records": request.total_records, "split_size": split_size, "output_format": request.output_format,
            "request": base, "profiles": {}, "registry_path": str(run_dir / "uniqueness.sqlite"),
            "parts": [{**p, "job_id": f"{run_id}_part{p['part']:03d}",
                       "job_dir": str(run_dir / "parts" / f"{run_id}_part{p['part']:03d}"), "status": "pending"}
                      for p in parts],
        }
        self._save(run_dir, manifest)
        logger.info("Split run %s: %d records as %d sub-jobs of <= %d records in %s", run_id,
                    request.total_records, len(parts), split_size, run_dir)
        return self._execute(run_dir, manifest)

    def resume(self, run_dir: Path) -> JobResult:
        """Continue a split run; finished sub-jobs are not regenerated."""
        try:
            manifest = read_json(run_dir / MANIFEST)
        except StorageError as exc:
            raise ConfigurationError(f"Split manifest in {run_dir} is unreadable: {exc}") from exc
        if not manifest:
            raise ConfigurationError(f"No {MANIFEST} in {run_dir}")
        logger.info("Resuming split run %s (%d/%d sub-jobs done)", manifest["run_id"],
                    sum(p["status"] == "completed" for p in manifest["parts"]), len(manifest["parts"]))
        return self._execute(run_dir, manifest)

    # -- execution ------------------------------------------------------------------
    def _execute(self, run_dir: Path, manifest: dict[str, Any]) -> JobResult:
        self.engine.availability_cache.clear()
        self.engine.in_split_run = True  # sub-jobs reuse one round of model checks
        try:
            return self._execute_parts(run_dir, manifest)
        finally:
            self.engine.in_split_run = False

    def _execute_parts(self, run_dir: Path, manifest: dict[str, Any]) -> JobResult:
        from app.engine import JobResult  # pylint: disable=import-outside-toplevel

        started = time.perf_counter()
        manifest["status"] = "running"
        for part in manifest["parts"]:
            if part["status"] == "completed" and Path(part.get("final_output") or "").exists():
                continue
            if self.engine.cancel_requested:
                manifest["status"] = "cancelled"
                self._save(run_dir, manifest)
                return JobResult(status="cancelled", job_id=manifest["run_id"], job_dir=str(run_dir),
                                 final_output=None, summary=self._summary(manifest, None, started),
                                 error=f"cancelled; continue with: python -m app.main resume {manifest['run_id']}")
            result = self._run_part(manifest, part)
            part.update(status=result.status, final_output=result.final_output, error=result.error,
                        seconds=(result.summary.get("execution") or {}).get("execution_time_seconds"))
            if result.ok and not manifest["profiles"]:
                config = read_json(Path(part["job_dir"]) / "metadata" / "job_config.json", {}) or {}
                manifest["profiles"] = config.get("profiles", {})
            self._save(run_dir, manifest)
            if not result.ok:
                manifest["status"] = result.status
                self._save(run_dir, manifest)
                reason = (result.error or "").strip()
                reason = reason if len(reason) <= 400 else reason[:400] + "..."
                message = (f"sub-job {part['part']}/{len(manifest['parts'])} {result.status}: {reason} "
                           f"Finished sub-jobs are kept; continue with: python -m app.main resume {manifest['run_id']}")
                logger.error("Split run %s stopped: %s", manifest["run_id"], message)
                return JobResult(status=result.status, job_id=manifest["run_id"], job_dir=str(run_dir),
                                 final_output=None, summary=self._summary(manifest, None, started), error=message,
                                 error_type=result.error_type)
            logger.info("Split run %s: sub-job %d/%d completed", manifest["run_id"], part["part"],
                        len(manifest["parts"]))
        try:
            validation = self._merge(run_dir, manifest)
        except (MergeError, StorageError) as exc:
            manifest["status"] = "failed"
            self._save(run_dir, manifest)
            logger.error("Split run %s: final merge failed: %s", manifest["run_id"], exc)
            return JobResult(status="failed", job_id=manifest["run_id"], job_dir=str(run_dir), final_output=None,
                             summary=self._summary(manifest, None, started), error=f"{type(exc).__name__}: {exc}",
                             error_type=type(exc).__name__)
        manifest.update(status="completed", final_output=validation["output_path"], finished_at=utc_now_iso())
        self._save(run_dir, manifest)
        summary = self._summary(manifest, validation, started)
        write_json_atomic(run_dir / "metadata" / "split_summary.json", summary)
        logger.info("Split run %s completed: %d records in %s", manifest["run_id"], validation["record_count"],
                    validation["output_path"])
        return JobResult(status="completed", job_id=manifest["run_id"], job_dir=str(run_dir),
                         final_output=validation["output_path"], summary=summary)

    def _run_part(self, manifest: dict[str, Any], part: dict[str, Any]) -> JobResult:
        job_dir = Path(part["job_dir"])
        if (job_dir / "metadata" / "job_config.json").exists():
            logger.info("Resuming sub-job %d (%s)", part["part"], part["job_id"])
            return self.engine.resume(job_dir)
        request = dict(manifest["request"])
        request.update(total_records=part["record_count"], seed=stable_seed(request["seed"], part["part"]) % 2 ** 31,
                       sub_job={"part": part["part"], "record_offset": part["record_offset"],
                                "dataset_total_records": manifest["total_records"], "job_dir": str(job_dir),
                                "registry_path": manifest["registry_path"], "profiles": manifest["profiles"],
                                "dataset_seed": manifest["request"]["seed"]})
        logger.info("Starting sub-job %d/%d: global records %d-%d", part["part"], len(manifest["parts"]),
                    part["record_offset"] + 1, part["record_offset"] + part["record_count"])
        return self.engine.run(GenerationRequest(**request))

    @staticmethod
    def _save(run_dir: Path, manifest: dict[str, Any]) -> None:
        write_json_atomic(run_dir / MANIFEST, {**manifest, "updated_at": utc_now_iso()})

    # -- merge and cross-sub-job validation -------------------------------------------------
    def _merge(self, run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        """Stream all sub-job outputs into one file while validating the whole dataset.

        Raises:
            MergeError: On a count, column, duplicate or reference problem (nothing is published).
        """
        fmt = manifest["output_format"]
        config = read_json(Path(manifest["parts"][0]["job_dir"]) / "metadata" / "job_config.json")
        schema = DatasetSchema(**config["schema"])
        columns = schema.columns
        record_fields = schema.record_fields
        hashes: dict[str, array] = {name: array("q") for name in [*schema.unique_fields, RECORD_KEY]}
        ref_fields = {n: s.reference for n, s in schema.fields.items() if s.generator == "reference"}
        refs: dict[str, array] = {n: array("q") for n in ref_fields}
        ids: dict[str, array] = {t: array("q") for t in set(ref_fields.values()) if t}
        target = run_dir / "final" / f"generated_dataset.{fmt}"
        total = 0
        with atomic_writer(target, newline="" if fmt == "csv" else None) as out:
            writer = csv.writer(out) if fmt == "csv" else None
            if writer is not None:
                writer.writerow(columns)
            elif fmt == "json":
                out.write("[")
            for part in manifest["parts"]:
                source = Path(part["final_output"])
                if read_header(source, fmt) != columns:
                    raise MergeError(f"Sub-job {part['part']} has different columns")
                rows = 0
                for record in iter_records(source, fmt):
                    for name in schema.unique_fields:
                        hashes[name].append(value_hash(record.get(name)))
                    if record_fields:
                        hashes[RECORD_KEY].append(record_hash(record, record_fields))
                    for name in ref_fields:
                        if record.get(name) not in (None, ""):
                            refs[name].append(int(record[name]))
                    for name in ids:
                        ids[name].append(int(record[name]))
                    if writer is not None:
                        writer.writerow(["" if record.get(c) is None else record.get(c) for c in columns])
                    elif fmt == "jsonl":
                        out.write(json.dumps({c: record.get(c) for c in columns}, ensure_ascii=False) + "\n")
                    else:
                        out.write(",\n" if total else "\n")
                        out.write(json.dumps({c: record.get(c) for c in columns}, ensure_ascii=False))
                    rows += 1
                    total += 1
                if rows != part["record_count"]:
                    raise MergeError(f"Sub-job {part['part']} holds {rows} records, expected {part['record_count']}")
            if fmt == "json":
                out.write("\n]\n" if total else "]\n")
            duplicates = {name: len(values) - len(np.unique(np.frombuffer(values, dtype=np.int64)))
                          for name, values in hashes.items() if len(values)}
            if any(duplicates.values()):
                raise MergeError(f"Duplicates across sub-jobs: {duplicates}")
            broken = {name: int((~np.isin(np.frombuffer(values, dtype=np.int64),
                                          np.frombuffer(ids[ref_fields[name]], dtype=np.int64))).sum())
                      for name, values in refs.items() if len(values)}
            if any(broken.values()):
                raise MergeError(f"References to missing records: {broken}")
            if total != manifest["total_records"]:
                raise MergeError(f"Merged {total} records but {manifest['total_records']} were requested")
        final_count = count_records(target, fmt)
        if final_count != manifest["total_records"]:
            raise MergeError(f"Final file holds {final_count} records, expected {manifest['total_records']}")
        return {"output_path": str(target), "record_count": final_count, "duplicates_across_sub_jobs": 0,
                "broken_references": 0, "file_size": target.stat().st_size, "checksum": file_sha256(target),
                "passed": True}

    @staticmethod
    def _summary(manifest: dict[str, Any], validation: dict[str, Any] | None, started: float) -> dict[str, Any]:
        part_summaries = []
        quality_scores = []
        for part in manifest["parts"]:
            execution = read_json(Path(part["job_dir"]) / "metadata" / "execution_summary.json", {}) or {}
            checks = read_json(Path(part["job_dir"]) / "metadata" / "validation_summary.json", {}) or {}
            score = (checks.get("quality") or {}).get("score")
            if score is not None:
                quality_scores.append(score)
            part_summaries.append({"part": part["part"], "job_id": part["job_id"], "status": part["status"],
                                   "records": part["record_count"], "dynamic_tasks": execution.get("dynamic_tasks"),
                                   "seconds": execution.get("execution_time_seconds"), "quality_score": score})
        done = sum(p["record_count"] for p in manifest["parts"] if p["status"] == "completed")
        elapsed = time.perf_counter() - started
        execution = {
            "split_run": True, "sub_jobs": len(manifest["parts"]), "split_size": manifest["split_size"],
            "valid_records_generated": done, "requested_records": manifest["total_records"],
            "dynamic_tasks": sum(p.get("dynamic_tasks") or 0 for p in part_summaries),
            "execution_time_seconds": round(sum(p.get("seconds") or 0 for p in part_summaries), 2),
            "this_session_seconds": round(elapsed, 2),
            "records_per_second": round(done / elapsed, 1) if elapsed else 0.0, "parts": part_summaries,
        }
        quality = {"score": min(quality_scores)} if quality_scores else None
        return {"execution": execution, "validation": {"cross_sub_job": validation, "quality": quality}}
