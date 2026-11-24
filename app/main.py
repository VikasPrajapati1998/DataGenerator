"""Command-line interface.

Examples::

    python -m app.main generate --config samples/schemas/employees.json
    python -m app.main generate --config samples/schemas/employees.json --records 5000 --work-stealing
    python -m app.main resume 20261120_120000_ab12cd
    python -m app.main cancel 20261120_120000_ab12cd
    python -m app.main plan --records 100000 --batch-size 1000
    python -m app.main models check
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.config.model_registry import load_registry
from app.config.settings import get_settings
from app.core.cancellation import CancellationToken
from app.core.exceptions import EngineError
from app.models.model_factory import create_adapter
from app.monitoring.logger import redact
from app.scheduler.cpu_detector import detect_resources
from app.storage.file_manager import find_job_dir
from app.scheduler.task_allocator import allocate_to_cores, calculate_dynamic_nodes, plan_tasks


def _load_request(args: argparse.Namespace) -> dict[str, Any]:
    request = json.loads(Path(args.config).read_text(encoding="utf-8"))
    overrides = {"total_records": args.records, "output_format": args.format, "model_provider": args.provider,
                 "model_name": args.model, "batch_size": args.batch_size, "max_workers": args.max_workers,
                 "execution_mode": args.mode, "seed": args.seed, "job_id": args.job_id,
                 "output_dir": args.output_dir, "split_size": args.split_size}
    request.update({k: v for k, v in overrides.items() if v is not None})
    request.setdefault("schema_name", args.schema_name or Path(args.config).stem)
    if args.schema_name:
        request["schema_name"] = args.schema_name
    if args.work_stealing:
        request["work_stealing"] = True
    if args.cpu_affinity:
        request["cpu_affinity"] = True
    if args.allow_no_llm:
        request["allow_no_llm"] = True
    if args.force:
        request["force"] = True
    if args.llm_mode:
        request["llm_mode"] = args.llm_mode
    return request


EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_BAD_INPUT = 2
EXIT_NO_MODEL = 3
EXIT_RESOURCES = 4
EXIT_STORAGE = 5
EXIT_INTERNAL = 70
EXIT_INTERRUPTED = 130

_EXIT_BY_ERROR = {
    "SchemaError": EXIT_BAD_INPUT, "ConfigurationError": EXIT_BAD_INPUT, "ValidationError": EXIT_BAD_INPUT,
    "ModelUnavailableError": EXIT_NO_MODEL, "QuotaExceededError": EXIT_NO_MODEL,
    "ResourceError": EXIT_RESOURCES, "FeasibilityError": EXIT_RESOURCES,
    "StorageError": EXIT_STORAGE, "MergeError": EXIT_STORAGE, "RecoveryError": EXIT_STORAGE,
    "CancelledError": EXIT_INTERRUPTED,
}
_HINTS = {
    EXIT_BAD_INPUT: "Fix the request/schema file (the field is named above) or the SDE_* settings in .env.",
    EXIT_NO_MODEL: "Start Ollama with `ollama serve`, check OLLAMA_HOST / SDE_MODEL_POOL in .env, "
                   "or add --allow-no-llm to use placeholder text for LLM fields.",
    EXIT_RESOURCES: "Free RAM or disk space, lower --max-workers, or set SDE_RAM_CHECK=warn in .env.",
    EXIT_STORAGE: "Check free disk space and write permission for the output folder.",
    EXIT_INTERRUPTED: "Stopped on request; finished work is kept.",
    EXIT_JOB_FAILED: "Some batches could not be completed; see the log file. Resume retries only those.",
}
_HINTS_BY_ERROR = {
    "FeasibilityError": "The LLM is too slow for this many records. Use --llm-mode pooled, --allow-no-llm, "
                        "fewer --records or a faster/cloud model; raise SDE_MAX_ETA_HOURS or add --force to run anyway.",
}


def _hint(code: int, error_type: str | None) -> str | None:
    """Hint for a failed run: specific to the error type when one exists, else per exit code."""
    return _HINTS_BY_ERROR.get(error_type or "") or _HINTS.get(code)


def _short(text: str | None, limit: int = 500) -> str | None:
    """First line of an error, capped in length (the full text is in the log file)."""
    if not text:
        return None
    line = text.strip().splitlines()[0]
    return line if len(line) <= limit else line[:limit].rstrip() + "..."


def exit_code_for(result: Any) -> int:
    """Map a job result to the CLI exit code."""
    if result.ok:
        return EXIT_OK
    if result.status == "cancelled":
        return EXIT_INTERRUPTED
    return _EXIT_BY_ERROR.get(result.error_type or "", EXIT_JOB_FAILED)


def _resume_command(result: Any) -> str | None:
    if result.ok or not result.job_dir or not result.job_id:
        return None
    job_dir = Path(result.job_dir)
    if (job_dir / "metadata" / "job_config.json").exists() or (job_dir / "split_run.json").exists():
        return f"python -m app.main resume {result.job_id}"
    return None


def _print_result(result: Any) -> int:
    execution = result.summary.get("execution", {})
    code = exit_code_for(result)
    resume = _resume_command(result)
    payload = {
        "status": result.status, "job_id": result.job_id, "job_dir": result.job_dir,
        "final_output": result.final_output,
        "records": execution.get("valid_records_generated"),
        "dynamic_tasks": execution.get("dynamic_tasks"), "queues": execution.get("effective_queues"),
        "execution_time_seconds": execution.get("execution_time_seconds"),
        "records_per_second": execution.get("records_per_second"),
        "quality_score": (result.summary.get("validation", {}).get("quality") or {}).get("score"),
        **({"sub_jobs": execution.get("sub_jobs")} if execution.get("split_run") else {}),
    }
    if code != EXIT_OK:
        payload.update(error=_short(result.error), error_type=result.error_type, hint=_hint(code, result.error_type),
                       resume=resume, exit_code=code)
    print(json.dumps(payload, indent=2))
    if code != EXIT_OK:
        print(f"ERROR: {_short(result.error) or result.status}", file=sys.stderr)
        print(f"Hint: {_hint(code, result.error_type)}", file=sys.stderr)
        if resume:
            print(f"Resume with: {resume}", file=sys.stderr)
    return code


def _hard_exit(code: int) -> None:
    """Exit immediately (an abandoned worker thread must not keep the process alive)."""
    logging.shutdown()
    os._exit(code)


def _run_engine(action: str, payload: Any) -> int:
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    engine = SyntheticDataEngine()
    interrupts = {"count": 0}

    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        interrupts["count"] += 1
        if interrupts["count"] == 1:
            print("\nCancelling gracefully (press Ctrl+C again to abort immediately)...", file=sys.stderr)
            engine.cancel()
        else:
            raise KeyboardInterrupt

    def terminate(signum: int, frame: Any) -> None:  # noqa: ARG001
        print("\nTermination requested; stopping workers...", file=sys.stderr)
        engine.cancel()
        raise KeyboardInterrupt

    signals = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])
    previous = {sig: signal.getsignal(sig) for sig in signals}
    signal.signal(signal.SIGINT, handler)
    for sig in signals[1:]:  # SIGTERM, and on Windows Ctrl+Break / console close
        signal.signal(sig, terminate)
    atexit.register(engine.shutdown_now)  # last line of defence: never leave worker processes behind
    try:
        result = engine.run(payload) if action == "generate" else engine.resume(payload)
    except KeyboardInterrupt:
        engine.shutdown_now()
        print("\nAborted. Finished batches are kept; resume with: python -m app.main resume <job_id>",
              file=sys.stderr)
        _hard_exit(EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED  # only reached when _hard_exit is replaced (tests)
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)
        atexit.unregister(engine.shutdown_now)
    return _print_result(result)


def cmd_plan(args: argparse.Namespace) -> int:
    """Print the power-of-two plan and core allocation without generating data."""
    cores = args.cores or detect_resources().cpu_cores
    tasks = plan_tasks(args.records, args.batch_size)
    allocation = allocate_to_cores([t.task_id for t in tasks], cores)
    print(f"records={args.records} batch_size={args.batch_size} "
          f"power_of_two_tasks={calculate_dynamic_nodes(args.records, args.batch_size)} tasks={len(tasks)} "
          f"cores={cores}")
    counts = sorted({t.record_count for t in tasks})
    print(f"records per task: {counts[0]}..{counts[-1]} (sum {sum(t.record_count for t in tasks)})")
    for core, ids in allocation.items():
        span = f"{ids[0]}-{ids[-1]}" if len(ids) > 1 else str(ids[0])
        print(f"  {core:<8} tasks {span:<12} ({len(ids)} tasks)")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List registered models or check their availability."""
    registry = load_registry()
    settings = get_settings()
    for config in registry.all():
        line = f"{config.key:<45} {config.execution:<6} structured={config.structured_output}"
        if args.action == "check":
            adapter = create_adapter(config, settings)
            try:
                result = adapter.check_availability()
                line += f"  -> {'AVAILABLE' if result.available else 'unavailable'}: {redact(result.detail)}"
            except EngineError as exc:
                line += f"  -> unavailable: {redact(str(exc))}"
            finally:
                adapter.close()
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="python -m app.main", description="Synthetic data generation engine")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on the console")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a dataset from a request JSON file")
    gen.add_argument("--config", required=True, help="request JSON (schema, total_records, ...)")
    gen.add_argument("--records", type=int)
    gen.add_argument("--format", choices=["csv", "json", "jsonl"])
    gen.add_argument("--provider")
    gen.add_argument("--model")
    gen.add_argument("--batch-size", type=int)
    gen.add_argument("--max-workers", type=int)
    gen.add_argument("--mode", choices=["auto", "thread", "process"])
    gen.add_argument("--seed", type=int)
    gen.add_argument("--job-id")
    gen.add_argument("--schema-name", help="job-id prefix and output folder (default: config file name)")
    gen.add_argument("--split-size", type=int,
                     help="max records per sub-job; larger requests run as sequential sub-jobs merged at the end "
                          "(default SDE_MAX_RECORDS_PER_JOB=100000, 0 disables)")
    gen.add_argument("--output-dir")
    gen.add_argument("--work-stealing", action="store_true")
    gen.add_argument("--cpu-affinity", action="store_true")
    gen.add_argument("--allow-no-llm", action="store_true",
                     help="if no model is reachable, fill LLM text fields with placeholder text instead of failing")
    gen.add_argument("--llm-mode", choices=["auto", "per_record", "pooled"],
                     help="per_record: one LLM answer per record; pooled: build a text pool once and recombine it "
                          "(fast); auto (default): pooled from SDE_POOLED_MIN_RECORDS records")
    gen.add_argument("--force", action="store_true",
                     help="start even if the projected run time exceeds SDE_MAX_ETA_HOURS")

    res = sub.add_parser("resume", help="resume an interrupted or failed job")
    res.add_argument("job", help="job id or job directory")

    can = sub.add_parser("cancel", help="request cancellation of a running job (from another terminal)")
    can.add_argument("job", help="job id or job directory")

    plan = sub.add_parser("plan", help="show the task plan and core allocation")
    plan.add_argument("--records", type=int, required=True)
    plan.add_argument("--batch-size", type=int, default=1000)
    plan.add_argument("--cores", type=int)

    models = sub.add_parser("models", help="list or check models")
    models.add_argument("action", choices=["list", "check"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    if args.debug:
        os.environ["SDE_DEBUG"] = "1"  # read when logging is configured: tracebacks on the console too
    try:
        if args.command == "generate":
            return _run_engine("generate", _load_request(args))
        if args.command == "resume":
            return _run_engine("resume", args.job)
        if args.command == "cancel":
            job_dir = find_job_dir(get_settings().output_dir, args.job)
            if job_dir is None:
                return _fail(EXIT_BAD_INPUT, f"job {args.job} not found under {get_settings().output_dir}")
            CancellationToken(job_dir).cancel()
            print(f"Cancellation requested for {job_dir}")
            return EXIT_OK
        if args.command == "plan":
            return cmd_plan(args)
        return cmd_models(args)
    except FileNotFoundError as exc:
        return _fail(EXIT_BAD_INPUT, f"file not found: {exc.filename}")
    except json.JSONDecodeError as exc:
        return _fail(EXIT_BAD_INPUT, f"invalid JSON in {getattr(args, 'config', 'input')}: "
                                     f"line {exc.lineno}, column {exc.colno}: {exc.msg}")
    except PydanticValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:10])
        return _fail(EXIT_BAD_INPUT, f"invalid request: {problems}")
    except EngineError as exc:
        code = _EXIT_BY_ERROR.get(type(exc).__name__, EXIT_BAD_INPUT)
        return _fail(code, _short(redact(f"{type(exc).__name__}: {exc}")) or type(exc).__name__)
    except PermissionError as exc:
        return _fail(EXIT_STORAGE, f"permission denied: {exc.filename or exc}")
    except OSError as exc:
        return _fail(EXIT_STORAGE, redact(f"{type(exc).__name__}: {exc}"))
    except ValueError as exc:
        return _fail(EXIT_BAD_INPUT, redact(str(exc)))
    except KeyboardInterrupt:
        return _fail(EXIT_INTERRUPTED, "aborted")
    except Exception as exc:  # noqa: BLE001 - last resort: a bug must still end with a clean message
        logging.getLogger("sde").critical("Internal error", exc_info=True)
        if args.debug:
            traceback.print_exc()
        return _fail(EXIT_INTERNAL, f"internal error ({type(exc).__name__}: {redact(str(exc))[:200]}). "
                                    "The full traceback is in the log file; please report it")


def _fail(code: int, message: str) -> int:
    """Print one clean error line plus a hint, and return the exit code."""
    print(f"ERROR: {message}", file=sys.stderr)
    if hint := _HINTS.get(code):
        print(f"Hint: {hint}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
