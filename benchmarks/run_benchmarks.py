"""Throughput and resource benchmarks.

Usage (from the project root)::

    python benchmarks/run_benchmarks.py                       # 1k, 10k, 100k with the mock model
    python benchmarks/run_benchmarks.py --sizes 1000000       # 1M records
    python benchmarks/run_benchmarks.py --modes thread process --sizes 100000
    python benchmarks/run_benchmarks.py --ollama --sizes 64   # real local model, LLM text field

Results are appended to ``benchmarks/results/benchmark_results.json`` and
rendered to ``benchmarks/results/benchmark_results.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine import SyntheticDataEngine  # noqa: E402
from app.scheduler.cpu_detector import detect_resources  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
EMPLOYEES = json.loads((ROOT / "samples" / "schemas" / "employees_mock.json").read_text(encoding="utf-8"))
LLM_SCHEMA = {"customer_id": "unique integer", "name": "realistic full name", "segment": ["retail", "business"],
              "bio": "one-sentence description of the customer's interests"}


def _row(result: Any, label: str, size: int, mode: str) -> dict[str, Any]:
    execution = result.summary.get("execution", {})
    validation = result.summary.get("validation", {})
    return {
        "label": label, "records": size, "mode": execution.get("execution_mode", mode), "status": result.status,
        "tasks": execution.get("dynamic_tasks"), "queues": execution.get("effective_queues"),
        "seconds": execution.get("execution_time_seconds"), "records_per_second": execution.get("records_per_second"),
        "cpu_avg_pct": (execution.get("cpu_utilization_pct") or {}).get("avg"),
        "cpu_peak_pct": (execution.get("cpu_utilization_pct") or {}).get("peak"),
        "engine_ram_peak_mb": (execution.get("engine_ram_mb") or {}).get("peak"),
        "rejected": execution.get("rejected_records"), "duplicate_rejections": execution.get("duplicate_rejections"),
        "regeneration_rounds": execution.get("regeneration_rounds"), "task_retries": execution.get("task_retries"),
        "llm_calls": (execution.get("llm_usage") or {}).get("calls"),
        "avg_llm_latency_s": execution.get("avg_llm_latency_seconds"),
        "quality_score": (validation.get("quality") or {}).get("score"),
        "final_file_mb": round((execution.get("final_file_size_bytes") or 0) / 1048576, 2),
        "error": result.error,
    }


def _markdown(rows: list[dict[str, Any]], host: dict[str, Any]) -> str:
    cols = ["label", "records", "mode", "status", "tasks", "queues", "seconds", "records_per_second", "cpu_avg_pct",
            "cpu_peak_pct", "engine_ram_peak_mb", "rejected", "llm_calls", "avg_llm_latency_s", "quality_score",
            "final_file_mb"]
    lines = [f"# Benchmark results\n\nHost: {host['platform']}, {host['cores']} logical cores, "
             f"{host['ram_mb']} MB RAM, GPU: {host['gpu'] or 'none'}\n",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the requested benchmark matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10_000, 100_000])
    parser.add_argument("--modes", nargs="+", default=["auto"], choices=["auto", "thread", "process"])
    parser.add_argument("--work-stealing", action="store_true")
    parser.add_argument("--ollama", action="store_true", help="benchmark the real local model on an LLM text field")
    parser.add_argument("--model", default="qwen3:0.6b")
    args = parser.parse_args()

    resources = detect_resources()
    host = {"platform": f"{platform.system()} {platform.release()}", "cores": resources.logical_cores,
            "ram_mb": resources.total_ram_mb, "gpu": ", ".join(g.name for g in resources.gpus)}
    engine = SyntheticDataEngine()
    RESULTS.mkdir(parents=True, exist_ok=True)
    store = RESULTS / "benchmark_results.json"
    history = json.loads(store.read_text(encoding="utf-8")) if store.exists() else []
    rows: list[dict[str, Any]] = []
    for size in args.sizes:
        for mode in args.modes:
            if args.ollama:
                request = {"schema": LLM_SCHEMA, "total_records": size, "batch_size": max(8, size // 8),
                           "model_provider": "ollama", "model_name": args.model, "execution_mode": mode,
                           "quality_requirements": {"max_regeneration_rounds": 6, "min_quality_score": 70}}
                label = f"ollama:{args.model}"
            else:
                request = {**EMPLOYEES, "total_records": size, "execution_mode": mode,
                           "work_stealing": args.work_stealing}
                label = "mock:employees"
            request["schema_name"] = f"benchmark_{label.split(':')[0]}_{size}"
            started = time.perf_counter()
            result = engine.run(request)
            row = _row(result, label, size, mode)
            row["wall_seconds"] = round(time.perf_counter() - started, 2)
            row["host"] = host
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "host"}))
    history.extend(rows)
    store.write_text(json.dumps(history, indent=2), encoding="utf-8")
    (RESULTS / "benchmark_results.md").write_text(_markdown(history, host), encoding="utf-8")
    return 0 if all(r["status"] == "completed" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
