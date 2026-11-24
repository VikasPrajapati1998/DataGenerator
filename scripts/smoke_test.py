"""End-to-end smoke test of the generator in mock mode (offline, no Ollama needed).

Run from the project root::

    python scripts/smoke_test.py                 # quick: every sample schema, 1,000 records each
    python scripts/smoke_test.py --records 10000 # bigger runs
    python scripts/smoke_test.py --keep          # keep the generated files (path is printed)

For every sample schema it runs ``python -m app.main generate`` with the mock model and checks the
output independently (``scripts/check_output.py``: duplicates, nulls, ranges, patterns, date order,
references). Then it checks reproducibility: the same seed must give the same file for different
batch sizes, and for per-record LLM text; and two different schemas generated with the same seed
must not share values row by row. Exit code 0 = all passed, 1 = something failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "samples" / "schemas"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_output import check  # noqa: E402

# Request files that pin real (Ollama) models in the file itself; the mock override cannot apply.
_SKIP = {"customer_profiles_pool"}


def generate(config: Path, out: Path, records: int, *extra: str) -> tuple[Path | None, str]:
    """Run the CLI once; return the final dataset path (or None) and an error summary."""
    command = [sys.executable, "-m", "app.main", "generate", "--config", str(config), "--records", str(records),
               "--provider", "mock", "--model", "mock-llm", "--seed", "42", "--output-dir", str(out), *extra]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=False)
    try:
        result = json.loads(proc.stdout[proc.stdout.index("{"):])
    except ValueError:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return None, f"exit {proc.returncode}: {tail[0][:300]}"
    if result.get("status") != "completed" or not result.get("final_output"):
        return None, f"exit {proc.returncode}: {result.get('status')}: {str(result.get('error'))[:300]}"
    return Path(result["final_output"]), ""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=int, default=1000, help="records per schema (default 1000)")
    parser.add_argument("--keep", action="store_true", help="keep the generated files")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="sde_smoke_"))
    failures: list[str] = []
    outputs: dict[str, Path] = {}
    started = time.perf_counter()
    print(f"Smoke test: mock mode, {args.records:,} records per schema, output in {work}\n")

    for config in sorted(SCHEMAS.glob("*.json")):
        if config.stem in _SKIP:
            print(f"[skip] {config.stem} (pins Ollama models in the request file)")
            continue
        t0 = time.perf_counter()
        dataset, error = generate(config, work / "runs", args.records)
        if dataset is None:
            failures.append(f"{config.stem}: generation failed ({error})")
            print(f"[FAIL] {config.stem}: generation failed ({error})")
            continue
        outputs[config.stem] = dataset
        errors, _notes = check(config, dataset)
        status = "ok  " if not errors else "FAIL"
        print(f"[{status}] {config.stem}: {args.records:,} records in {time.perf_counter() - t0:.1f}s")
        for line in errors:
            print(f"        ERROR {line}")
            failures.append(f"{config.stem}: {line}")

    print("\nReproducibility")
    config = SCHEMAS / "employees_mock.json"
    first, e1 = generate(config, work / "det_a", 3000, "--batch-size", "300")
    second, e2 = generate(config, work / "det_b", 3000, "--batch-size", "1000")
    same = first is not None and second is not None and digest(first) == digest(second)
    print(f"[{'ok  ' if same else 'FAIL'}] same seed, batch size 300 vs 1000 -> identical file")
    if not same:
        failures.append(f"employees_mock: same seed gave different files for different batch sizes {e1}{e2}")

    config = SCHEMAS / "customer_profiles.json"
    runs = [generate(config, work / f"llm_{i}", 300, "--llm-mode", "per_record", "--format", "csv") for i in (1, 2)]
    same = all(p for p, _ in runs) and digest(runs[0][0]) == digest(runs[1][0])  # type: ignore[arg-type]
    print(f"[{'ok  ' if same else 'FAIL'}] per-record mock LLM text, two runs -> identical file")
    if not same:
        failures.append("customer_profiles: two identical mock runs gave different files")

    patients, appointments = outputs.get("healthcare_patients"), outputs.get("healthcare_appointments")
    if patients and appointments:
        import csv  # pylint: disable=import-outside-toplevel

        with patients.open(encoding="utf-8") as a, appointments.open(encoding="utf-8") as b:
            pairs = list(zip(csv.DictReader(a), csv.DictReader(b)))
        shared = sum(p["mrn"] == q["patient_mrn"] for p, q in pairs)
        ok = shared <= len(pairs) * 0.01
        print(f"[{'ok  ' if ok else 'FAIL'}] patients vs appointments with the same seed: "
              f"{shared} of {len(pairs)} rows share an MRN by accident")
        if not ok:
            failures.append("patients/appointments: same seed produced row-by-row correlated tables")

    print(f"\nFinished in {time.perf_counter() - started:.0f}s")
    if args.keep:
        print(f"Generated files kept in {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} problem(s)")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
