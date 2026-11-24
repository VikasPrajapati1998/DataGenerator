"""Generate the sample output datasets shipped in ``samples/outputs``.

Usage (from the project root)::

    python scripts/make_samples.py            # offline mock model only
    python scripts/make_samples.py --ollama   # also run the Ollama-backed samples

Each run copies the final dataset plus its execution/validation summaries
into ``samples/outputs/<name>/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine import SyntheticDataEngine  # noqa: E402

SAMPLES = [
    ("employees_mock_1k", "employees_mock.json", {"total_records": 1000, "batch_size": 250}, False),
    ("employees_ollama_200", "employees.json", {"total_records": 200, "batch_size": 50}, True),
    ("ecommerce_orders_ollama_120", "ecommerce_orders.json", {"total_records": 120, "batch_size": 40}, True),
    ("customer_profiles_ollama_60", "customer_profiles.json", {"total_records": 60, "batch_size": 20}, True),
]


def main() -> int:
    """Run the selected samples and copy their outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true", help="also run samples that need a local Ollama model")
    args = parser.parse_args()
    engine = SyntheticDataEngine()
    failures = 0
    for name, schema_file, overrides, needs_ollama in SAMPLES:
        if needs_ollama and not args.ollama:
            continue
        request = json.loads((ROOT / "samples" / "schemas" / schema_file).read_text(encoding="utf-8"))
        request.update(overrides)
        request.setdefault("schema_name", Path(schema_file).stem)
        result = engine.run(request)
        print(f"{name}: {result.status} {result.error or ''}")
        if not result.ok:
            failures += 1
            continue
        target = ROOT / "samples" / "outputs" / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.final_output, target / Path(result.final_output).name)
        for summary in ("execution_summary.json", "validation_summary.json", "job_config.json"):
            shutil.copy2(Path(result.job_dir) / "metadata" / summary, target / summary)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
