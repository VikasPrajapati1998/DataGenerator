"""Multi-table schemas: links (single and composite keys), per-parent children, inherited columns, cases,
formulas, exact counts, hidden columns, integrity checks, resume, and the API / MCP delivery of one file per table."""

from __future__ import annotations

import copy
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from app.core.exceptions import MergeError, SchemaError
from app.engine import SyntheticDataEngine
from app.generation.links import ExactAssigner, Permutation, resolve_counts
from app.multi_table import MultiTableRunner, parse_tables
from app.schemas.input_schema import GenerationRequest
from app.storage.formats import iter_records

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "samples" / "schemas" / "procurement_multi.json").read_text(encoding="utf-8"))


def rows(final: str | Path, table: str, fmt: str = "jsonl") -> list[dict[str, Any]]:
    return list(iter_records(Path(final) / f"{table}.{fmt}", fmt))


def run(settings, registry, schema: dict[str, Any], **extra: Any):
    request = {"schema": schema, "model_provider": "mock", "model_name": "mock-llm", "seed": 11,
               "execution_mode": "thread", "output_format": "jsonl", **extra}
    return SyntheticDataEngine(settings=settings, registry=registry).run(request)


# -- building blocks ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 7, 1000, 4097])
def test_permutation_is_a_bijection(n):
    perm = Permutation(n, 42)
    assert sorted(perm(i) for i in range(n)) == list(range(n))


def test_exact_counts_are_exact_and_spread():
    counts = resolve_counts("x", ["a", "b", "c"], [10, None, 5], None, 100)
    assert counts == [10, 85, 5]
    assigner = ExactAssigner(["a", "b", "c"], counts, 9)
    values = [assigner.value(i) for i in range(100)]
    assert Counter(values) == {"a": 10, "b": 85, "c": 5}
    assert values[:10].count("a") < 10  # spread over the dataset, not one block
    assert resolve_counts("x", ["a", "b"], None, [1, 2], 10) == [3, 7]
    with pytest.raises(SchemaError, match="add up"):
        resolve_counts("x", ["a", "b"], [3, 4], None, 10)


# -- schema validation -------------------------------------------------------------------------------------------
def _tables(**tables: Any) -> dict[str, Any]:
    return {"tables": tables}


PARENT = {"records": 10, "schema": {"id": {"type": "integer", "generator": "sequence", "unique": True}}}


@pytest.mark.parametrize("schema,needle", [
    (_tables(a={"records": 5, "links": {"p": {"table": "nope"}}, "schema": {"x": "integer between 1 and 2"}}),
     "must name an existing table"),
    (_tables(a={"records": 5, "links": {"b": "b"}, "schema": {"x": {"link": "b", "column": "y"}}},
             b={"records": 5, "links": {"a": "a"}, "schema": {"y": {"link": "a", "column": "x"}}}), "cycle"),
    (_tables(p=PARENT, c={"records": 5, "links": {"p": "p"}, "schema": {"x": {"link": "p", "column": "zzz"}}}),
     "has no column 'zzz'"),
    (_tables(p=PARENT, c={"records": 5, "schema": {"n": {"type": "integer", "generator": "child_index"}}}),
     "child_index needs"),
    (_tables(p=PARENT, c={"records": 5, "links": {"p": "p"}, "schema": {"x": {"link": "q", "column": "id"}}}),
     "unknown link"),
    (_tables(p=PARENT, c={"schema": {"x": "integer between 1 and 2"}}), "either \"records\""),
    (_tables(p=PARENT, c={"records": 5, "links": {"p": {"table": "p", "mode": "cover", "match": {"x": "id"}}},
                          "schema": {"x": "integer between 1 and 2"}}), "match only works"),
    (_tables(p={"records": 5, "schema": {"s": {"allowed_values": ["a", "b"], "weights": [1, 1],
                                                "cases_by": "s", "cases": {"a": None}}}}), "cases_by"),
    (_tables(p={"records": 5, "schema": {"s": ["a", "b"], "n": {"type": "integer", "min": 0, "max": 5,
                                                                "cases_by": "s", "cases": {"a": {"max": 50}}}}}),
     "within the field's max"),
])
def test_invalid_multi_table_schemas_are_explained(schema, needle):
    with pytest.raises(SchemaError, match=needle):
        parse_tables(schema)


def test_tables_are_ordered_parents_first_and_types_are_inherited():
    specs = parse_tables(SAMPLE["schema"])
    order = [s.name for s in specs]
    assert order.index("suppliers") < order.index("purchase_orders") < order.index("po_lines") < order.index("grn")
    lines = next(s for s in specs if s.name == "po_lines")
    assert lines.schema.fields["unit_price"].type == "float" and lines.schema.fields["po_no"].type == "string"
    assert lines.hidden == ["po_status"] and lines.estimate == 1000  # 400 orders x 2.5 lines
    orders = next(s for s in specs if s.name == "purchase_orders")
    assert {"po_no", "status"} <= orders.needed
    assert GenerationRequest(**SAMPLE).total_records == sum(s.estimate for s in specs)


# -- end to end -------------------------------------------------------------------------------------------------
def test_procurement_sample_is_consistent(settings, registry):
    result = run(settings, registry, copy.deepcopy(SAMPLE["schema"]))
    assert result.ok, result.error
    final = result.final_output
    suppliers, parts = rows(final, "suppliers"), rows(final, "parts")
    orders, lines, grn = rows(final, "purchase_orders"), rows(final, "po_lines"), rows(final, "grn")

    # exact counts and exact proportions
    assert Counter(s["tier"] for s in suppliers) == {"strategic": 5, "preferred": 15, "approved": 30}
    assert Counter(o["status"] for o in orders) == {"open": 200, "closed": 160, "cancelled": 40}
    assert Counter(o["po_type"] for o in orders)["legacy"] == 60
    assert Counter(g["receipt_type"] for g in grn) == {"split": 45, "full": 255}

    # foreign keys and inherited columns agree with the parent row
    by_supplier = {s["supplier_id"]: s for s in suppliers}
    assert all(by_supplier[o["supplier_id"]]["tier"] == o["supplier_tier"] for o in orders)
    # status drives the dependent column
    assert all((o["closed_date"] is None) == (o["status"] != "closed") for o in orders)

    # per-parent lines: only for open/closed orders, numbered 1..k, composite key unique, hidden column dropped
    by_po = {o["po_no"]: o for o in orders}
    per_po: dict[str, list[int]] = {}
    for line in lines:
        assert by_po[line["po_no"]]["status"] in ("open", "closed") and "po_status" not in line
        per_po.setdefault(line["po_no"], []).append(line["line_no"])
    assert all(sorted(nums) == list(range(1, len(nums) + 1)) and len(nums) <= 4 for nums in per_po.values())
    assert set(per_po) == {o["po_no"] for o in orders if o["status"] != "cancelled"}
    prices = {p["part_no"]: p["unit_price"] for p in parts}
    assert all(abs(line["line_amount"] - line["ordered_qty"] * prices[line["part_no"]]) < 0.011 for line in lines)

    # GRNs: composite key (po_no, line_no) of a closed order's line, each line at most once, quantities agree
    by_line = {(line["po_no"], line["line_no"]): line for line in lines}
    assert len({(g["po_no"], g["line_no"]) for g in grn}) == len(grn)
    for g in grn:
        line = by_line[(g["po_no"], g["line_no"])]
        assert by_po[g["po_no"]]["status"] == "closed" and g["ordered_qty"] == line["ordered_qty"]
        assert g["received_qty"] == (g["ordered_qty"] if g["receipt_type"] == "full" else int(g["ordered_qty"] * 0.6))

    tables = result.summary["tables"]
    assert tables["grn"]["links"]["line"]["broken"] == 0 and tables["po_lines"]["primary_key"]["duplicates"] == 0
    assert result.summary["execution"]["valid_records_generated"] == sum(t["records"] for t in tables.values())
    assert (Path(result.job_dir) / "metadata" / "multi_table_summary.json").exists()


def test_the_same_seed_gives_the_same_tables(settings, registry):
    first = run(settings, registry, copy.deepcopy(SAMPLE["schema"]))
    second = run(settings, registry, copy.deepcopy(SAMPLE["schema"]))
    assert first.ok and second.ok
    for table in ("purchase_orders", "po_lines", "grn"):
        assert rows(first.final_output, table) == rows(second.final_output, table)


MATCHED = {"tables": {
    "pos": {"records": 900, "schema": {
        "po_no": {"type": "integer", "generator": "sequence", "unique": True},
        "po_type": {"type": "string", "allowed_values": ["legacy", "standard"], "weights": [30, 70]}}},
    "asn": {"records": 3000,
            "links": {"po": {"table": "pos", "match": {"po_type": "po_type"}}, "any": {"table": "pos", "mode": "cover"}},
            "schema": {"asn_no": {"type": "integer", "generator": "sequence", "unique": True},
                       "po_type": {"type": "string", "allowed_values": ["legacy", "standard"], "counts": [1234, None]},
                       "po_no": {"link": "po", "column": "po_no"},
                       "covered_po": {"link": "any", "column": "po_no"},
                       "note": {"type": "string", "generator": "llm", "instructions": "a short shipping note"}}}}}


@pytest.mark.parametrize("mode,split", [("thread", 0), ("process", 1000)])
def test_match_cover_and_exact_counts_survive_processes_and_split_runs(settings, registry, mode, split):
    result = run(settings, registry, copy.deepcopy(MATCHED), execution_mode=mode, split_size=split)
    assert result.ok, result.error
    pos, asn = rows(result.final_output, "pos"), rows(result.final_output, "asn")
    types = {p["po_no"]: p["po_type"] for p in pos}
    assert Counter(a["po_type"] for a in asn)["legacy"] == 1234  # exact across sub-jobs
    assert all(types[a["po_no"]] == a["po_type"] for a in asn)  # matched parent
    assert {a["covered_po"] for a in asn} == set(types)  # cover: every order gets at least one ASN
    assert all(a["note"] for a in asn)


def test_cases_formula_and_counts_work_in_a_single_table(settings, registry):
    request = {"schema": {
        "id": {"type": "integer", "generator": "sequence", "unique": True},
        "status": {"type": "string", "allowed_values": ["late", "on_time"], "counts": [30, None]},
        "delay_days": {"type": "integer", "min": 0, "max": 40, "cases_by": "status",
                       "cases": {"late": {"min": 3, "max": 40}, "on_time": {"value": 0}}},
        "penalty": {"type": "float", "formula": "delay_days * 12.5"},
    }, "total_records": 200, "model_provider": "mock", "model_name": "mock-llm", "seed": 4,
        "execution_mode": "thread", "output_format": "jsonl"}
    result = SyntheticDataEngine(settings=settings, registry=registry).run(request)
    assert result.ok, result.error
    records = list(iter_records(Path(result.final_output), "jsonl"))
    assert Counter(r["status"] for r in records) == {"late": 30, "on_time": 170}
    assert all((r["delay_days"] >= 3) if r["status"] == "late" else r["delay_days"] == 0 for r in records)
    assert all(r["penalty"] == r["delay_days"] * 12.5 for r in records)


def test_unique_link_needs_enough_parents(settings, registry):
    schema = {"tables": {"p": PARENT, "c": {"records": 20, "links": {"p": {"table": "p", "mode": "unique"}},
                                            "schema": {"pid": {"link": "p", "column": "id"}}}}}
    result = run(settings, registry, schema)
    assert result.status == "failed" and "only 10 rows" in result.error


def test_a_broken_link_is_refused(settings, registry, monkeypatch):
    original = MultiTableRunner._export

    def corrupt(self, run_dir, manifest, spec, source, specs):  # noqa: ANN001, ANN202
        if spec.name == "c":
            lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            lines[0]["pid"] = 999
            source.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
        return original(self, run_dir, manifest, spec, source, specs)

    monkeypatch.setattr(MultiTableRunner, "_export", corrupt)
    schema = {"tables": {"p": PARENT, "c": {"records": 5, "links": {"p": "p"},
                                            "schema": {"pid": {"link": "p", "column": "id"}}}}}
    result = run(settings, registry, schema)
    assert result.status == "failed" and "point to no matching row" in result.error


def test_resume_keeps_finished_tables(settings, registry, monkeypatch):
    original = MultiTableRunner._export
    calls: list[str] = []

    def fail_on_grn(self, run_dir, manifest, spec, source, specs):  # noqa: ANN001, ANN202
        calls.append(spec.name)
        if spec.name == "grn" and calls.count("grn") == 1:
            raise MergeError("simulated crash")
        return original(self, run_dir, manifest, spec, source, specs)

    monkeypatch.setattr(MultiTableRunner, "_export", fail_on_grn)
    engine = SyntheticDataEngine(settings=settings, registry=registry)
    first = engine.run({**copy.deepcopy(SAMPLE), "model_provider": "mock", "model_name": "mock-llm", "seed": 5,
                        "execution_mode": "thread", "output_format": "csv"})
    assert first.status == "failed" and "simulated crash" in first.error
    lines_file = Path(first.job_dir) / "final" / "po_lines.csv"
    stamp = lines_file.stat().st_mtime_ns
    second = engine.resume(first.job_id)
    assert second.ok, second.error
    assert lines_file.stat().st_mtime_ns == stamp  # not regenerated
    assert calls.count("po_lines") == 1 and calls.count("grn") == 2
    assert len(rows(second.final_output, "grn", "csv")) == 300


# -- API and MCP ------------------------------------------------------------------------------------------------
def test_api_delivers_one_file_per_table(tmp_path, settings, registry):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # pylint: disable=import-outside-toplevel

    from app.api.server import create_app  # pylint: disable=import-outside-toplevel
    from app.service.config import ServiceSettings  # pylint: disable=import-outside-toplevel

    body = json.dumps({**SAMPLE, "seed": 3, "execution_mode": "thread"}).encode()
    with TestClient(create_app(ServiceSettings(state_dir=tmp_path / "jobs"), settings, registry)) as client:
        def start(**form: str) -> str:
            response = client.post("/jobs", files={"file": ("procurement.json", body)}, data={"mode": "mock", **form})
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
            deadline = time.monotonic() + 120
            while client.get(f"/jobs/{job_id}").json()["status"] in ("queued", "running"):
                assert time.monotonic() < deadline
                time.sleep(0.2)
            status = client.get(f"/jobs/{job_id}").json()
            assert status["status"] == "completed", status
            assert [t["name"] for t in status["tables"]][-1] == "grn"
            return job_id

        job = start(format="csv", compress="zip")
        files = client.get(f"/jobs/{job}/files").json()["files"]
        data = next(f for f in files if f["kind"] == "data")
        assert data["name"] == "procurement_tables.zip" and "grn" in data["tables"]
        archive = zipfile.ZipFile(__import__("io").BytesIO(client.get(data["url"]).content))
        assert sorted(archive.namelist()) == sorted(f"{t}.csv" for t in data["tables"])
        preview = client.get(f"/jobs/{job}/preview", params={"rows": 3, "table": "grn"}).json()
        assert preview["table"] == "grn" and set(preview["rows"][0]) >= {"grn_no", "po_no", "line_no"}
        assert client.get(f"/jobs/{job}/preview", params={"table": "nope"}).status_code == 404

        job = start(format="excel")
        names = {f.get("table"): f["name"] for f in client.get(f"/jobs/{job}/files").json()["files"]}
        assert names["po_lines"] == "po_lines.xlsx"
        assert client.get(f"/jobs/{job}/preview", params={"table": "po_lines"}).json()["rows"]

        bad = client.post("/jobs", files={"file": ("p.json", body)}, data={"mode": "mock", "records": "5"})
        assert bad.status_code == 422 and "per table" in bad.json()["error"]


def test_mcp_runs_multi_table_files_as_jobs(tmp_path, settings, registry):
    pytest.importorskip("mcp")
    import asyncio  # pylint: disable=import-outside-toplevel

    from mcp import Client  # pylint: disable=import-outside-toplevel

    from app.mcp_server.backend import LocalBackend  # pylint: disable=import-outside-toplevel
    from app.mcp_server.server import build_server  # pylint: disable=import-outside-toplevel
    from app.service.config import ServiceSettings  # pylint: disable=import-outside-toplevel
    from app.service.jobs import JobManager  # pylint: disable=import-outside-toplevel

    limits = ServiceSettings(state_dir=tmp_path / "jobs")
    manager = JobManager(limits, settings, registry)
    server = build_server(LocalBackend(limits, manager), limits)

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(name, args)
            return result.structured_content or json.loads(result.content[0].text)

    try:
        report = asyncio.run(call("validate_schema", {"sample": "procurement_multi"}))
        assert report["ok"] and report["multi_table"] and report["order"][-1] == "grn" and report["mode"] == "job"
        job = asyncio.run(call("generate_data", {"sample": "procurement_multi", "mode": "mock", "wait_seconds": 90}))
        assert job["status"] == "completed", job
        assert {f.get("table") for f in job["files"] if f["kind"] == "data"} >= {"suppliers", "grn"}
        preview = asyncio.run(call("preview_job_output", {"job_id": job["job_id"], "table": "po_lines", "rows": 2}))
        assert preview["returned"] == 2 and "line_no" in preview["rows"][0]
    finally:
        manager.shutdown()
