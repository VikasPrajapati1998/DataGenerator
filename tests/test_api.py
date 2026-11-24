"""HTTP API: upload a schema file, jobs (status, stop, resume, delete), files, formats, packaging, limits."""

from __future__ import annotations

import gzip
import io
import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api.server import create_app  # noqa: E402
from app.engine import JobResult  # noqa: E402
from app.service.config import ServiceSettings  # noqa: E402
from app.service.jobs import JobManager, gzip_file, iter_gzip, preview_rows, to_excel, zip_file  # noqa: E402
from app.service.requests import prepare_job  # noqa: E402

from .conftest import EMPLOYEE_SCHEMA  # noqa: E402

SCHEMA_FILE = json.dumps({"schema": EMPLOYEE_SCHEMA, "total_records": 300, "seed": 7,
                          "execution_mode": "thread"}).encode()


@pytest.fixture
def limits(tmp_path: Path) -> ServiceSettings:
    return ServiceSettings(state_dir=tmp_path / "api_jobs", chunk_size=4096)


@pytest.fixture
def client(limits, settings, registry):
    with TestClient(create_app(limits, settings, registry)) as test_client:
        yield test_client


def start(client: TestClient, body: bytes = SCHEMA_FILE, name: str = "people.json", **form: Any) -> dict[str, Any]:
    form.setdefault("mode", "mock")
    response = client.post("/jobs", files={"file": (name, body, "application/json")},
                           data={k: str(v) for k, v in form.items()})
    assert response.status_code == 202, response.text
    return response.json()


def wait_done(client: TestClient, job_id: str, seconds: float = 60) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] not in ("queued", "running"):
            return status
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish")


def download(client: TestClient, job_id: str) -> tuple[dict[str, Any], bytes]:
    files = client.get(f"/jobs/{job_id}/files").json()["files"]
    data = next(f for f in files if f["kind"] == "data")
    response = client.get(data["url"])
    assert response.status_code == 200
    return data, response.content


def test_health_lists_the_choices(client):
    health = client.get("/health").json()
    assert health["formats"] == ["csv", "json", "jsonl", "excel"] and health["modes"] == ["mock", "llm"]
    assert health["compress"] == ["none", "zip", "gzip"]


def test_only_the_simple_endpoints_are_public(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/health", "/jobs", "/jobs/{job_id}", "/jobs/{job_id}/stop", "/jobs/{job_id}/resume",
                     "/jobs/{job_id}/files", "/jobs/{job_id}/files/{name}", "/jobs/{job_id}/preview"}
    assert client.get("/dev/samples").status_code == 404


def test_no_login_header_is_asked_for(client):
    spec = client.get("/openapi.json").json()
    params = [p["name"] for item in spec["paths"].values() for op in item.values() for p in op.get("parameters", [])]
    assert not any(name.lower() == "x-api-key" for name in params)
    assert client.get("/jobs").status_code == 200


def test_csv_job_files_download_preview_delete(client):
    job = start(client, records=1000)
    assert job["status"] == "queued" and job["links"]["files"].endswith(f"/jobs/{job['job_id']}/files")
    assert job["job_id"].startswith("people_")  # named after the uploaded file
    status = wait_done(client, job["job_id"])
    assert status["status"] == "completed", status
    assert status["progress"]["percent"] == 100.0 and status["records"] == 1000

    listing = client.get(f"/jobs/{job['job_id']}/files").json()
    kinds = {f["kind"]: f for f in listing["files"]}
    assert set(kinds) == {"data", "report"} and kinds["data"]["name"] == "generated_dataset.csv"
    assert "path" not in kinds["data"]
    data, body = download(client, job["job_id"])
    lines = body.decode().strip().splitlines()
    assert lines[0].split(",") == list(EMPLOYEE_SCHEMA) and len(lines) == 1001
    partial = client.get(data["url"], headers={"Range": "bytes=0-9"})
    assert partial.status_code == 206 and len(partial.content) == 10
    report = client.get(kinds["report"]["url"]).json()
    assert report  # the quality report is JSON

    preview = client.get(f"/jobs/{job['job_id']}/preview", params={"rows": 3}).json()
    assert preview["returned"] == 3 and preview["rows"][0]["employee_id"] == "1"
    assert job["job_id"] in [j["job_id"] for j in client.get("/jobs").json()]
    assert client.delete(f"/jobs/{job['job_id']}").status_code == 200
    assert client.get(f"/jobs/{job['job_id']}").status_code == 404


@pytest.mark.parametrize("fmt,compress,name", [
    ("excel", "none", "generated_dataset.xlsx"),
    ("excel", "zip", "generated_dataset.xlsx.zip"),
    ("json", "gzip", "generated_dataset.json.gz"),
    ("jsonl", "zip", "generated_dataset.jsonl.zip"),
])
def test_formats_and_packaging(client, fmt, compress, name):
    job = start(client, records=200, format=fmt, compress=compress)
    assert wait_done(client, job["job_id"])["status"] == "completed"
    data, body = download(client, job["job_id"])
    assert data["name"] == name
    if compress == "zip":
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            assert archive.namelist() == [name[:-4]]
            body = archive.read(name[:-4])
    elif compress == "gzip":
        body = gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
    if fmt == "excel":
        from openpyxl import load_workbook  # pylint: disable=import-outside-toplevel

        rows = list(load_workbook(io.BytesIO(body), read_only=True).worksheets[0].iter_rows(values_only=True))
        assert list(rows[0]) == list(EMPLOYEE_SCHEMA) and len(rows) == 201
        assert isinstance(rows[1][0], int)  # numbers stay numbers in Excel
    elif fmt == "json":
        assert len(json.loads(body)) == 200
    else:
        assert len(body.decode().strip().splitlines()) == 200
    preview = client.get(f"/jobs/{job['job_id']}/preview", params={"rows": 2}).json()
    assert preview["returned"] == 2 and set(preview["rows"][0]) == set(EMPLOYEE_SCHEMA)


def test_only_the_final_file_is_kept(client):
    job = start(client, records=100, format="excel", compress="zip")
    status = wait_done(client, job["job_id"])
    final_dir = Path(status["download_file"]).parent
    assert sorted(p.name for p in final_dir.iterdir()) == ["generated_dataset.xlsx.zip"]


@pytest.mark.parametrize("body,form,needle", [
    (json.dumps({"schema": {"bad name": "integer"}, "total_records": 5}).encode(), {}, "Invalid field name"),
    (json.dumps({"schema": EMPLOYEE_SCHEMA}).encode(), {}, "number of records is missing"),
    (b"{not json", {}, "invalid JSON"),
    (b"   ", {}, "empty"),
])
def test_bad_files_are_explained(client, body, form, needle):
    response = client.post("/jobs", files={"file": ("x.json", body)}, data={"mode": "mock", **form})
    assert response.status_code == 422
    assert needle in response.json()["error"] and response.json()["problems"]


def test_unknown_choices_are_rejected(client):
    for form in ({"mode": "gpt"}, {"format": "xml"}, {"compress": "rar"}, {"records": "0"}):
        response = client.post("/jobs", files={"file": ("x.json", SCHEMA_FILE)}, data=form)
        assert response.status_code == 422, form


def test_mode_decides_the_model_not_the_file():
    file = {"schema": EMPLOYEE_SCHEMA, "total_records": 5, "model_provider": "ollama_local",
            "model_name": "big-model", "model_pool": ["x/y"], "output_dir": "C:/elsewhere", "job_id": "x"}
    mock, fmt = prepare_job(file, mode="mock")
    assert (mock["model_provider"], mock["model_name"], fmt) == ("mock", "mock-llm", "csv")
    assert "model_pool" not in mock and "output_dir" not in mock and "job_id" not in mock
    llm, _ = prepare_job(file, mode="llm", output_format="excel")
    assert "model_provider" not in llm and "model_name" not in llm  # the .env model pool is used
    assert llm["output_format"] == "jsonl"  # Excel is converted from the engine's JSONL file


def test_failed_job_has_error_hint_and_can_resume(client):
    class FailingEngine:
        def run(self, request):  # noqa: ANN001, ANN201, ARG002
            return JobResult("failed", None, None, None, error="ModelUnavailableError: no model reachable",
                             error_type="ModelUnavailableError")

        resume = run

        def cancel(self) -> None:
            pass

    client.app.state.jobs._engine = FailingEngine
    job = start(client, mode="llm")
    status = wait_done(client, job["job_id"])
    assert status["status"] == "failed" and "no model" in status["error"] and "ollama" in status["hint"].lower()
    assert client.get(f"/jobs/{job['job_id']}/files").status_code == 409
    assert client.post(f"/jobs/{job['job_id']}/resume").status_code == 202
    assert wait_done(client, job["job_id"])["status"] == "failed"


def test_jobs_survive_restart_as_interrupted(limits, settings, registry):
    limits.state_dir.mkdir(parents=True)
    record = {"job_id": "dataset_20261122_000000_abcdef", "status": "running", "schema_name": "dataset",
              "total_records": 10, "output_format": "csv", "compress": False, "created_at": "2026-11-22T00:00:00",
              "unknown_old_field": 1, "request": {}}
    (limits.state_dir / f"{record['job_id']}.json").write_text(json.dumps(record), encoding="utf-8")
    manager = JobManager(limits, settings, registry)
    try:
        loaded = manager.get(record["job_id"])
        assert loaded.status == "interrupted" and loaded.compress == "none"
    finally:
        manager.shutdown()


class _BlockingEngine:
    """Stands in for the engine: runs until released or cancelled."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.cancelled = False

    def run(self, request: dict[str, Any]) -> JobResult:  # noqa: ARG002
        while not self.release.wait(0.05):
            if self.cancelled:
                return JobResult("cancelled", None, None, None, error="cancelled", error_type="CancelledError")
        return JobResult("failed", None, None, None, error="released", error_type="TaskExecutionError")

    def cancel(self) -> None:
        self.cancelled = True


def test_stop_running_and_queued_jobs(limits, settings, registry):
    manager = JobManager(limits, settings, registry)
    engines: list[_BlockingEngine] = []

    def make() -> _BlockingEngine:
        engines.append(_BlockingEngine())
        return engines[-1]

    manager._engine = make  # type: ignore[method-assign]
    try:
        first = manager.submit(_model())
        second = manager.submit(_model())
        deadline = time.monotonic() + 10
        while manager.get(first.job_id).status != "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert manager.get(second.job_id).status == "queued"  # one job at a time by default
        assert manager.status(first.job_id)["progress"]["phase"]
        assert manager.cancel(second.job_id).status == "cancelled"
        manager.cancel(first.job_id)
        assert manager.wait(first.job_id, 10).status == "cancelled"
        assert manager.get(first.job_id).hint
    finally:
        for engine in engines:
            engine.release.set()
        manager.shutdown()


def test_stop_endpoint(client):
    job = start(client, records=100)
    stopped = client.post(f"/jobs/{job['job_id']}/stop")
    assert stopped.status_code == 200 and stopped.json()["job_id"] == job["job_id"]
    status = wait_done(client, job["job_id"])
    assert status["status"] in ("cancelled", "completed")  # a 100-record job may finish before the stop lands


def _model():
    from app.schemas.input_schema import GenerationRequest  # pylint: disable=import-outside-toplevel

    request, _ = prepare_job(SCHEMA_FILE, mode="mock")
    return GenerationRequest(**request)


@pytest.mark.parametrize("fmt", ["csv", "jsonl", "json"])
def test_preview_reads_only_the_start(tmp_path, fmt):
    rows = [{"id": i, "name": f"n{i}"} for i in range(1000)]
    path = tmp_path / f"data.{fmt}"
    if fmt == "csv":
        path.write_text("id,name\n" + "".join(f"{r['id']},{r['name']}\n" for r in rows), encoding="utf-8")
    elif fmt == "jsonl":
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    else:
        path.write_text("[\n" + ",\n".join(json.dumps(r) for r in rows) + "\n]\n", encoding="utf-8")
    for source in (path, gzip_file(path, 4096), zip_file(path)):
        got = preview_rows(source, 5)
        assert [str(r["id"]) for r in got] == ["0", "1", "2", "3", "4"]
    assert len(preview_rows(path, 5000)) == 1000


def test_excel_continues_on_new_sheets(tmp_path, monkeypatch):
    from openpyxl import load_workbook  # pylint: disable=import-outside-toplevel

    import app.service.jobs as jobs_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(jobs_module, "EXCEL_MAX_ROWS", 4)
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps({"id": i, "tags": ["a"]}) + "\n" for i in range(10)), encoding="utf-8")
    book = load_workbook(to_excel(path, "jsonl"), read_only=True)
    assert book.sheetnames == ["data", "data_2", "data_3"]
    assert [r[0] for r in book["data_3"].iter_rows(values_only=True)] == ["id", 8, 9]
    assert list(book["data"].iter_rows(values_only=True))[1] == (0, '["a"]')


def test_iter_gzip_round_trip(tmp_path):
    path = tmp_path / "big.txt"
    path.write_bytes(b"x,y\n" * 50_000)
    assert gzip.GzipFile(fileobj=io.BytesIO(b"".join(iter_gzip(path, 4096)))).read() == path.read_bytes()


def test_registry_close_releases_every_thread_connection(tmp_path):
    """Open SQLite handles lock the job folder on Windows, so job output could not be deleted."""
    import sqlite3  # pylint: disable=import-outside-toplevel

    from app.guardrails.uniqueness import UniquenessRegistry  # pylint: disable=import-outside-toplevel

    registry = UniquenessRegistry(tmp_path / "uniqueness.sqlite")
    seen: list[sqlite3.Connection] = []
    worker = threading.Thread(target=lambda: seen.append(registry._connect()))
    worker.start()
    worker.join()
    seen.append(registry._connect())
    registry.close()
    for conn in seen:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    registry._connect().execute("SELECT 1")  # still usable afterwards: reopens on demand
    registry.close()
