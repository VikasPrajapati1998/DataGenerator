"""End-to-end smoke test of the HTTP API and the MCP server (mock mode, offline, no Ollama needed).

Run from the project root::

    python scripts\\api_smoke_test.py                       # about 20 seconds
    python scripts\\api_smoke_test.py --keep                # keep the generated files (path is printed)
    python scripts\\api_smoke_test.py --job-records 1000000 # bigger CSV job

It starts ``python -m app.api`` on a free port with its own output folder, then checks:

* ``/health``; only the simple endpoints are public, and no login header is asked for
* ``POST /jobs`` with an uploaded schema file: progress polling, ``/files``, download (with HTTP Range),
  ``check_output`` on the CSV, the quality report, preview, delete
* every format and packaging: Excel (also zipped), JSON gzip, JSONL zip
* stop and resume of a running job, ``422`` for a bad schema file, nothing left behind after delete
* a multi-table file: one zip with a CSV per table, composite keys that join, exact counts, table preview
* MCP over stdio (in-process generation), MCP over stdio forwarding to the API (``--api-url``), and MCP over
  streamable HTTP (``--transport http``)

Exit code 0 = all passed, 1 = something failed.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "schemas" / "employees_mock.json"
MULTI_SAMPLE = ROOT / "samples" / "schemas" / "procurement_multi.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_output import check  # noqa: E402

FAILURES: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label} {detail}".strip())
    return ok


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(args: list[str], env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    return subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=log.open("wb"),
                            stderr=subprocess.STDOUT, creationflags=flags)


def wait_http(url: str, proc: subprocess.Popen[bytes], seconds: float = 60) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            httpx.get(url, timeout=2)
            return True
        except httpx.HTTPError:
            time.sleep(0.3)
    return False


def stream_to(response: httpx.Response, target: Path) -> int:
    size = 0
    with target.open("wb") as handle:
        for chunk in response.iter_raw():
            handle.write(chunk)
            size += len(chunk)
    return size


def gunzip(source: Path, target: Path) -> Path:
    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    return target


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def upload(http: httpx.Client, records: int, **form: str) -> httpx.Response:
    files = {"file": ("employees.json", SAMPLE.read_bytes(), "application/json")}
    return http.post("/jobs", files=files, data={"records": str(records), "mode": "mock", **form})


def wait_job(http: httpx.Client, job_id: str, total: int, show: bool = False, seconds: float = 900) -> dict[str, Any]:
    started, status, seen = time.monotonic(), {}, set()
    while time.monotonic() - started < seconds:
        status = http.get(f"/jobs/{job_id}").json()
        progress = status.get("progress", {})
        line = f"{status['status']} {progress.get('records_done', 0):,}/{total:,}"
        if show and line not in seen:
            print(f"        {line} ({progress.get('phase') or ''})")
            seen.add(line)
        if status["status"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    return status


def data_file(http: httpx.Client, job_id: str) -> dict[str, Any]:
    return next(f for f in http.get(f"/jobs/{job_id}/files").json()["files"] if f["kind"] == "data")


def api_checks(base: str, work: Path, out_dir: Path, job_records: int) -> None:
    with httpx.Client(base_url=base, timeout=600) as http:
        health = http.get("/health").json()
        report(health.get("status") == "ok" and "excel" in health.get("formats", []), "GET /health",
               f"formats {health.get('formats')}, modes {health.get('modes')}, compress {health.get('compress')}")
        spec = http.get("/openapi.json").json()
        paths = sorted(spec["paths"])
        headers = [p["name"] for item in spec["paths"].values() for op in item.values()
                   for p in op.get("parameters", []) if p.get("in") == "header"]
        report(not any(p.startswith("/dev") for p in paths) and not headers,
               "only the simple endpoints are public, no login header", f"{len(paths)} paths")

        created = upload(http, job_records)
        if not report(created.status_code == 202, f"POST /jobs (upload schema file, {job_records:,} records, csv)",
                      created.text[:160]):
            return
        job_id = created.json()["job_id"]
        started = time.monotonic()
        status = wait_job(http, job_id, job_records, show=True)
        if not report(status.get("status") == "completed", "job completed",
                      f"{time.monotonic() - started:.1f}s {status.get('error') or ''}"):
            return
        listing = http.get(f"/jobs/{job_id}/files").json()["files"]
        report({f["kind"] for f in listing} == {"data", "report"}, "GET /jobs/{id}/files",
               ", ".join(f"{f['name']} ({f['size_bytes']:,} bytes)" for f in listing))
        data = next(f for f in listing if f["kind"] == "data")
        target = work / "job.csv"
        with http.stream("GET", data["url"]) as response:
            size = stream_to(response, target)
        errors = check(SAMPLE, target)[0]
        report(count_lines(target) == job_records + 1 and not errors, "download the CSV",
               f"{size:,} bytes, {count_lines(target) - 1:,} records{'; ' + '; '.join(errors[:3]) if errors else ''}")
        partial = http.get(data["url"], headers={"Range": "bytes=0-99"})
        report(partial.status_code == 206 and len(partial.content) == 100, "download supports HTTP Range (resume)")
        quality = next(f for f in listing if f["kind"] == "report")
        report(http.get(quality["url"]).status_code == 200, "download the quality report")
        preview = http.get(f"/jobs/{job_id}/preview", params={"rows": 3}).json()
        report(preview.get("returned") == 3, "GET /jobs/{id}/preview", json.dumps(preview["rows"][0])[:100])
        report(job_id in [j["job_id"] for j in http.get("/jobs").json()], "GET /jobs lists the job")
        job_dir = Path(status["job_dir"])
        deleted = http.delete(f"/jobs/{job_id}")
        report(deleted.status_code == 200 and http.get(f"/jobs/{job_id}").status_code == 404
               and not job_dir.exists(), "DELETE /jobs/{id} removes the job and its folder")

        for fmt, packing, name in (("excel", "none", "generated_dataset.xlsx"),
                                   ("excel", "zip", "generated_dataset.xlsx.zip"),
                                   ("json", "gzip", "generated_dataset.json.gz"),
                                   ("jsonl", "zip", "generated_dataset.jsonl.zip")):
            job = upload(http, 2000, format=fmt, compress=packing).json()
            done = wait_job(http, job["job_id"], 2000)
            ok = done.get("status") == "completed"
            got = data_file(http, job["job_id"]) if ok else {}
            body = http.get(got["url"]).content if ok else b""
            rows = rows_in(body, fmt, packing) if ok else -1
            report(ok and got["name"] == name and rows == 2000, f"format {fmt}, compress {packing}",
                   f"{got.get('name')}, {len(body):,} bytes, {rows:,} rows")
            http.delete(f"/jobs/{job['job_id']}")

        big = upload(http, 300_000).json()
        time.sleep(1.0)
        stopped = http.post(f"/jobs/{big['job_id']}/stop")
        status = wait_job(http, big["job_id"], 300_000, seconds=120)
        report(stopped.status_code == 200 and status.get("status") in ("cancelled", "completed"),
               "POST /jobs/{id}/stop", f"status {status.get('status')}")
        if status.get("status") == "cancelled":
            resumed = http.post(f"/jobs/{big['job_id']}/resume")
            time.sleep(1.0)
            http.post(f"/jobs/{big['job_id']}/stop")
            again = wait_job(http, big["job_id"], 300_000, seconds=300)
            report(resumed.status_code == 202 and again.get("status") in ("cancelled", "completed"),
                   "POST /jobs/{id}/resume", f"status {again.get('status')}")
        http.delete(f"/jobs/{big['job_id']}")

        multi_checks(http)

        bad = http.post("/jobs", files={"file": ("bad.json", b'{"schema": {"bad name": "integer"}, '
                                                         b'"total_records": 5}')}, data={"mode": "mock"})
        report(bad.status_code == 422 and "problems" in bad.json(), "bad schema file -> 422 with the problem")
        leftovers = [p for p in out_dir.rglob("*") if p.is_file()] if out_dir.exists() else []
        report(not leftovers, "nothing left in the output folder after deleting the jobs",
               f"{len(leftovers)} files left" + (f", e.g. {leftovers[0]}" if leftovers else ""))


def multi_checks(http: httpx.Client) -> None:
    """A multi-table file: one zip with every table, preview of one table, keys that join."""
    import csv  # pylint: disable=import-outside-toplevel
    import io  # pylint: disable=import-outside-toplevel
    import zipfile  # pylint: disable=import-outside-toplevel

    body = MULTI_SAMPLE.read_bytes()
    created = http.post("/jobs", files={"file": ("procurement.json", body, "application/json")},
                        data={"mode": "mock", "compress": "zip"})
    if not report(created.status_code == 202, "POST /jobs with a multi-table file", created.text[:160]):
        return
    job_id = created.json()["job_id"]
    done = wait_job(http, job_id, created.json()["total_records"])
    if not report(done.get("status") == "completed", "multi-table job completed",
                  f"{done.get('status')} {done.get('error') or ''} tables {[t['name'] for t in done.get('tables', [])]}"):
        return
    data = data_file(http, job_id)
    archive = zipfile.ZipFile(io.BytesIO(http.get(data["url"]).content))
    tables = {Path(n).stem: list(csv.DictReader(io.TextIOWrapper(archive.open(n), encoding="utf-8")))
              for n in archive.namelist()}
    report(sorted(tables) == sorted(data.get("tables", [])), "multi-table zip holds one CSV per table",
           ", ".join(f"{k} {len(v):,}" for k, v in tables.items()))
    lines = {(r["po_no"], r["line_no"]) for r in tables.get("po_lines", [])}
    grn = tables.get("grn", [])
    report(bool(grn) and all((g["po_no"], g["line_no"]) in lines for g in grn),
           "GRN composite keys (po_no, line_no) all exist in po_lines", f"{len(grn)} GRNs")
    report(sum(g["receipt_type"] == "split" for g in grn) == 45, "exact count: 45 split GRNs")
    preview = http.get(f"/jobs/{job_id}/preview", params={"rows": 2, "table": "po_lines"}).json()
    report(preview.get("table") == "po_lines" and len(preview.get("rows", [])) == 2, "preview one table")
    http.delete(f"/jobs/{job_id}")


def rows_in(body: bytes, fmt: str, packing: str) -> int:
    import io  # pylint: disable=import-outside-toplevel
    import zipfile  # pylint: disable=import-outside-toplevel

    if packing == "zip":
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            body = archive.read(archive.namelist()[0])
    elif packing == "gzip":
        body = gzip.decompress(body)
    if fmt == "excel":
        from openpyxl import load_workbook  # pylint: disable=import-outside-toplevel

        return sum(1 for _ in load_workbook(io.BytesIO(body), read_only=True).worksheets[0].iter_rows()) - 1
    if fmt == "json":
        return len(json.loads(body))
    return len(body.decode("utf-8").strip().splitlines())


async def mcp_checks(base: str | None, env: dict[str, str], work: Path) -> None:
    """MCP over stdio (in-process), stdio forwarding to the API at ``base`` (skipped when None), and HTTP."""
    from mcp import Client, StdioServerParameters, stdio_client  # pylint: disable=import-outside-toplevel

    server_log = (work / "mcp_stdio.log").open("w", encoding="utf-8")  # the servers' own log lines

    def payload(result: Any) -> dict[str, Any]:
        return result.structured_content or json.loads(result.content[0].text)

    local = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT), env=env)
    async with Client(stdio_client(local, errlog=server_log)) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        report({"generate_data", "validate_schema", "job_status", "stop_job", "delete_job", "list_job_files",
                "list_sample_schemas"} <= set(names), "MCP stdio: tools listed", ", ".join(names))
        inline = payload(await client.call_tool("generate_data", {"sample": "employees_mock", "records": 5,
                                                                  "mode": "mock"}))
        report(inline.get("mode") == "inline" and len(inline.get("rows", [])) == 5, "MCP stdio: inline rows")
        job = payload(await client.call_tool("generate_data", {"sample": "employees_mock", "records": 3000,
                                                               "mode": "mock", "output_format": "excel",
                                                               "wait_seconds": 120}))
        path = next((f["path"] for f in job.get("files", []) if f["kind"] == "data"), None)
        report(job.get("status") == "completed" and path and Path(path).exists(),
               "MCP stdio: job -> file path (rows not sent to the model)", str(path))
        invalid = payload(await client.call_tool("validate_schema", {"request": {"schema": {"a b": "x"},
                                                                                 "total_records": 3}}))
        report(invalid.get("ok") is False and invalid.get("problems"), "MCP stdio: invalid schema explained")

    if base is not None:
        remote = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server", "--api-url", base],
                                       cwd=str(ROOT), env=env)
        async with Client(stdio_client(remote, errlog=server_log)) as client:
            job = payload(await client.call_tool("generate_data", {"sample": "employees_mock", "records": 2000,
                                                                   "mode": "mock", "compress": "zip",
                                                                   "wait_seconds": 120}))
            url = next((f["url"] for f in job.get("files", []) if f["kind"] == "data"), "")
            ok = job.get("status") == "completed" and url.startswith(base)
            if ok:
                ok = httpx.get(url, timeout=60).status_code == 200
            report(ok, "MCP stdio -> API (--api-url): job + download link", url)

    server_log.close()
    port = free_port()
    log = work / "mcp_http.log"
    proc = start(["-m", "app.mcp_server", "--transport", "http", "--port", str(port)], env, log)
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        wait_http(f"http://127.0.0.1:{port}/", proc, 60)
        async with Client(url) as client:
            samples = payload(await client.call_tool("list_sample_schemas", {}))
            report(len(samples.get("samples", [])) >= 10, "MCP streamable HTTP: list_sample_schemas", url)
    except Exception as exc:  # noqa: BLE001
        report(False, "MCP streamable HTTP", f"{type(exc).__name__}: {exc} (see {log})")
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-records", type=int, default=20000, help="size of the background job (default 20000)")
    parser.add_argument("--keep", action="store_true", help="keep the generated files")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="sde_api_smoke_"))
    out_dir = work / "output"
    env = {**os.environ, "SDE_OUTPUT_DIR": str(out_dir), "SDE_LOG_DIR": str(work / "logs"),
           "PYTHONIOENCODING": "utf-8"}
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"API smoke test: mock mode, server on {base}, files in {work}\n")
    started = time.perf_counter()
    log = work / "api_server.log"
    proc = start(["-m", "app.api", "--port", str(port), "--log-level", "warning"], env, log)
    try:
        if not report(wait_http(f"{base}/health", proc), "API server started", f"log: {log}"):
            print(log.read_text(encoding="utf-8", errors="replace")[-2000:])
        else:
            api_checks(base, work, out_dir, args.job_records)
            print()
            asyncio.run(mcp_checks(base, env, work))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\nFinished in {time.perf_counter() - started:.0f}s")
    if args.keep:
        print(f"Generated files kept in {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print(f"API SMOKE TEST FAILED: {len(FAILURES)} problem(s)")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("API SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
