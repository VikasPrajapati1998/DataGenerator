"""MCP server tools (in-process client), the API-forwarding backend, and the example LLM client loop."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from app.mcp_server.backend import HttpBackend, LocalBackend  # noqa: E402
from app.mcp_server.server import build_server  # noqa: E402
from app.service.config import ServiceSettings  # noqa: E402
from app.service.jobs import JobManager  # noqa: E402

from .conftest import EMPLOYEE_SCHEMA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REQUEST = {"schema": EMPLOYEE_SCHEMA, "total_records": 10, "seed": 7, "execution_mode": "thread"}


@pytest.fixture
def limits(tmp_path: Path) -> ServiceSettings:
    return ServiceSettings(state_dir=tmp_path / "api_jobs", mcp_inline_max_records=20)


@pytest.fixture
def manager(limits, settings, registry):
    jobs = JobManager(limits, settings, registry)
    yield jobs
    jobs.shutdown()


def call(server: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(name, args)
            return result.structured_content or json.loads(result.content[0].text)

    return asyncio.run(run())


def test_tools_are_listed_with_descriptions(limits, manager):
    server = build_server(LocalBackend(limits, manager), limits)

    async def run() -> list[Any]:
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(run())}
    assert set(tools) == {"list_sample_schemas", "get_sample_schema", "validate_schema", "generate_data", "job_status",
                          "list_jobs", "stop_job", "delete_job", "list_job_files", "preview_job_output"}
    props = tools["generate_data"].input_schema["properties"]
    assert "description" in props["request"] and "description" in props["records"]
    assert not {"provider", "model", "seed"} & set(props)  # no model knobs: mode only


def test_small_requests_return_rows_inline(limits, manager):
    server = build_server(LocalBackend(limits, manager), limits)
    result = call(server, "generate_data", {"request": REQUEST, "records": 5, "mode": "mock"})
    assert result["ok"] and result["mode"] == "inline" and len(result["rows"]) == 5
    assert set(result["rows"][0]) == set(EMPLOYEE_SCHEMA)
    as_text = call(server, "generate_data", {"request": json.dumps(REQUEST), "records": 3, "mode": "mock"})
    assert len(as_text["rows"]) == 3


def test_large_requests_become_jobs_with_a_path(limits, manager):
    server = build_server(LocalBackend(limits, manager), limits)
    job = call(server, "generate_data", {"request": REQUEST, "records": 500, "mode": "mock", "output_format": "excel",
                                         "compress": "zip", "wait_seconds": 60})
    assert job["mode"] == "job" and job["status"] == "completed"
    assert "rows" not in job
    data = next(f for f in job["files"] if f["kind"] == "data")
    assert data["name"] == "generated_dataset.xlsx.zip" and Path(data["path"]).exists()
    listing = call(server, "list_job_files", {"job_id": job["job_id"]})
    assert listing["records"] == 500
    status = call(server, "job_status", {"job_id": job["job_id"]})
    assert status["status"] == "completed" and status["progress"]["percent"] == 100.0
    preview = call(server, "preview_job_output", {"job_id": job["job_id"], "rows": 2})
    assert preview["returned"] == 2
    assert job["job_id"] in [j["job_id"] for j in call(server, "list_jobs", {})["jobs"]]
    assert call(server, "stop_job", {"job_id": job["job_id"]})["ok"]
    assert call(server, "delete_job", {"job_id": job["job_id"]})["deleted"] == job["job_id"]
    assert not Path(data["path"]).exists()


def test_samples_and_validation(limits, manager):
    server = build_server(LocalBackend(limits, manager), limits)
    names = [s["name"] for s in call(server, "list_sample_schemas", {})["samples"]]
    assert "employees_mock" in names
    assert "schema" in call(server, "get_sample_schema", {"name": "employees_mock"})["request"]
    report = call(server, "validate_schema", {"sample": "employees_mock", "records": 50_000})
    assert report["ok"] and report["mode"] == "job"


@pytest.mark.parametrize("args,needle", [
    ({"request": {"schema": {"a b": "x"}, "total_records": 3}}, "Invalid field name"),
    ({}, "pass either request"),
    ({"sample": "nope"}, "unknown sample"),
])
def test_errors_come_back_as_readable_json(limits, manager, args, needle):
    server = build_server(LocalBackend(limits, manager), limits)
    result = call(server, "validate_schema", args)
    assert result["ok"] is False and needle in result["error"]
    assert call(server, "job_status", {"job_id": "missing"})["ok"] is False


def test_http_backend_forwards_to_the_api(limits, settings, registry):
    from fastapi.testclient import TestClient  # pylint: disable=import-outside-toplevel

    from app.api.server import create_app  # pylint: disable=import-outside-toplevel

    with TestClient(create_app(limits, settings, registry)) as api:
        backend = HttpBackend("http://testserver", limits, client=api)
        server = build_server(backend, limits)
        inline = call(server, "generate_data", {"request": REQUEST, "records": 4, "mode": "mock"})
        assert inline["mode"] == "inline" and len(inline["rows"]) == 4
        assert api.get("/jobs").json() == []  # the helper job of an inline sample is removed
        job = call(server, "generate_data", {"request": REQUEST, "records": 300, "mode": "mock", "wait_seconds": 60})
        assert job["status"] == "completed"
        url = next(f["url"] for f in job["files"] if f["kind"] == "data")
        assert url == f"http://testserver/jobs/{job['job_id']}/files/generated_dataset.csv"
        assert api.get(url).status_code == 200
        bad = call(server, "validate_schema", {"request": {"schema": {"a b": "x"}, "total_records": 3}})
        assert bad["ok"] is False and bad["problems"]


def _example_client():
    spec = importlib.util.spec_from_file_location("mcp_llm_client", ROOT / "examples" / "mcp_llm_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_example_client_simplifies_schemas_for_small_models():
    module = _example_client()
    schema = {"type": "object", "title": "x", "properties": {
        "request": {"anyOf": [{"type": "object"}, {"type": "string"}, {"type": "null"}], "description": "d"},
        "records": {"anyOf": [{"type": "integer"}, {"type": "null"}], "title": "Records"}}}
    simple = module._simplify(schema)
    assert simple["properties"]["request"] == {"type": "object", "description": "d"}
    assert simple["properties"]["records"] == {"type": "integer"}


def test_example_client_runs_tool_calls_until_the_model_answers(limits, manager, capsys):
    module = _example_client()
    server = build_server(LocalBackend(limits, manager), limits)

    class FakeLLM:
        """First asks for a tool call, then answers with what the tool returned."""

        def __init__(self) -> None:
            self.seen: list[dict[str, Any]] = []

        async def step(self, messages, tools):  # noqa: ANN001, ANN202
            self.seen = messages
            assert any(t["function"]["name"] == "generate_data" for t in tools)
            if messages[-1]["role"] == "user":
                return {"role": "assistant", "content": ""}, [("1", "generate_data", {"request": REQUEST,
                                                                                       "records": 2, "mode": "mock"})]
            rows = json.loads(messages[-1]["content"])["rows"]
            return {"role": "assistant", "content": f"made {len(rows)} rows"}, []

        @staticmethod
        def tool_message(call_id, name, text):  # noqa: ANN001, ANN205
            return {"role": "tool", "tool_name": name, "content": text}

    async def run() -> None:
        async with Client(server) as client:
            tools = module.to_llm_tools((await client.list_tools()).tools)
            await module.chat_loop(client, FakeLLM(), tools, "make two employees", 4)

    asyncio.run(run())
    assert "made 2 rows" in capsys.readouterr().out
