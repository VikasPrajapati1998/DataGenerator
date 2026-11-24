"""User-facing failures: one clean ERROR line, a hint, a specific exit code, never a traceback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import main as cli
from app.config.settings import get_settings
from app.core.exceptions import ModelUnavailableError
from app.graph.nodes import describe_unavailable
from app.models.mock_adapter import MockBehavior, MockLLMAdapter

BIO = {"id": "unique integer", "segment": ["a", "b"], "bio": "short customer bio"}


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch):
    """Point the CLI's settings at a temporary output/log folder."""
    monkeypatch.setenv("SDE_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SDE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SDE_MODEL_POOL", "")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _write(tmp_path: Path, name: str, request: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(request), encoding="utf-8")
    return str(path)


def _clean(captured) -> None:
    assert "Traceback" not in captured.err and "Traceback" not in captured.out
    assert "During task with name" not in captured.err


def test_no_reachable_model_is_one_clean_error(isolated, capsys) -> None:
    MockLLMAdapter.behaviors["mock-down"] = MockBehavior(available=False)
    config = _write(isolated, "bio.json", {"schema": BIO, "total_records": 20, "model_provider": "mock",
                                           "model_name": "mock-down", "execution_mode": "thread"})
    code = cli.main(["generate", "--config", config])
    captured = capsys.readouterr()
    _clean(captured)
    assert code == cli.EXIT_NO_MODEL
    assert "ERROR: ModelUnavailableError: No usable model" in captured.err
    assert "Hint: Start Ollama" in captured.err
    result = json.loads(captured.out)
    assert result["exit_code"] == 3 and len(result["error"]) <= 303 and result["error_type"] == "ModelUnavailableError"


def test_allow_no_llm_runs_offline_with_placeholder_text(isolated, capsys) -> None:
    MockLLMAdapter.behaviors["mock-down"] = MockBehavior(available=False)
    config = _write(isolated, "bio.json", {"schema": BIO, "total_records": 30, "model_provider": "mock",
                                           "model_name": "mock-down", "execution_mode": "thread"})
    code = cli.main(["generate", "--config", config, "--allow-no-llm"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_OK, captured.err
    final = json.loads(captured.out)["final_output"]
    assert Path(final).exists()


def test_schema_error_exit_code_and_message(isolated, capsys) -> None:
    config = _write(isolated, "bad.json", {"schema": {"fields": {"a": {"type": "integer", "min": 5, "max": 1}}},
                                           "total_records": 10})
    code = cli.main(["generate", "--config", config])
    captured = capsys.readouterr()
    _clean(captured)
    assert code == cli.EXIT_BAD_INPUT and "SchemaError" in captured.err and "min is greater than max" in captured.err


@pytest.mark.parametrize(("argv", "expected"), [
    (["generate", "--config", "does/not/exist.json"], "file not found"),
    (["resume", "nothing_20990101_000000_abcdef"], "not found"),
    (["cancel", "nothing_20990101_000000_abcdef"], "not found"),
])
def test_bad_input_paths(isolated, capsys, argv, expected) -> None:
    assert cli.main(argv) == cli.EXIT_BAD_INPUT
    captured = capsys.readouterr()
    _clean(captured)
    assert expected in captured.err and captured.err.startswith("ERROR:")


def test_bad_env_setting_is_reported_cleanly(isolated, capsys, monkeypatch) -> None:
    monkeypatch.setenv("SDE_MAX_RETRIES", "lots")
    get_settings.cache_clear()
    config = _write(isolated, "ok.json", {"schema": {"id": "unique integer"}, "total_records": 5})
    assert cli.main(["generate", "--config", config]) == cli.EXIT_BAD_INPUT
    captured = capsys.readouterr()
    _clean(captured)
    assert "SDE_MAX_RETRIES" in captured.err


def test_unexpected_bug_ends_with_clean_internal_error(isolated, capsys, monkeypatch) -> None:
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    def explode(self, request):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(SyntheticDataEngine, "run", explode)
    config = _write(isolated, "ok.json", {"schema": {"id": "unique integer"}, "total_records": 5})
    assert cli.main(["generate", "--config", config]) == cli.EXIT_INTERNAL
    captured = capsys.readouterr()
    _clean(captured)
    assert "internal error" in captured.err and "simulated bug" in captured.err


def test_interrupt_exits_130_and_releases_the_engine(isolated, capsys, monkeypatch) -> None:
    from app.engine import SyntheticDataEngine  # pylint: disable=import-outside-toplevel

    released = []

    def interrupted(self, request):
        raise KeyboardInterrupt

    monkeypatch.setattr(SyntheticDataEngine, "run", interrupted)
    monkeypatch.setattr(SyntheticDataEngine, "shutdown_now", lambda self: released.append(True))
    monkeypatch.setattr(cli, "_hard_exit", lambda code: None)
    config = _write(isolated, "ok.json", {"schema": {"id": "unique integer"}, "total_records": 5})
    assert cli.main(["generate", "--config", config]) == cli.EXIT_INTERRUPTED
    captured = capsys.readouterr()
    _clean(captured)
    assert released and "resume" in captured.err


def test_unreachable_server_is_described_once() -> None:
    detail = "Ollama server not reachable at http://127.0.0.1:11434"
    status = {key: {"available": False, "detail": detail}
              for key in ("ollama_local/qwen3:0.6b", "ollama_local/llama3.2:1b", "ollama_local/gpt-oss:120b-cloud")}
    message = describe_unavailable(status)
    assert message.count("not reachable") == 1 and "ollama serve" in message
    assert "qwen3:0.6b, ollama_local/llama3.2:1b, ollama_local/gpt-oss:120b-cloud" in message
    assert isinstance(ModelUnavailableError(message), Exception)


def test_localhost_is_rewritten_on_windows(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setattr("sys.platform", "win32")
    get_settings.cache_clear()
    try:
        assert get_settings().ollama_host == "http://127.0.0.1:11434"
    finally:
        get_settings.cache_clear()
