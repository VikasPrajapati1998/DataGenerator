"""Example MCP client: lets any tool-calling LLM use the data generator's MCP server.

Ollama runs models; it does not connect to MCP servers by itself. This script is the bridge: it starts
(or connects to) the MCP server, gives the model the server's tools, runs the tool calls the model asks
for, and sends the results back, until the model answers.

Works with:
* Ollama (native ``/api/chat``), any model with tool support, e.g. qwen3, llama3.1/3.2, gpt-oss, mistral.
* Any OpenAI-compatible chat API (``--llm openai``): OpenAI GPT models, OpenRouter, NVIDIA, LM Studio, vLLM,
  or Ollama's own ``/v1`` endpoint.

Examples (Windows cmd, from the project root, venv active)::

    python examples\\mcp_llm_client.py --list-tools
    python examples\\mcp_llm_client.py --call validate_schema --args "{\\"sample\\": \\"employees_mock\\"}"
    python examples\\mcp_llm_client.py --model qwen3:4b --prompt "Generate 5 sample employees with the mock model"
    python examples\\mcp_llm_client.py --model gpt-oss:120b-cloud
    set OPENAI_API_KEY=sk-...
    python examples\\mcp_llm_client.py --llm openai --model gpt-4.1-mini
    python examples\\mcp_llm_client.py --server http://127.0.0.1:8765/mcp     (server started with --transport http)

Only the standard tools (``mcp`` and ``httpx``) are needed, so the file can be copied into other projects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT = (
    "You can generate synthetic datasets with the tools provided. Use list_sample_schemas or get_sample_schema "
    "to learn the schema format, validate_schema before large runs, and generate_data to run. Large runs return "
    "a job_id: check job_status, then report the files from list_job_files and a few rows from "
    "preview_job_output. Never try to print a whole large dataset. Use mode 'mock' when the user asks for an "
    "offline or test run; otherwise mode 'llm'."
)
MAX_TOOL_RESULT_CHARS = 8000


def _simplify(schema: Any) -> Any:
    """Make a JSON schema easier for small models: ``anyOf [X, null]`` -> ``X``, no titles."""
    if isinstance(schema, list):
        return [_simplify(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "anyOf" in schema:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        base = {k: v for k, v in schema.items() if k != "anyOf"}
        return _simplify({**options[0], **base}) if options else _simplify(base)
    return {k: _simplify(v) for k, v in schema.items() if k != "title"}


def to_llm_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """MCP tool list -> the ``tools`` array of an Ollama / OpenAI chat request."""
    return [{"type": "function", "function": {"name": t.name, "description": (t.description or "").strip(),
                                              "parameters": _simplify(t.input_schema)}} for t in tools]


def result_text(result: Any) -> str:
    """Tool result as text for the model (structured JSON when available), capped in length."""
    if getattr(result, "structured_content", None):
        text = json.dumps(result.structured_content, ensure_ascii=False, default=str)
    else:
        text = "\n".join(getattr(part, "text", "") for part in result.content)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + " ...[truncated]"
    return text


class OllamaChat:
    """Ollama's native chat API with tool calling."""

    def __init__(self, host: str, model: str) -> None:
        self.http = httpx.AsyncClient(base_url=host.rstrip("/"), timeout=600)
        self.model = model

    async def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
                   ) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]:
        response = await self.http.post("/api/chat", json={"model": self.model, "messages": messages,
                                                           "tools": tools, "stream": False})
        response.raise_for_status()
        message = response.json()["message"]
        calls = []
        for i, call in enumerate(message.get("tool_calls") or []):
            args = call["function"].get("arguments") or {}
            calls.append((str(i), call["function"]["name"], json.loads(args) if isinstance(args, str) else args))
        return message, calls

    @staticmethod
    def tool_message(call_id: str, name: str, text: str) -> dict[str, Any]:  # noqa: ARG004
        return {"role": "tool", "tool_name": name, "content": text}


class OpenAIChat:
    """Any OpenAI-compatible ``/chat/completions`` endpoint with tool calling."""

    def __init__(self, base_url: str, model: str, api_key: str | None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=600)
        self.model = model

    async def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
                   ) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]:
        response = await self.http.post("/chat/completions", json={"model": self.model, "messages": messages,
                                                                   "tools": tools})
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        calls = [(c["id"], c["function"]["name"], json.loads(c["function"].get("arguments") or "{}"))
                 for c in message.get("tool_calls") or []]
        return message, calls

    @staticmethod
    def tool_message(call_id: str, name: str, text: str) -> dict[str, Any]:  # noqa: ARG004
        return {"role": "tool", "tool_call_id": call_id, "content": text}


def server_target(spec: str) -> Any:
    """``stdio`` -> start ``python -m app.mcp_server`` from this project; a URL -> streamable HTTP."""
    if spec.startswith(("http://", "https://")):
        return spec
    return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT),
                                 env={"PYTHONIOENCODING": "utf-8"})


async def chat_loop(client: Client, llm: Any, tools: list[dict[str, Any]], prompt: str | None,
                    max_steps: int) -> None:
    """Ask, run tool calls, repeat until the model answers (or ``max_steps`` tool rounds)."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        question = prompt if prompt is not None else input("\nyou> ").strip()
        if question.lower() in ("", "exit", "quit"):
            return
        messages.append({"role": "user", "content": question})
        for _ in range(max_steps):
            message, calls = await llm.step(messages, tools)
            messages.append(message)
            if not calls:
                print(f"\nassistant> {message.get('content', '').strip()}")
                break
            for call_id, name, args in calls:
                print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)[:300]})")
                result = await client.call_tool(name, args)
                text = result_text(result)
                print(f"  [result] {text[:300]}{'...' if len(text) > 300 else ''}")
                messages.append(llm.tool_message(call_id, name, text))
        else:
            print(f"\n(stopped after {max_steps} tool rounds)")
        if prompt is not None:
            return


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="stdio", help="'stdio' (start the server) or an MCP URL")
    parser.add_argument("--llm", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default=os.getenv("MCP_CLIENT_MODEL", "qwen3:4b"),
                        help="tool-calling model (default qwen3:4b; small models call tools less reliably)")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--prompt", help="ask one question and exit (default: interactive)")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--list-tools", action="store_true", help="print the server's tools and exit")
    parser.add_argument("--call", metavar="TOOL", help="call one tool directly (no LLM) and print the result")
    parser.add_argument("--args", default="{}", help="JSON arguments for --call")
    args = parser.parse_args()

    async with Client(server_target(args.server)) as client:
        listed = (await client.list_tools()).tools
        if args.list_tools:
            for tool in listed:
                print(f"{tool.name}: {(tool.description or '').strip().splitlines()[0]}")
            return 0
        if args.call:
            result = await client.call_tool(args.call, json.loads(args.args))
            print(result_text(result))
            return 1 if result.is_error else 0
        host = args.ollama_host if args.ollama_host.startswith("http") else f"http://{args.ollama_host}"
        llm = OllamaChat(host, args.model) if args.llm == "ollama" else \
            OpenAIChat(args.openai_base_url, args.model, os.getenv("OPENAI_API_KEY"))
        print(f"Connected: {len(listed)} tools. Model: {args.llm}/{args.model}. Type 'exit' to quit.")
        try:
            await chat_loop(client, llm, to_llm_tools(listed), args.prompt, args.max_steps)
        except httpx.HTTPStatusError as exc:
            print(f"LLM request failed: {exc.response.status_code} {exc.response.text[:300]}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"Cannot reach the LLM ({exc}). Is Ollama running (ollama serve)?", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
