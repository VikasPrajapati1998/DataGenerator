# MCP server: setup and usage

The MCP server lets LLM apps call the generator as tools: Claude (Desktop, Code), ChatGPT and OpenAI
agents, IDEs such as Cursor or VS Code, and Ollama models through an MCP client. It is built with the
official Python MCP SDK (`mcp` 2.x) and runs on the same engine, models and `.env` settings as the CLI.

## Why large datasets never go through the model

A model's context holds a few hundred thousand tokens at most, and every token costs time and money, so
the server never sends a large dataset to the model:

* `generate_data` returns rows inline only for tiny samples (up to 20 records).
* Anything larger becomes a background job. The model gets a job id, progress, a few preview rows and,
  at the end, a file path (and a download URL when the API is used). The file itself stays on disk.

## Tools

| Tool | What it does |
|---|---|
| `generate_data(request or sample, records, output_format, mode, compress, wait_seconds)` | Tiny runs return rows inline; anything larger starts a job (optionally waits up to 120 s) |
| `job_status(job_id)` | Status and progress, or the error and a hint |
| `list_jobs()` | All jobs |
| `stop_job(job_id)` | Stops a job (finished batches are kept) |
| `delete_job(job_id)` | Deletes a job that is not running, with its files |
| `list_job_files(job_id)` | The dataset and quality report: name, size, local path, and URL when the API is used |
| `preview_job_output(job_id, rows, table)` | First rows of the dataset, or of one table of a multi-table job (at most 50) |
| `validate_schema(request or sample, records)` | Checks a schema without generating |
| `list_sample_schemas`, `get_sample_schema(name)` | Example schemas the model can copy the format from |

The choices are the same as in the API: `output_format` `csv`, `json`, `jsonl` or `excel`; `mode` `mock`
(offline placeholder model) or `llm` (the models configured in `.env`, the default); `compress` `none`, `zip`
or `gzip`. Model selection and tuning stay in `.env`. `request` accepts a request object
(`{"schema": {...}, "total_records": N}`), a bare schema, or the same as JSON text. Errors come back as
`{"ok": false, "error": ..., "problems": [...]}` so the model can fix its input.

Multi-table schemas ([MULTI_TABLE.md](MULTI_TABLE.md)) always run as a job. `validate_schema` shows the
table order and links. `preview_job_output(job_id, rows, table)` shows one table. The sample
`procurement_multi` is a template the model can copy.

## Install (Windows cmd)

```bat
cd D:\Study\DataGenerator
.venv\Scripts\activate
pip install -r requirements.txt
```

Check that the server starts and lists its tools (this uses the example client described below):

```bat
python examples\mcp_llm_client.py --list-tools
python examples\mcp_llm_client.py --call generate_data --args "{\"sample\": \"employees\", \"records\": 5, \"mode\": \"mock\"}"
```

## Run the server

| Transport | Command | Used by |
|---|---|---|
| stdio (default) | `python -m app.mcp_server` | Claude Desktop, Claude Code, Cursor, VS Code, the example client: the app starts the server itself |
| Streamable HTTP | `python -m app.mcp_server --transport http --port 8765` | Clients that connect to a URL: `http://127.0.0.1:8765/mcp` |

You normally do not start the stdio server by hand; the MCP client launches it. For clients that cannot
set a working folder, use the launcher `D:\Study\DataGenerator\run_mcp_server.py`.

### Two ways the server can do the work

* **In the MCP server process (default).** Simple. Jobs run while the MCP client keeps the server running;
  if you close Claude Desktop, a running job stops (resume it later with `python -m app.main resume <job_id>`).
* **Forwarded to the HTTP API.** Start `python -m app.api` once, then give the MCP server
  `--api-url http://127.0.0.1:8000`. Jobs then run in the API, survive closing
  the chat app, are shared by every client, and `list_job_files` returns download URLs. See
  [API_SETUP.md](API_SETUP.md).

## Connect clients

Replace the paths if the project is somewhere else. In JSON files, Windows backslashes are written twice.

### Claude Desktop

Open Settings > Developer > Edit Config (the file is `%APPDATA%\Claude\claude_desktop_config.json`) and add:

```json
{
  "mcpServers": {
    "synthetic-data": {
      "command": "D:\\Study\\DataGenerator\\.venv\\Scripts\\python.exe",
      "args": ["D:\\Study\\DataGenerator\\run_mcp_server.py"]
    }
  }
}
```

To forward to the API, use `"args": ["D:\\Study\\DataGenerator\\run_mcp_server.py", "--api-url", "http://127.0.0.1:8000"]`.
Restart Claude Desktop; the tools appear under the tools icon. Try: "Generate 5 sample employees in mock
mode", then "Start a job for 100000 employees as Excel, zipped, and tell me when the file is ready."

### Claude Code

Run these in Command Prompt (cmd). `--scope user` makes the server available in every folder, not only the
one you run the command in. Everything after `--` is the command that starts the server.

```bat
claude mcp add --scope user synthetic-data -- D:\Study\DataGenerator\.venv\Scripts\python.exe D:\Study\DataGenerator\run_mcp_server.py
claude mcp list
claude mcp get synthetic-data
```

`claude mcp list` should show `synthetic-data: ... - √ Connected`. Inside a Claude Code session, `/mcp` lists the
server and its 10 tools. In PowerShell, write the double dash in quotes (`'--'`), otherwise PowerShell may
swallow it.

To forward jobs to the API, add `--api-url http://127.0.0.1:8000` at the end of the command. To use the HTTP
transport instead, start `python -m app.mcp_server --transport http --port 8765` in another window, then:

```bat
claude mcp add --scope user --transport http synthetic-data-http http://127.0.0.1:8765/mcp
```

The HTTP server must be running whenever Claude Code starts, so stdio is simpler. Remove a server with
`claude mcp remove synthetic-data -s user`.

### opencode (v2)

The quickest way is the CLI, which writes the config for you (`--global` = every project):

```bat
opencode mcp add synthetic-data --global -- D:\Study\DataGenerator\.venv\Scripts\python.exe D:\Study\DataGenerator\run_mcp_server.py
opencode mcp list
```

`opencode mcp list` should show `✓ synthetic-data connected`. Inside opencode, `/mcps` lists the servers.

Or edit the config by hand: `opencode.json` in a project folder, or the global file
(`%USERPROFILE%\.config\opencode\opencode.json`). opencode v2 puts servers under `mcp.servers` and takes the
command as a list:

```json
{
  "mcp": {
    "servers": {
      "synthetic-data": {
        "type": "local",
        "command": ["D:\\Study\\DataGenerator\\.venv\\Scripts\\python.exe", "D:\\Study\\DataGenerator\\run_mcp_server.py"]
      },
      "synthetic-data-http": {
        "type": "remote",
        "url": "http://127.0.0.1:8765/mcp",
        "oauth": false
      }
    }
  }
}
```

Keep only the entry you use. `"oauth": false` matters for the HTTP entry: the server has no login, and opencode
would otherwise try to sign in. opencode v1 used a different shape (names directly under `mcp`, with
`"enabled": true`); if `opencode --version` shows 1.x, use that shape.

### Cursor, VS Code and other IDEs

Use the same command and argument as for Claude Desktop in the IDE's MCP settings (for example
`.cursor\mcp.json`, or `.vscode\mcp.json` with `"servers"` instead of `"mcpServers"`). Clients that take a
URL can use the HTTP transport: `http://127.0.0.1:8765/mcp`.

### ChatGPT and OpenAI models

ChatGPT connectors and OpenAI's API connect to remote MCP servers over HTTPS, not to programs on your PC.
Two options:

* Run `python -m app.mcp_server --transport http --port 8765`, publish it with an HTTPS tunnel or reverse
  proxy, and add that URL as a connector. The server has no login, so only publish it behind a tunnel or
  proxy that asks for one.
* Or use the example client below with `--llm openai`: it runs on your PC, calls GPT models through the
  OpenAI API, and runs the tools locally.

### Ollama models

Ollama runs models and supports tool calling (for example qwen3, llama3.1 / 3.2, mistral, gpt-oss), but the
Ollama server does not connect to MCP servers by itself. It needs an MCP client or bridge between the model
and the tools. `examples\mcp_llm_client.py` is that bridge, and any MCP-capable chat app that supports
Ollama models works too.

## Example client: any tool-calling LLM

`examples\mcp_llm_client.py` (about 200 lines, needs only `mcp` and `httpx`) starts the MCP server, gives
the model the tool list, runs the tool calls the model makes, sends back the results, and repeats until
the model answers. Copy it into other projects to wire these tools into your own LLM app.

```bat
rem Ollama (start Ollama first; pull a tool-calling model)
ollama pull qwen3:4b
python examples\mcp_llm_client.py --model qwen3:4b
python examples\mcp_llm_client.py --model qwen3:4b --prompt "Generate 5 employees in mock mode and show them"
python examples\mcp_llm_client.py --model gpt-oss:120b-cloud --prompt "Start a job for 200000 healthcare patients and tell me where the file is"

rem OpenAI GPT models (or any OpenAI-compatible API: OpenRouter, LM Studio, vLLM, Ollama's /v1)
set OPENAI_API_KEY=sk-...
python examples\mcp_llm_client.py --llm openai --model gpt-4.1-mini
set OPENAI_BASE_URL=http://127.0.0.1:11434/v1
python examples\mcp_llm_client.py --llm openai --model qwen3:4b

rem Connect to an HTTP MCP server instead of starting one
python examples\mcp_llm_client.py --server http://127.0.0.1:8765/mcp --model qwen3:4b
```

| Option | Default | Meaning |
|---|---|---|
| `--server` | `stdio` | `stdio` starts `python -m app.mcp_server`; or an MCP URL |
| `--llm` | `ollama` | `ollama` (native `/api/chat`) or `openai` (`/chat/completions`) |
| `--model` | `qwen3:4b` (or `MCP_CLIENT_MODEL`) | A model with tool support |
| `--ollama-host` | `OLLAMA_HOST` or `http://127.0.0.1:11434` | Ollama server |
| `--openai-base-url` | `OPENAI_BASE_URL` or `https://api.openai.com/v1` | OpenAI-compatible endpoint; key from `OPENAI_API_KEY` |
| `--prompt` | interactive | One question, then exit |
| `--max-steps` | 8 | Tool rounds per question |
| `--list-tools`, `--call TOOL --args JSON` | | Test the server without any LLM |

Model choice matters: models under about 3B parameters (qwen3:0.6b, llama3.2:1b) often call tools with
wrong arguments or not at all. qwen3:4b or larger, llama3.1:8b, or a cloud model works much better. The
client simplifies tool schemas (no `anyOf`) to help small models.

## Settings

There are none to set for the MCP server. Up to 20 records come back as rows; anything larger becomes a job.
`preview_job_output` returns at most 50 rows. The only option is `--api-url` (see above). `mode=llm` uses
the model settings in `.env` (`SDE_MODEL_POOL`, ...), the same as the CLI.

## Test it

```bat
python -m pytest tests\test_mcp_server.py
python scripts\mcp_smoke_test.py
python scripts\api_smoke_test.py
```

`mcp_smoke_test.py` checks the MCP server alone (stdio and streamable HTTP, about 5 s). `api_smoke_test.py`
runs the same MCP checks plus the HTTP API and MCP forwarding to the API. Both use mock mode, no Ollama.

## Troubleshooting

| Problem | Fix |
|---|---|
| Claude Desktop shows the server as failed | Use the full path to `.venv\Scripts\python.exe` and `run_mcp_server.py`; read `%APPDATA%\Claude\logs\mcp-server-synthetic-data.log` |
| `No module named 'mcp.server.mcpserver'` or `fastmcp` errors | Wrong SDK version: `pip install "mcp>=2.3,<3"` in the venv |
| `claude mcp list` or `opencode mcp list` shows failed | Run the same python.exe and run_mcp_server.py command in cmd by hand; it should keep running and wait (Ctrl+C to quit). A Python error there is the real problem |
| `ModuleNotFoundError: app` | Start with `run_mcp_server.py`, or run `python -m app.mcp_server` from the project folder |
| The model never calls a tool | Use a model with tool support (`ollama show <model>` lists `tools` under Capabilities) and a larger one |
| `Cannot reach the LLM` | Start Ollama (`ollama serve`) or check `--ollama-host` |
| Tools fail with "no model reachable" | `mode=llm` uses `SDE_MODEL_POOL` from `.env`; start Ollama, or ask for `mode` `mock` |
| A job stopped when the chat app closed | Forward to the API (`--api-url`), or resume it with `python -m app.main resume <job_id>` |
| `[WinError 10048]` with `--transport http` | Port in use: `--port 8766` |

Server logs go to stderr (shown in the client's MCP log) and to `logs\<date>\`. Never print to stdout in
code the stdio server runs: stdout carries the MCP protocol.
