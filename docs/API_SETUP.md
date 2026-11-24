# HTTP API (FastAPI): setup and usage

Upload a schema JSON file, the server generates the data in the background, and you download the file.
It uses the same engine as the CLI (`python -m app.main generate`).

## The whole API

| Method and path | What it is for |
|---|---|
| `POST /jobs` | Upload a schema file and start generating. Returns a job id at once. |
| `GET /jobs` | List all jobs, newest first. |
| `GET /jobs/{id}` | Status and progress (records done, percent, phase), or the error and a hint. |
| `POST /jobs/{id}/stop` | Stop a queued or running job. Finished batches are kept. |
| `POST /jobs/{id}/resume` | Continue a stopped, failed or interrupted job (only the missing part is generated). |
| `DELETE /jobs/{id}` | Delete a job that is not running, with all its files. |
| `GET /jobs/{id}/files` | List the generated files: the dataset and its quality report, with download URLs. |
| `GET /jobs/{id}/files/{name}` | Download one file. |
| `GET /jobs/{id}/preview?rows=10` | Look at the first rows without downloading. |
| `GET /health` | Is the server up, and which formats / modes / packaging it accepts. |

Interactive docs, where you can try every endpoint in the browser: `http://127.0.0.1:8000/docs`.

## Starting a job: four choices only

`POST /jobs` is a form upload (`multipart/form-data`):

| Field | Values | Default |
|---|---|---|
| `file` | the schema JSON file (required) | |
| `records` | number of records | `total_records` from the file |
| `format` | `csv`, `json`, `jsonl`, `excel` | the file's `output_format`, else `csv` |
| `mode` | `mock` = offline placeholder model, fast, for testing; `llm` = the real models from `.env` | `llm` |
| `compress` | `none`, `zip`, `gzip` | `none` |

Everything else is decided by the server from `.env`: which models (`SDE_MODEL_POOL`), batch sizes, CPU
queues, retries, pooled LLM mode for big runs, and so on. Model settings inside an uploaded file are ignored;
`mode` decides. The job is named after the uploaded file (`employees.json` gives `employees_<date>_<time>_<id>`).

The schema file is the same JSON as the files in `samples\schemas`: either a request with a `schema` key, or
just the schema itself:

```json
{
  "schema": {
    "employee_id": "unique integer",
    "name": "realistic full name",
    "email": "valid email",
    "age": "integer between 22 and 60",
    "department": ["IT", "HR", "Finance", "Sales"],
    "salary": "realistic salary based on department"
  },
  "total_records": 100000,
  "generation_instructions": "Generate realistic Indian employee records."
}
```

A file with a `tables` object describes several related tables (see [MULTI_TABLE.md](MULTI_TABLE.md)). Leave
`records` empty for it, because each table sets its own size. The job then delivers one file per table.
`compress=zip` packs all of them into one archive. `GET /jobs/{id}/preview?table=<name>` shows one table.

## Why large datasets are fine

Every request is a background job, so there is no big HTTP response to wait for. The engine writes batches to
disk and merges them into one file; downloads are streamed from disk with HTTP range support, so millions of
rows never sit in memory and a broken download can continue where it stopped (`curl -C -`). Requests above
`SDE_MAX_RECORDS_PER_JOB` (100,000) run as sequential sub-jobs that are merged at the end.

Limits: one job is capped at 50,000,000 records by the engine; for more, run several jobs. Excel holds at most
1,048,575 data rows per sheet, so a bigger Excel file continues on sheets `data_2`, `data_3`, ... Excel is slow
to write and open at that size; prefer CSV with `compress=zip` for millions of rows.

## Install (Windows cmd)

```bat
cd D:\Study\DataGenerator
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `fastapi`, `uvicorn`, `python-multipart`, `openpyxl` (Excel) and `mcp`. Without a
venv yet, create it first with `python -m venv .venv`.

## Start the server

```bat
python -m app.api
```

It listens on `http://127.0.0.1:8000`. Option: `--port 9000` if that port is taken. Stop it with Ctrl+C: running jobs are stopped and can be resumed after the restart.

## Examples (Windows cmd, `curl.exe` ships with Windows 10 and 11)

```bat
rem Start a job: 10,000 records, CSV, mock mode (offline)
curl -F "file=@samples\schemas\employees.json" -F records=10000 -F mode=mock http://127.0.0.1:8000/jobs

rem Real models from .env, Excel, zipped
curl -F "file=@my_schema.json" -F records=50000 -F format=excel -F compress=zip http://127.0.0.1:8000/jobs

rem Progress (replace JOB_ID with the job_id from the answer)
curl http://127.0.0.1:8000/jobs/JOB_ID

rem Files of the finished job, then download one
curl http://127.0.0.1:8000/jobs/JOB_ID/files
curl -C - -o employees.csv http://127.0.0.1:8000/jobs/JOB_ID/files/generated_dataset.csv

rem Look at a few rows
curl "http://127.0.0.1:8000/jobs/JOB_ID/preview?rows=5"

rem Manage jobs
curl http://127.0.0.1:8000/jobs
curl -X POST http://127.0.0.1:8000/jobs/JOB_ID/stop
curl -X POST http://127.0.0.1:8000/jobs/JOB_ID/resume
curl -X DELETE http://127.0.0.1:8000/jobs/JOB_ID
```

The data file is called `generated_dataset.<csv|json|jsonl|xlsx>`, plus `.zip` or `.gz` when packed; the
quality report is `validation_summary.json` (or `split_summary.json` for split runs). No login or key is
needed: in Swagger (`/docs`) click an endpoint, then "Try it out", choose the file and press "Execute".

### Python client

```python
import time
import httpx

api = httpx.Client(base_url="http://127.0.0.1:8000", timeout=600)
with open("my_schema.json", "rb") as schema:
    job = api.post("/jobs", files={"file": schema},
                   data={"records": 1_000_000, "format": "csv", "mode": "llm", "compress": "zip"}).json()

while (status := api.get(f"/jobs/{job['job_id']}").json())["status"] in ("queued", "running"):
    print(status["progress"])
    time.sleep(5)

for item in api.get(f"/jobs/{job['job_id']}/files").json()["files"]:
    with api.stream("GET", item["url"]) as response, open(item["name"], "wb") as out:
        for chunk in response.iter_bytes():
            out.write(chunk)
```

## Job states

`queued` (waiting for a free slot; one job runs at a time by default), `running`, `completed`, `failed`
(the status shows the error and a hint), `cancelled` (stopped), `interrupted` (the server was stopped while
the job was active). `resume` works for `failed`, `cancelled` and `interrupted`.

## Settings

There are none to set for the API. The only choices are the four fields of `POST /jobs` above. Fixed
limits: at most 50,000,000 records per job, and one job runs at a time (others wait as `queued`). Job
records are kept in `output\api_jobs` and job folders in `output\Jobs_<date>\<name>\<job_id>\`, so the CLI
can also resume an API job: `python -m app.main resume <job_id>`. `mode=llm` uses the model settings in
`.env` (`SDE_MODEL_POOL`, `SDE_LLM_MODE`, ...), the same as the CLI; see the README.

## Test it

```bat
python -m pytest tests\test_api.py tests\test_mcp_server.py
python scripts\api_smoke_test.py
```

The smoke test starts its own server on a free port with a temporary output folder (mock mode, no Ollama) and
checks the whole flow: upload, progress, files, downloads in every format and packaging, stop, resume, delete,
and the MCP server. It ends with `API SMOKE TEST PASSED`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: fastapi`, `uvicorn` or `openpyxl` | Activate the venv, then `pip install -r requirements.txt` |
| `[WinError 10048]` address already in use | Another program uses port 8000: `python -m app.api --port 9000` |
| `422` with `problems` | The schema file is invalid, or `records` is missing; the message names the field |
| Job `failed`, hint mentions Ollama | `mode=llm` needs the models from `.env`: start `ollama serve` and pull them, or use `mode=mock` |
| Job `failed` with `FeasibilityError` | The LLM is too slow for that many records: set `SDE_LLM_MODE=pooled`, use a faster model, or raise `SDE_MAX_ETA_HOURS` |
| `409` on `/files` | The job is not `completed` yet; check `GET /jobs/{id}` |
| Job shows `interrupted` | The server was stopped while it ran: `POST /jobs/{id}/resume` |

Logs are in `logs\<date>\`, as for the CLI.
