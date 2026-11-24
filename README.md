# Intelligent Dynamic Parallel Synthetic Data Generation Engine

Generate large, realistic and validated synthetic datasets from a schema. The
engine plans a power-of-two number of tasks, spreads them over CPU-based
logical queues, generates each batch with a hybrid of Python, Faker and an LLM,
validates every batch with node-level guardrails, re-checks the whole dataset
with global guardrails, and merges everything into one file with a deterministic
Python merger.

**Stack:** Python 3.11+, LangGraph 1.2 (`Send` API), Ollama (local and cloud),
OpenAI-compatible providers (NVIDIA NIM, OpenRouter), Faker, Pydantic v2,
NumPy, SQLite, psutil.

## Highlights

* **Power-of-two planning.** `N = 2^ceil(log2(ceil(records / batch_size)))`.
  Records are split evenly (sizes differ by at most one) and never exceed the
  request. 100,000 records at batch size 1,000 gives 128 tasks of 781 or 782 records.
* **CPU-aware scheduling.** Detects cores, RAM, GPU and disk. Tasks go to
  balanced logical core queues (`divmod`). Each queue runs one task at a time;
  queues run concurrently. Work stealing and CPU pinning are optional.
* **Resource-aware concurrency.** Queue count respects cores, RAM (minus the
  loaded model's size), `max_workers` and task count. LLM concurrency respects
  `OLLAMA_NUM_PARALLEL` or cloud rate limits.
* **Multiple providers** behind one adapter interface: Ollama Local, Ollama
  Cloud, NVIDIA NIM (Nemotron), OpenRouter (Reflection AI Beam), any
  OpenAI-compatible server, plus a deterministic mock. It includes a central
  registry, availability checks, fallback chains, retries with backoff, rate
  limiting and a circuit breaker. API keys are never logged.
* **Hybrid generation.** Python sequences for IDs, Faker for identities, curated
  city/state tables for geographic consistency, explicit distributions for
  numbers. Category-dependent ranges (for example salary by department) are
  calibrated once by the LLM. The LLM fills only the fields that need context.
* **Guardrails at two levels.**
  * Node level: Pydantic schema, business rules, realism, in-batch and cross-task
    uniqueness (shared SQLite registry), semantic near-duplicates, prompt-injection
    resistance and targeted regeneration of only the rejected records.
  * Global level: completeness, dataset-wide re-validation, cross-batch exact and
    semantic duplicates, referential integrity, distribution consistency and a
    quality score. Flagged records are repaired automatically.
* **Recoverable.** Atomic writes, checksummed task status files, idempotent
  tasks, `resume` that never regenerates verified batches, and graceful
  cancellation (Ctrl+C, or `cancel` from another terminal).
* **Observable.** Live progress lines plus `execution_summary.json` and
  `validation_summary.json` per job.

## Quick start (Windows cmd)

```bat
cd D:\Study\DataGenerator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

rem Offline run with the deterministic mock model (no Ollama needed)
python -m app.main generate --config samples\schemas\employees_mock.json --records 10000

rem Real models via Ollama (uses the SDE_MODEL_POOL from .env; pull those models first)
python -m app.main generate --config samples\schemas\employees.json --records 5000

rem Smoke test: every sample schema end to end in mock mode, plus output and reproducibility checks
python scripts\smoke_test.py

rem HTTP API and MCP server (see "HTTP API and MCP server" below)
python -m app.api
python scripts\api_smoke_test.py
```

On Linux or macOS, use `source .venv/bin/activate` and forward slashes. The
output path is printed at the end. Jobs are grouped by day and schema:

```text
output/Jobs_2026-11-20/
├── employees/
│   ├── employees_20261120_185312_84be8c/   # one folder per run, never overwritten
│   │   ├── metadata/  batches/  rejected/
│   │   └── final/generated_dataset.csv
│   └── employees_20261120_191004_1f2e3d/
└── ecommerce_orders/
    └── ecommerce_orders_20261120_192210_a9b8c7/
```

The schema name is the request file name (`--config samples\schemas\employees.json` gives
`employees`). Override it with `--schema-name` or `"schema_name"` in the request. A request with no
name falls back to the rich schema's `entity`, then to `dataset`. The job id is
`<schema_name>_<YYYYMMDD_HHMMSS>_<6 hex>`, and `resume` and `cancel` accept either the id or the
folder path. Jobs from the older `output/jobs/job_<id>` layout can still be resumed. See
[docs/architecture.md](docs/architecture.md#4-job-directory) for the folder contents.

## Mock mode vs Ollama mode

| | Mock mode | Ollama mode |
|---|---|---|
| Command | `python -m app.main generate --config samples\schemas\employees_mock.json --records 10000` (or any sample with `--provider mock --model mock-llm`) | `python -m app.main generate --config samples\schemas\employees.json --records 5000` |
| Models | `mock/mock-llm`, offline, no Ollama needed | The model pool: request `model_pool` > `--model` > `SDE_MODEL_POOL` in `.env` (default: `qwen3:0.6b`, `llama3.2:1b`, `gemma3:1b`, `gpt-oss:120b-cloud`), one model per task, round-robin |
| Rule-based fields (IDs, names, emails, numbers, enums, dates, regex, references) | Generated by Python | Generated by the same Python code |
| `profile` ranges without explicit values (e.g. "salary based on department") | Placeholder ranges from a hash, not realistic | Calibrated once per job by the strongest available model (cloud first), temperature 0, fixed seed |
| `llm` text fields | Placeholder template text (reads like a short bio whatever the schema) | Written by the pool models; from `SDE_POOLED_MIN_RECORDS` (10,000) on, recombined from a pool of answers, so some text repeats on purpose |
| Reproducible with `--seed` | Yes, the whole file | Rule-based fields yes; LLM text no |

Schemas without `llm` fields (like `employees.json`) call the model only to calibrate ranges, so every
row is rule-based in both modes. In both modes the guardrails enforce unique fields and whole-record
uniqueness (all fields except `sequence`/`uuid` IDs) across all batches, so whole-row duplicates do not
occur. The whole-record check is switched off only when the non-ID fields allow fewer than 2x the
requested combinations, and then repeats apart from the ID are legitimate.

Mock mode is meant for pipeline tests and dry runs. Do not judge realism of text or calibrated
amounts from it.

### Reproducibility (`seed`)

Every record's random values are seeded from the job seed, a fingerprint of the schema and the
record's global row number. So:

* the same seed and schema give the same rows on any machine, whatever the batch size, core count,
  worker count or `SDE_BATCH_SIZE`;
* different schemas generated with the same seed (for example patients and appointments, both seed 42)
  no longer get matching random values row by row (before this fix, row N of appointments carried
  the MRN of patient N and the patient's name as the doctor);
* mock LLM answers depend only on the prompt, not on the order threads call the model, so mock text
  is reproducible too.

Two exceptions: when two records collide on a unique field (rare, for example two identical emails),
which one is regenerated can depend on task timing, so a handful of rows per 10,000 may differ. And
date fields without `min`/`max` use a window ending today, so their values move with the run date;
give such fields explicit bounds for reproducible data. Changing the schema itself changes the seed
fingerprint, so changing one field gives different rows.

## Sample schemas

All samples live in `samples\schemas\`. Run any of them with
`python -m app.main generate --config samples\schemas\<file> [--records N]`. The samples contain only
the data definition. Batch size is chosen at run time, and models come from `SDE_MODEL_POOL` in
`.env` (or `--model` on the command line). The exceptions are `employees.json` (the specification's
example), `employees_mock.json` (offline mock model) and `customer_profiles_pool.json` (demonstrates
a per-request pool). A schema that needs no LLM skips model checks entirely. Most samples have no
LLM text fields: their category ranges are fixed in the schema, so they run in seconds and need no
model. The ones marked *LLM* have a free-text field and therefore need a model (they are slow on CPU,
so start with `--records 200`).

| File | Domain | Highlights | Example |
|---|---|---|---|
| `employees.json` / `employees_mock.json` | HR | shorthand schema; salary by department calibrated by the LLM, correlated with age | `--config samples\schemas\employees_mock.json --records 10000` |
| `hr_payroll.json` | HR / payroll | basic and HRA by grade (rule: HRA < basic), PAN format, work email | `--config samples\schemas\hr_payroll.json --records 5000` |
| `automobile_vehicles.json` | Automobile | VIN and registration formats, price by segment, manufacture → purchase → service dates | `--config samples\schemas\automobile_vehicles.json --records 5000` |
| `automobile_service_records.json` | Automobile, *LLM* | labour/parts cost by service type, chain of previous services (reference), workshop notes | `--config samples\schemas\automobile_service_records.json --records 200` |
| `pharma_drug_batches.json` | Pharma | batch numbers, MRP by product, expiry 18-36 months after manufacture, QC status | `--config samples\schemas\pharma_drug_batches.json --records 5000` |
| `pharma_prescriptions.json` | Pharma | diagnosis + matching drug, bill by treatment, dispensed after prescribed (JSONL) | `--config samples\schemas\pharma_prescriptions.json --records 5000` |
| `healthcare_patients.json` | Healthcare | MRN, blood-group distribution, height/weight by gender (weight correlated with height), city/state consistent | `--config samples\schemas\healthcare_patients.json --records 5000` |
| `healthcare_appointments.json` | Healthcare | fee by department, appointment slots, follow-ups referencing earlier appointments | `--config samples\schemas\healthcare_appointments.json --records 5000` |
| `healthcare_lab_results.json` | Healthcare | result value by test (units in the test name), reported after collection | `--config samples\schemas\healthcare_lab_results.json --records 5000` |
| `education_students.json` | Education | roll numbers, fee by program, CGPA and attendance distributions, minimum age rule | `--config samples\schemas\education_students.json --records 5000` |
| `education_course_grades.json` | Education | letter grade → consistent marks and grade points, results after exam | `--config samples\schemas\education_course_grades.json --records 10000` |
| `banking_transactions.json` | Banking | amount by channel (UPI, NEFT, RTGS ≥ 2 lakh...), IFSC, timestamps, lognormal balances | `--config samples\schemas\banking_transactions.json --records 10000` |
| `insurance_claims.json` | Insurance, *LLM* | sum insured and claim by policy type (claim ≤ cover), incident → claim → decision dates | `--config samples\schemas\insurance_claims.json --records 200` |
| `telecom_subscribers.json` | Telecom | ARPU by plan, lognormal data usage, circles, handset mix | `--config samples\schemas\telecom_subscribers.json --records 10000` |
| `travel_hotel_bookings.json` | Travel | rate by hotel category, booked before check-in, 1-10 night stays | `--config samples\schemas\travel_hotel_bookings.json --records 10000` |
| `logistics_shipments.json` | Logistics | weight and declared value by category, carriers, promised date after pickup | `--config samples\schemas\logistics_shipments.json --records 10000` |
| `retail_inventory.json` | Retail | unit cost by category, stock and reorder levels, GST slabs | `--config samples\schemas\retail_inventory.json --records 10000` |
| `ecommerce_orders.json` | E-commerce (US), *LLM* | regex order codes, weighted channels, delivery after order, reviews | `--config samples\schemas\ecommerce_orders.json --records 200` |
| `customer_profiles.json` / `customer_profiles_pool.json` | Retail CRM, *LLM* | age by segment, income calibrated by the LLM; the `_pool` variant spreads tasks over 4 models | `--config samples\schemas\customer_profiles.json --records 200` |

`tests/test_samples.py` generates 400 records from every sample and checks that they all pass schema,
rule and realism validation, so a broken sample is caught by `pytest`.

## CLI

| Command | Purpose |
|---|---|
| `python -m app.main generate --config FILE [--records N] [--format csv\|json\|jsonl] [--provider P] [--model M] [--batch-size B] [--max-workers W] [--mode auto\|thread\|process] [--work-stealing] [--cpu-affinity] [--seed S] [--job-id ID] [--output-dir DIR]` | Run a job |
| `python -m app.main resume <job_id or dir>` | Continue an interrupted or failed job (verified batches are reused) |
| `python -m app.main cancel <job_id or dir>` | Ask a running job to stop gracefully |
| `python -m app.main plan --records N [--batch-size B] [--cores C]` | Show the task plan and core allocation without generating |
| `python -m app.main models list\|check` | List registered models or probe their availability |

Options for every command: `--debug` (before the command, e.g. `python -m app.main --debug generate ...`)
also prints full tracebacks on the console. Without it, the console shows one line per problem, and
the traceback goes to the log file only. `generate --allow-no-llm` lets a job run when no model is
reachable: LLM text fields are filled with placeholder text and a warning is logged.
`generate --llm-mode auto|per_record|pooled` chooses how LLM text is produced, and `generate --force`
starts a job even when its projected run time exceeds `SDE_MAX_ETA_HOURS` (see
[Slow models](#slow-models-speed-test-pooled-mode-and-run-time-limit)).

When something goes wrong, the CLI prints one `ERROR:` line with the cause, a `Hint:` line saying
what to do, and the `resume` command when there is something to resume. The JSON result carries the
same short `error`, `error_type`, `hint`, `resume` and `exit_code`.

| Exit code | Meaning |
|---|---|
| 0 | dataset delivered |
| 1 | job ran but some batches failed (see the log; `resume` retries only those) |
| 2 | bad input: missing or invalid request file, schema error, invalid `.env` value, unknown job |
| 3 | no usable model (e.g. Ollama not running, no credits) |
| 4 | not enough RAM or disk, or the projected run time exceeds `SDE_MAX_ETA_HOURS` |
| 5 | file-system problem (permission, disk full, corrupt files) |
| 70 | internal error (a bug): full traceback in the log file |
| 130 | cancelled or interrupted (Ctrl+C, SIGTERM); finished work is kept |

Python API:

```python
from app.engine import SyntheticDataEngine

result = SyntheticDataEngine().run({
    "schema": {"employee_id": "unique integer", "name": "realistic full name",
               "department": ["IT", "HR"], "salary": "realistic salary based on department"},
    "total_records": 1000,
    "model_provider": "ollama", "model_name": "qwen3:0.6b",
})
print(result.status, result.final_output)
```

## Request format

| Key | Default | Meaning |
|---|---|---|
| `schema` | required | Shorthand or rich schema (below) |
| `total_records` | required | Exact number of records delivered |
| `output_format` | `csv` | `csv`, `json` (array) or `jsonl` |
| `generation_instructions` | none | Free-text guidance (sanitised against prompt injection) |
| `model_provider`, `model_name` | none | Explicit single-model override (e.g. `ollama` + `qwen3:0.6b`). Without it the job uses `model_pool`, then `SDE_MODEL_POOL` from `.env`, then `SDE_DEFAULT_MODEL` |
| `fallback_models` | `[]` | Registry keys tried when the primary fails, e.g. `["ollama_local/llama3.2:1b"]` |
| `model_pool` | `[]` | Models spread over tasks, one per task (see [Model pool](#model-pool-different-models-on-different-tasks)) |
| `model_assignment`, `model_weights` | `round_robin`, none | Pool assignment strategy and weights |
| `batch_size` | chosen at run time | Target records per task (drives the power-of-two count). Leave it out: the planner picks about 8 tasks per core (250-5,000 records per task), or 100 when records need LLM text. Set it, or `SDE_BATCH_SIZE`, only to force a size |
| `quality_requirements` | `{}` | Overrides of [`GuardrailConfig`](app/config/guardrail_config.py) |
| `max_workers` | cores | Cap on logical queues |
| `work_stealing` | `false` | Let idle queues take pending tasks |
| `execution_mode` | `auto` | `thread`, `process`, or `auto` (process for CPU-bound jobs of 20k+ records) |
| `cpu_affinity` | `false` | Pin queue threads to logical CPUs (Windows/Linux) |
| `seed` | random | Reproducible generation |
| `schema_name` | config file name | Job-id prefix and output folder name |

### Schema reference

Shorthand is the fastest way to start:

```json
{
  "employee_id": "unique integer",
  "name": "realistic full name",
  "email": "valid email",
  "age": "integer between 22 and 60",
  "city": "Indian city",
  "department": ["IT", "HR", "Finance", "Sales"],
  "salary": "realistic salary based on department"
}
```

How the shorthand is read: `unique integer` becomes a sequence; `valid email`
is derived from the name and kept unique; `integer between A and B` becomes a
range; `Indian city` picks the locale and uses curated geographic data; a list
becomes an enumeration; `... based on <field>` becomes a category-dependent
profile calibrated by the LLM (and correlated with `age` or experience if
present). Anything the engine cannot derive deterministically becomes an `llm`
field.

The rich form (`{"fields": {...}, "rules": [...]}`) exposes every feature. See
[samples/schemas/ecommerce_orders.json](samples/schemas/ecommerce_orders.json)
and [customer_profiles.json](samples/schemas/customer_profiles.json).

| Field key | Purpose |
|---|---|
| `type` | `integer`, `float`, `string`, `boolean`, `date`, `datetime` |
| `required`, `nullable`, `null_ratio` | Presence rules and the share of nulls |
| `unique` | Globally unique across all tasks |
| `min`, `max` | Numeric or ISO-date bounds |
| `allowed_values`, `weights` | Enumerations with category distribution |
| `pattern` | Regex the value must match (also used to generate it) |
| `min_length`, `max_length`, `decimals` | Formatting constraints |
| `generator` | `sequence`, `uuid`, `faker`, `enum`, `range`, `date_range`, `email`, `geo_city`, `geo_state`, `geo_country`, `profile`, `reference`, `regex`, `llm`, `boolean`, `constant` |
| `faker_method`, `faker_args` | Any Faker provider |
| `depends_on`, `profile` | Category-dependent numeric ranges (`{"IT": {"min", "typical", "max"}}`) |
| `distribution` | `uniform`, `normal`, `lognormal`, `triangular` |
| `correlate_with` | Positive correlation with another numeric field |
| `reference` | Relationship to an earlier record's sequence id (validated globally) |
| `offset_days` | Date relative to another date field |
| `instructions` | Custom LLM guidance for the field |

Schema-level keys: `entity`, `locale`, `rules` (safe expressions such as
`"delivery_date > order_date"` or `"age <= 35 if segment == 'student' else True"`),
`semantic_fields` and `dedup_exclude`.

## HTTP API and MCP server

Two more ways to use the same engine:

* **HTTP API (FastAPI)**: upload a schema JSON file to start a job, follow its progress, stop, resume or
  delete it, list its files and download them. Per job you choose only the number of records, the format
  (`csv`, `json`, `jsonl`, `excel`), the mode (`mock` for offline test data, `llm` for the models in `.env`)
  and the packaging (`none`, `zip`, `gzip`); models and tuning stay in `.env`. Every request is a background
  job and files are streamed from disk, so millions of rows never sit in memory or in one HTTP response.
  Setup, endpoints and examples: [docs/API_SETUP.md](docs/API_SETUP.md).
* **MCP server** (official Python MCP SDK, stdio and streamable HTTP): tools such as `generate_data`,
  `job_status`, `stop_job`, `delete_job` and `list_job_files` for Claude, ChatGPT/OpenAI, IDEs and Ollama
  models. Tiny samples come back inline; large runs return a job id and file paths or links, never the rows.
  Ollama models need an MCP client between them and the tools; `examples\mcp_llm_client.py` is a ready-made
  one for Ollama and any OpenAI-compatible API. Setup: [docs/MCP_SETUP.md](docs/MCP_SETUP.md).
  Claude Code: `claude mcp add --scope user synthetic-data -- D:\Study\DataGenerator\.venv\Scripts\python.exe
  D:\Study\DataGenerator\run_mcp_server.py` (one line), then `claude mcp list`. opencode: the same command after
  `opencode mcp add synthetic-data --global --`, then `opencode mcp list`. Details in MCP_SETUP.md.

* **Multi-table datasets**: one schema file with a `tables` object generates related tables in dependency
  order: foreign keys taken from real parent rows (also composite keys), child rows per parent, inherited
  columns, status-driven columns (`cases`), formulas, and exact counts. Every run checks that the keys join.
  One file per table (zip packs them all). Format and example: [docs/MULTI_TABLE.md](docs/MULTI_TABLE.md),
  `samples\schemas\procurement_multi.json`.

| API endpoint | Purpose |
|---|---|
| `POST /jobs` | Upload a schema file (`records`, `format`, `mode`, `compress`) and start |
| `GET /jobs`, `GET /jobs/{id}` | List jobs; status and progress of one |
| `POST /jobs/{id}/stop`, `POST /jobs/{id}/resume`, `DELETE /jobs/{id}` | Manage a job |
| `GET /jobs/{id}/files`, `GET /jobs/{id}/files/{name}` | List the generated files; download one |
| `GET /jobs/{id}/preview` | First rows without downloading |

```bat
pip install -r requirements.txt

rem API: http://127.0.0.1:8000 (interactive docs at /docs)
python -m app.api
curl -F "file=@samples\schemas\employees.json" -F records=10000 -F mode=mock -F format=excel http://127.0.0.1:8000/jobs
curl http://127.0.0.1:8000/jobs/JOB_ID/files

rem MCP server (clients usually start it themselves) and the example client
python -m app.mcp_server
python -m app.mcp_server --transport http --port 8765
python examples\mcp_llm_client.py --model qwen3:4b

rem End-to-end checks (mock mode): API + MCP (about 20 s), or MCP alone (about 5 s)
python scripts\api_smoke_test.py
python scripts\mcp_smoke_test.py
```

## Models

`app/config/models.json` is the registry. Every entry records the provider,
model id, local or cloud, endpoint, the *name* of the env var holding its key,
context window, output tokens, temperature, timeout, retry policy, concurrency,
rate limit and structured-output mode. Unknown models of a known provider are
created on the fly, so `--model llama3.2:1b` works without editing the file.

| Provider key | Use | Notes |
|---|---|---|
| `ollama_local` | Local Ollama, and `-cloud` models via a signed-in server | JSON-schema structured output for local models |
| `ollama_cloud` | Direct `https://ollama.com/api` | Needs `OLLAMA_API_KEY`. Cloud models have no schema-constrained output, so JSON mode plus validation is used |
| `nvidia_nim` | NVIDIA hosted API (Nemotron) | Needs `NVIDIA_API_KEY`; free developer tier is rate-limited |
| `openrouter` | e.g. Reflection AI Beam 1.0 | Needs `OPENROUTER_API_KEY` |
| `openai_compatible` | vLLM, llama.cpp server, LM Studio, ... | Set `endpoint` |
| `mock` | Tests and offline dry runs | Failure injection for retries and fallback |

### Model pool: different models on different tasks

List the models to use in `model_pool`. The engine gives **each generation task exactly one model**
from the pool and uses every other pool model (plus `fallback_models`) as that task's fallback. To
change the models, edit the list; no code changes. Four models today, ten tomorrow:

```json
{
  "model_pool": ["ollama_local/qwen3:0.6b", "ollama_local/llama3.2:1b",
                 "ollama_local/gemma3:1b", "ollama_local/gpt-oss:120b-cloud"],
  "model_assignment": "round_robin"
}
```

* `round_robin` (default): task 1 uses model 1, task 2 uses model 2, and so on, cycling. Because core
  queues hold contiguous task blocks, every queue ends up using every model.
* `weighted` with `"model_weights": [3, 1, 1, 1]` gives each model a proportional share of tasks,
  interleaved rather than in blocks.
* Unavailable pool models are dropped before the run (logged), and their share goes to the rest.
* The assignment is stored per task in `job_config.json` and the task status files, so `resume` and
  global repairs reuse the same model for the same batch.
* Each model has its own concurrency limit and rate limit (local Ollama: `OLLAMA_NUM_PARALLEL`; cloud:
  requests per minute), shared by all tasks that call it.
* The one-off profile calibration uses the first available pool model and falls back through the rest.
* A default pool for every job can be set with `SDE_MODEL_POOL=ollama_local/qwen3:0.6b,ollama_local/llama3.2:1b`.
* `execution_summary.json` reports `model_pool`, `model_assignment` (tasks per model), `llm_slots` and
  `llm_usage.by_model` (calls actually served per model, including fallbacks).

On a CPU-only machine, Ollama keeps about three models loaded at once (`OLLAMA_MAX_LOADED_MODELS`).
A larger *local* pool works, but models get swapped in and out. Cloud models (`-cloud`) do not use
local RAM.

**Cloud models through your local Ollama.** After `ollama signin`, `-cloud` models such as
`gpt-oss:120b-cloud` and `gemma4:31b-cloud` are registered as `ollama_local/<name>` and need no API
key. Use one as the primary (`--model gpt-oss:120b-cloud`) or as a fallback
(`"fallback_models": ["ollama_local/gemma4:31b-cloud"]`). They count against your Ollama Cloud
quota and are rate-limited client-side (30 requests/minute by default).

**Model identity findings.** "Beam" is Reflection AI's open-weight Beam 1.0
(registered as `ollama_local/beam:cloud` and on OpenRouter). No LLM named
"Iceman" could be confirmed on Ollama, Hugging Face, OpenRouter or NVIDIA NIM,
so it is not registered. Add it to `models.json` once its provider and model id
are known. Details are in [docs/architecture.md](docs/architecture.md#5-research-notes-november-2026).

New providers: subclass `BaseLLMAdapter`, decorate it with
`@register_provider("name")`, and add registry entries. The graph and the
workers do not change.

## Fault tolerance

| Failure | What happens |
|---|---|
| A pool or fallback model is missing (not pulled, wrong id, no API key, server down) | Every model is probed before the run (`/api/tags`, `/models`). Unavailable ones are dropped and logged, and the rest of the pool takes their tasks. The job stops with a clear error only if no usable model is left and the schema needs an LLM. |
| A model hits a limit: rate limit (429), timeout, or a request too long for its context window | The request **switches immediately** to the next healthy model (other pool models, then `fallback_models`), with no retries on the limited model. A rate-limited model pauses for its `Retry-After` (default 60 s) and a timing-out one for 30 s, for every task at once, and then rejoins. A context-window overflow only moves that request, since smaller requests still fit. If the limited model is the only one left, the engine waits and retries it instead. This applies to every generation task, sub-batch, repair and the profile calibration. |
| A model runs out of credits/quota (402, or 403/429 mentioning quota or billing) | Disabled for the rest of the job at the first occurrence; its tasks continue on the other models. |
| A model fails during the run (invalid JSON, 5xx) | Retried on the same model with exponential backoff, then the next model. After 3 consecutive failures a model is skipped for 30 s (circuit breaker). |
| Some records in a batch fail validation | Only those records are regenerated, up to `max_regeneration_rounds`. |
| A whole task fails | Its retry is **split into sub-batches** that several core queues generate in parallel, each on a different pool model (details below). Small tasks, or `split_failed_tasks: false`, are retried whole on the next pool model. Other tasks' batches are never redone. |
| A worker process dies (process mode) | That task fails and is retried like any other failure. The process pool is replaced (up to 2 times) so the remaining queues keep working. |
| Records fail global checks | Only the affected rows are repaired, in their own batch. |

### How a failed task is redistributed

On the next retry round, a failed task with 100 records and 4 free queues becomes, for example, 4
sub-batches of 25:

```text
task 7  (records 6001-6100)  →  7.1: 6001-6025   7.2: 6026-6050   7.3: 6051-6075   7.4: 6076-6100
```

* The task's record range never changes, so sequence IDs stay exactly where they were planned.
* Sub-batches go through the normal core queues; idle queues pick them up (work stealing helps too).
* Each sub-batch owns its uniqueness claims, so retrying one never disturbs the others.
* When all sub-batches are done, they are concatenated in order into the task's normal
  `batch_0007` file. The claims move back to the task, a new checksum is recorded, and the part
  files are deleted. Global guardrails, repairs, resume and the merger see an ordinary batch.
* A sub-batch that fails is retried on its own; finished ones are kept.
* A split manifest (`metadata/tasks/task_0007_split.json`) records the plan, so an interrupted job
  resumes only the unfinished sub-batches.
* Tunables in `quality_requirements`: `split_failed_tasks` (default `true`), `min_split_records`
  (default 50), `max_split_parts` (default 8), `max_task_retries` (default 2).

### Very large requests: split runs

Requests larger than `SDE_MAX_RECORDS_PER_JOB` (default 100,000), or than `--split-size N` /
`"split_size": N`, run as **sequential sub-jobs** that are merged at the end:

```bat
python -m app.main generate --config samples\schemas\employees.json --records 1000000
python -m app.main generate --config samples\schemas\employees.json --records 1000000 --split-size 250000
python -m app.main resume employees_20261121_040000_ab12cd        rem continues unfinished sub-jobs only
```

```text
output/Jobs_<date>/<schema>/<run_id>/
├── split_run.json             progress manifest
├── uniqueness.sqlite          shared by all sub-jobs
├── parts/<run_id>_part001/    a normal job folder per sub-job
├── final/generated_dataset.csv
└── metadata/split_summary.json
```

* Sub-jobs continue the global record index, so IDs run 1..N with no overlap. All sub-jobs share one
  uniqueness registry (each in its own namespace), so unique fields stay unique across the whole
  dataset. Each sub-job gets its own seed, so the data is not repeated.
* Category ranges are calibrated once, by the first sub-job, and reused by the others, so every
  sub-job follows the same distributions.
* Only one sub-job runs at a time, which bounds RAM and disk use. If a sub-job fails or is cancelled,
  the finished ones are kept and `resume <run_id>` continues from there.
* The final merge streams every sub-job output into one file and checks the total count, column
  order, duplicates across sub-jobs (unique fields and whole records) and references. If any check
  fails, nothing is published.
* `--split-size 0` (or `SDE_MAX_RECORDS_PER_JOB=0`) disables splitting.

### Slow models: speed test, pooled mode and run-time limit

For jobs of 1,000 records or more that need LLM text, every pool model first answers one small
request. The measured speed drives four decisions, all logged at the start of the job:

* **Models.** Models that fail the speed test are not used. On a machine without a GPU, local models
  share the same CPU cores, so running several at once makes each one slower: only the fastest local
  model is kept. Cloud models stay.
* **LLM mode** (`--llm-mode`, `"llm_mode"` or `SDE_LLM_MODE`, default `auto`):
  * `per_record`: the LLM writes the text of every record, 8 records per request.
  * `pooled`: the LLM writes a pool of answers within `SDE_POOL_BUILD_MINUTES` (20). Each answer is
    stored as a template: the values of the sample record it was written for (name, amounts, dates,
    city...) become placeholders. Every record then gets one or two templates that match its
    categories, filled with **its own** values, so the text agrees with the record and records rarely
    share a text. Composed texts are checked against the LLM output rules before use. A 10M-record job
    needs a few hundred LLM calls instead of 1.25M. The pool is saved as `llm_pool.json` and reused,
    without a new speed test, by resumed jobs and by every sub-job of a split run.
  * While records are generated, the pool keeps growing in the background up to
    `SDE_POOL_MAX_VALUES` (2,000) answers (`SDE_POOL_GROW=false` turns this off); later sub-jobs use the
    larger pool. Pooled text repeats by design, so the near-duplicate text check is off for it (exact
    repeats are still limited per batch) and low text diversity is reported but does not block delivery.
  * `auto`: `pooled` from `SDE_POOLED_MIN_RECORDS` (10,000) records when every LLM field is free
    text, else `per_record`.
* **Run-time limit.** If the projected run time exceeds `SDE_MAX_ETA_HOURS` (6, `0` = no limit), the
  job does not start. It exits with code 4, the estimate, and the options: `--llm-mode pooled`,
  `--allow-no-llm`, fewer records, a faster or cloud model, or `--force` to run anyway.
* **Timeouts during the run.** A request that times out is split in half and retried, and later
  requests ask for fewer records. A model that times out twice in a row is disabled for the job when
  another model is left. A request slower than 60 s also lowers the records per request.

**A few invalid records never fail a task.** When only a handful of a task's records are still invalid
after the regeneration rounds (at most 10, or `SDE_MAX_INVALID_FRACTION` of the task, default 0.1%),
those records get up to 10 extra rescue rounds instead of throwing away the whole batch. The dataset
keeps its exact size and contiguous IDs, because the rescued records are regenerated at the same
positions. Larger failures still fail the task, which is then split and retried as before.

The progress line shows the stage during retries (`retry round 1/2: ... 117 being retried`), the ETA
in days and hours (`4d07h`), and adds
`WARNING: no LLM answer for 2m30s` when tasks are running but no LLM call has finished for over a minute.

### No hangs, no leftovers

Every wait in the engine is bounded, and every exit path releases resources:

| Where it could block | Protection |
|---|---|
| A model call | HTTP timeout per call (`timeout` in `models.json`), bounded retries, then fallback |
| Waiting for a model slot (e.g. a worker died holding it) | at most `SDE_LLM_SLOT_TIMEOUT` seconds, and cancellation is checked every second |
| A worker process hung or crashed | the result wait polls every second. `SDE_TASK_TIMEOUT` or cancellation terminates the process, and a crashed pool is replaced. The task fails and is retried |
| The whole job making no progress | the watchdog stops the job after `SDE_STALL_TIMEOUT` seconds without a record, model answer or finished task. It reports `stalled`; `resume` continues |
| Ctrl+C on Windows | the graph runs in a helper thread and the main thread waits in 0.5 s steps, so Ctrl+C, Ctrl+Break and SIGTERM are seen at once |

On finish, failure, cancellation, Ctrl+C or SIGTERM the engine:
* shuts down worker processes: idle ones get 5 s, hung ones are terminated and then killed;
* stops the manager process and the monitor thread;
* closes HTTP clients and SQLite connections;
* asks Ollama to unload the local models the job used (`SDE_UNLOAD_MODELS=true`), so their RAM is
  freed instead of staying resident for Ollama's 5-minute keep-alive.

A second Ctrl+C aborts immediately: workers are killed and the process exits, even if a thread is
stuck in a network call. Worker processes ignore Ctrl+C themselves, so a batch is never cut off
half-written; the parent decides how to stop them. An `atexit` hook repeats the clean-up if Python
exits for any other reason.

### Cloud models without credits

A cloud model can be listed but unusable: Ollama Cloud returns HTTP 402 *"Add usage credits or
subscribe"* when the quota is exhausted. When a job needs an LLM, each cloud model in the pool is
probed with one tiny request before the run, and models without credits are excluded with a warning.
If credits run out during the run, the first 402 (or a 403/429 that mentions quota or billing)
disables that model for the rest of the job. There are no retries; its tasks continue on the other
pool models. `execution_summary.json` lists `models_disabled_during_run`.

### RAM check (`SDE_RAM_CHECK`)

Before dispatching, the engine budgets RAM for the queues plus the local model that still has to be
loaded:

* Cloud models (`-cloud` via local Ollama, Ollama Cloud, NVIDIA, OpenRouter) count as 0 MB.
* Models Ollama already has in memory (`/api/ps`) count as 0 MB, because they are already part of
  "used" RAM.
* Of the remaining local pool models, only the largest is reserved. Ollama loads models on demand and
  evicts idle ones, so a large local pool causes swapping, which is logged, rather than needing their
  combined size.

With `SDE_RAM_CHECK=warn` (default), tight RAM reduces the number of queues, down to 1, with a
warning. `strict` stops the job instead, and `off` skips the budget.

### Calibration stability

Category ranges such as salary by department are calibrated once per job with the strongest
available pool model: cloud first, then the largest local one. Calibration runs at temperature 0
with a fixed seed. Small local models can still give implausible ranges, so add a cloud model to the
pool, or pin the ranges in the schema with `"profile": {"IT": {"min": ..., "typical": ..., "max": ...}}`,
for fully reproducible values.

### Limiting retries (`SDE_MAX_RETRIES`)

Set `SDE_MAX_RETRIES` in `.env` (an integer from 0 to 20) to put one ceiling on every retry loop:

| Loop | Default | With `SDE_MAX_RETRIES=N` |
|---|---|---|
| Attempts per LLM call (per model, before falling back) | `retry.max_attempts` in `models.json` (3) | at most N + 1 |
| Regeneration rounds for rejected/missing records in a task | `max_regeneration_rounds` (4) | at most N |
| Retry rounds for failed tasks (incl. split sub-batches) | `max_task_retries` (2) | at most N |
| Global repair rounds | `max_repair_rounds` (2) | at most N |

It is a cap, not a target: a request's `quality_requirements` can still ask for fewer. The value is
validated at start-up (`SDE_MAX_RETRIES=abc` or `99` stops with a clear error). Each limit, when hit,
is logged as `retry limit reached` / `repair limit reached`, saying which setting stopped it. A job
that hits the task retry limit ends `failed` with its completed batches kept, so `resume` continues
from there. `SDE_MAX_RETRIES=0` means fail fast: one attempt everywhere.

### Error handling

Every intentional error derives from `EngineError`:

```text
EngineError
├── ConfigurationError   SchemaError   ValidationError   ResourceError   SchedulingError   CancelledError
├── TaskExecutionError ── WorkerCrashError
├── StorageError ── MergeError, RecoveryError
└── LLMError ── ModelUnavailableError, ProviderTimeoutError, RateLimitError, InvalidModelOutputError, ProviderError
```

* **Provider adapters** turn HTTP and network failures into these types: 429 → `RateLimitError` (honours
  `Retry-After`), timeouts → `ProviderTimeoutError`, 401/403/404 → `ModelUnavailableError`, 5xx or
  non-JSON → `ProviderError`. Error bodies are redacted before they reach messages or logs.
* **Storage and the uniqueness registry** wrap `OSError`, SQLite, CSV and JSON errors in `StorageError`
  with the file name. A corrupt status file means "not completed", so the batch is regenerated.
* **Workers** turn a failure into a failed task result with the cause, never a crashed queue. A
  failure while recording a failure is logged next to the original error.
* **Engine:** expected failures return `JobResult(status="failed", error=...)` and write
  `execution_summary.json`. Genuine bugs are logged with a stack trace, recorded in the job folder,
  and re-raised.
* **CLI:** clear one-line messages and exit code 2 for bad input (missing file, invalid JSON, invalid
  request, unknown job), and 130 when aborted with Ctrl+C twice.
* Secrets are masked everywhere: log filter, error messages, CLI output.

Models listed in `models.json` are only used if they appear in `model_pool`, `model_name` or
`fallback_models`. The registry is a catalogue; the request decides which models a job may use.

## Recovery and cancellation

* Every task writes its batch atomically, then a status file with the batch's
  SHA-256. A batch counts as done only if both agree.
* `resume` reloads the frozen `job_config.json` (same plan, seed, model chain
  and calibrated profiles), cleans orphaned temp files, skips verified batches,
  and regenerates only missing, failed or corrupt ones.
* Failed tasks are retried automatically (`max_task_retries`). Records flagged
  by global guardrails are repaired in place (`max_repair_rounds`).
* Ctrl+C once requests a graceful stop: in-flight tasks abort without writing
  partial files. Ctrl+C twice aborts immediately. `python -m app.main cancel <job>`
  does the same from another terminal.

## Testing

```bat
.venv\Scripts\python -m pytest                     rem unit + integration (mock LLM) + local Ollama if available
.venv\Scripts\python -m pytest -m slow             rem 100k-record end-to-end
set SDE_RUN_CLOUD_TESTS=1 && .venv\Scripts\python -m pytest -m cloud   rem Ollama Cloud (needs OLLAMA_API_KEY)
```

### Smoke test

```bat
python scripts\smoke_test.py                  rem every sample schema, 1,000 records each, mock mode (~1 min)
python scripts\smoke_test.py --records 10000  rem bigger runs
python scripts\smoke_test.py --keep           rem keep the generated files (their folder is printed)
```

It needs no Ollama. For every sample schema it runs `python -m app.main generate` with the mock
model and checks the output independently (see below). Then it checks reproducibility: the same seed
with batch sizes 300 and 1,000 must give an identical file, two per-record mock runs must match, and
patients and appointments generated with the same seed must not share values row by row.
`customer_profiles_pool.json` is skipped because it names Ollama models in the request file. The script
prints `ok`/`FAIL` per check, ends with `SMOKE TEST PASSED` or a list of problems, and exits with 1 on
failure.

### API and MCP smoke test

```bat
python scripts\api_smoke_test.py                       rem about 20 s, mock mode
python scripts\api_smoke_test.py --job-records 1000000 rem bigger background job
```

It starts the API on a free port with a temporary output folder and checks the whole flow (upload, progress,
files, range download, every format and packaging, preview, stop, resume, delete, nothing left behind), then
the MCP server over stdio, stdio forwarding to the API, and streamable HTTP. It ends with
`API SMOKE TEST PASSED`.

### Checking any output file

```bat
python scripts\check_output.py samples\schemas\healthcare_appointments.json output\Jobs_...\final\generated_dataset.csv
```

`scripts\check_output.py` (standard library only) reads the request file and the dataset (CSV, JSON or
JSONL) and reports, as errors: duplicate unique values, whole-row duplicates, empty required fields,
values outside `allowed_values`, `min`/`max`, profile ranges or regex patterns, unparseable numbers or
dates, dependent dates outside their `offset_days`, dangling references and malformed emails. Repeated
LLM text and dates after today are reported as notes only. Exit code 1 means at least one error.

| Test file | Covers |
|---|---|
| `test_planner.py` | power-of-two table, 1/2/8/16/64/128/1,024 tasks, partition invariants |
| `test_cpu_allocator.py` | the 2/4/8-core scenarios, uneven splits, resource-aware concurrency |
| `test_core_queue.py`, `test_concurrency.py` | one task per queue, parallel queues, overrides, work stealing, cancellation |
| `test_dispatcher.py` | `Send` fan-out and reducers |
| `test_generation.py` | schema parsing, hybrid generation, regex, profiles, LLM enrichment, retries, fallback, rate limits |
| `test_guardrails.py` | invalid values, duplicates, cross-task registry, semantic dedup, prompt safety, rule sandbox |
| `test_quality.py` | distributions, realism, quality score |
| `test_merger.py` | merging CSV/JSON/JSONL, integrity, atomic writes |
| `test_integration.py` | 1k and 10k end-to-end, LLM text fields, fallback, recovery, global repair, cancellation, process mode |
| `test_llm.py` | real Ollama Local (auto-skipped if unavailable), Ollama Cloud (opt-in) |
| `test_api.py` | HTTP API: public endpoint list, upload and jobs, files and download, csv/json/jsonl/excel, zip/gzip, preview, stop/resume, restart, no login header, mode vs file model settings |
| `test_mcp_server.py` | MCP tools (inline rows vs jobs, stop/delete, files), forwarding to the API, the example LLM client loop |
| `test_multi_table.py` | Multi-table schemas: validation, links (composite keys, where, match, cover, unique), per-parent children, cases, formulas, exact counts across processes and split runs, integrity checks, resume, API and MCP delivery |

## Benchmarks

```bat
.venv\Scripts\python benchmarks\run_benchmarks.py                        rem 1k / 10k / 100k, mock model
.venv\Scripts\python benchmarks\run_benchmarks.py --sizes 1000000        rem 1M records
.venv\Scripts\python benchmarks\run_benchmarks.py --modes thread process --sizes 100000
.venv\Scripts\python benchmarks\run_benchmarks.py --ollama --sizes 64    rem real qwen3:0.6b on an LLM text field
```

Results (throughput, CPU, RAM, LLM latency, rejections, quality score) are
written to `benchmarks/results/benchmark_results.{json,md}`. Sample datasets can
be regenerated with `python scripts\make_samples.py [--ollama]`.

## Project layout

```text
app/
  main.py            CLI                       engine.py      public API
  config/            settings, model registry (models.json), guardrail thresholds
  core/              exception hierarchy, cancellation token
  schemas/           request models, schema parser, dynamic Pydantic validators
  graph/             LangGraph state, reducers, planner, dispatcher, nodes, builder
  scheduler/         CPU detection, allocation, core queue, work stealing, concurrency, affinity
  workers/           generation task, batch re-validation, queue runner and process pool
  generation/        record builder, LLM enricher, profile calibrator, geo data, regex generator, table links
  multi_table.py     multi-table runs: table order, key stores, export, integrity checks, resume
  models/            provider adapters, factory, router (fallback/retry/circuit breaker), rate limiter
  guardrails/        node and global guardrails, uniqueness registry, semantic dedup, prompt safety
  quality/           realism, distributions, quality score
  storage/           atomic file manager, formats, batch writer, merger
  monitoring/        central logging, execution tracker
  utils/             retry, hashing, safe rule evaluation, helpers
  service/           shared by API and MCP: request building/validation, sync runs, background jobs
  api/               FastAPI backend (python -m app.api)
  mcp_server/        MCP server and its backends (python -m app.mcp_server)
examples/            mcp_llm_client.py: Ollama / OpenAI-compatible LLM using the MCP tools
run_mcp_server.py    MCP launcher for clients that cannot set a working folder
tests/  samples/  benchmarks/  scripts/  docs/
```

## Known limitations

* **Per-record LLM fields are bounded by inference speed.** Measured on the development machine
  (8 cores, no GPU, qwen3:0.6b): one request filling `interests` + `bio` for 8 records takes 22-46 s,
  so 3-6 s per record. Ollama serves one request at a time by default, so 100,000 such records need
  roughly 4-7 days of inference, while 500-2,000 records take minutes to a couple of hours. Purely
  deterministic schemas (like `employees.json`) run at thousands of records per second. For large
  text-heavy datasets: use a GPU, raise `OLLAMA_NUM_PARALLEL` on a GPU host, add cloud models to the
  `model_pool`, use `--llm-mode pooled` (the default from 10,000 records), or keep LLM fields to the
  records that need them. Pooled text is recombined from a limited set of answers, so it is less varied
  than per-record text. The progress line shows live LLM
  calls, average call time, requests waiting for a model slot, and an ETA, so the cost is visible
  within the first minute.
* Ollama Cloud models cannot use schema-constrained decoding; outputs are
  validated and retried instead.
* The hashing embedder catches near-identical text, not true paraphrases. Use
  `"embedder": "ollama:nomic-embed-text:v1.5"` for paraphrase-level dedup.
* In process mode, client-side rate limits are divided evenly between worker
  processes.

Data-modelling limits to plan around when you use the output for data engineering work:

* **No foreign keys across tables.** Each request file is generated on its own. A column such as
  `patient_mrn` in `healthcare_lab_results.json` is a random value with the right pattern, so it
  does not join to `healthcare_patients.json`. Generate keys for joins yourself (or post-process).
* **`reference` fields are valid but carry no meaning.** They point to an existing (earlier, if
  nullable) row of the same table, but inherit nothing from it. For example, a `follow_up_of`
  appointment usually belongs to a different patient, can be dated before the original, and can have
  `visit_type = new`. The same applies to `repeat_of_order` in orders.
* **Status columns do not drive other columns.** Fields are only linked through `depends_on`
  (profiles, date offsets) and rules that reject records. So an `under_review` claim still gets a
  `decision_date`, and a shipment can have the same origin and destination city. Add rules (they
  reject and regenerate such records) or post-process when that matters.
* **Date fields without `min`/`max` end today**, so they change with the run date (see
  [Reproducibility](#reproducibility-seed)).
* **Mock mode text and calibrated ranges are placeholders** (see
  [Mock mode vs Ollama mode](#mock-mode-vs-ollama-mode)).
* A `unique` `faker` field that collides gets a number appended (for example `"Acme Ltd 42"`) on
  regeneration.
