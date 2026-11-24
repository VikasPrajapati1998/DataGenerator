# Setup

## Requirements
* Python 3.11+ (tested target: 3.12 on Windows 11; Linux is supported).
* Optional: Ollama 0.4+ running locally (`ollama serve`) with a model, e.g. `ollama pull qwen3:0.6b`.
  Optional embedding model for neural semantic dedup: `ollama pull nomic-embed-text:v1.5`.
* Optional: API keys for Ollama Cloud (`OLLAMA_API_KEY`), NVIDIA NIM (`NVIDIA_API_KEY`), OpenRouter (`OPENROUTER_API_KEY`).
* Package manager: pip with a project-local virtual environment (`.venv`).

## Install (Windows cmd)
```bat
cd D:\Study\DataGenerator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && cp .env.example .env`.

## Environment variables (see `.env.example`)
| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | local Ollama server |
| `OLLAMA_NUM_PARALLEL` | 1 | parallel requests the server serves per model (keep in sync with the server) |
| `OLLAMA_API_KEY`, `OLLAMA_CLOUD_HOST` | unset, `https://ollama.com` | direct Ollama Cloud |
| `NVIDIA_API_KEY`, `OPENROUTER_API_KEY` | unset | OpenAI-compatible providers |
| `SDE_DEFAULT_MODEL` | `ollama_local/qwen3:0.6b` | default registry key |
| `SDE_MODEL_POOL` | unset | comma-separated registry keys spread over tasks (one model per task) |
| `SDE_BATCH_SIZE` | 1000 | default target batch size |
| `SDE_MAX_RECORDS_PER_JOB` | 100000 | larger requests become sequential sub-jobs merged at the end (0 = never split) |
| `SDE_STALL_TIMEOUT` | 900 | watchdog: stop a job with no progress for this many seconds (0 = off) |
| `SDE_TASK_TIMEOUT` | 7200 | terminate a worker process whose task runs longer (0 = off) |
| `SDE_LLM_SLOT_TIMEOUT` | 900 | max wait for a free model request slot |
| `SDE_UNLOAD_MODELS` | true | unload local Ollama models when a job ends |
| `SDE_RAM_CHECK` | `warn` | `warn` = fewer queues when RAM is tight, `strict` = fail, `off` = skip |
| `SDE_MAX_RETRIES` | unset | global retry cap 0-20 (LLM call retries, regeneration, task retry and repair rounds) |
| `SDE_MAX_WORKERS` | unset | global cap on logical queues |
| `SDE_OUTPUT_DIR`, `SDE_LOG_DIR`, `SDE_LOG_LEVEL` | `output`, `logs`, `INFO` | paths and verbosity |
| `SDE_MODEL_REGISTRY` | `app/config/models.json` | alternative registry file |
| `SDE_TEST_OLLAMA_MODEL`, `SDE_RUN_CLOUD_TESTS` | `qwen3:0.6b`, unset | test switches |

## Run
```bat
python -m app.main models check
python -m app.main plan --records 100000 --batch-size 1000
python -m app.main generate --config samples\schemas\employees_mock.json --records 10000
python -m app.main generate --config samples\schemas\employees.json --records 5000
python -m app.main resume <job_id>
```

## Test and benchmark
```bat
python -m pytest
python -m pytest -m slow
python benchmarks\run_benchmarks.py
python scripts\make_samples.py --ollama
```

No build step and no deployment prerequisites beyond the above. Outputs go to `output\jobs\`
and logs to `logs\`.
