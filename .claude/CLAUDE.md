# Project knowledge: Synthetic Data Generation Engine

## Conventions
* Python 3.11+ (developed on 3.12, Windows 11). Type hints everywhere, Google-style docstrings.
* Package `app/`; run as modules from the project root (`python -m app.main ...`).
* One central logger: `app.monitoring.logger.get_logger(__name__)`. Never `print` in library code
  (the CLI and scripts may print results). Logs go to `logs/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS.log`;
  secrets are redacted by a logging filter.
* All intentional errors derive from `app.core.exceptions.EngineError`. Chain with `raise ... from exc`.
* All file outputs use `storage.file_manager.atomic_writer` / `write_json_atomic`.
* Settings come from env / `.env` via `config.settings.get_settings()`. Secrets are `SecretStr`, and
  the registry stores only the *name* of the env var (`auth_env`).

## Architectural decisions
* **Records never enter LangGraph state.** State holds task specs, allocations, paths, statuses
  and reports. Runtime objects (scheduler, executor, tracker, services) go in the LangGraph
  runtime context (`EngineContext`).
* **One `Send` per logical core queue** to the single reusable `core_worker` node. The
  `CoreQueueScheduler` enforces one active task per queue and optional work stealing. Do not
  dispatch one Send per task: that loses per-queue sequencing.
* **Queues ≠ LLM concurrency.** Queue count is CPU/RAM bound; LLM calls are limited separately
  by per-model semaphores (`OLLAMA_NUM_PARALLEL`, cloud RPM).
* **Hybrid generation.** Deterministic generators first (`RecordBuilder`). The LLM only fills
  `llm` fields (batched, 8 per request) and calibrates `profile` ranges once per job (stored in
  `job_config.json`).
* **Uniqueness.** Deterministic ID partitioning plus a SQLite claim registry
  (`metadata/uniqueness.sqlite`) shared by threads and processes. Claims are tagged by task, and a
  retried task releases its own claims first (idempotency).
* **Model pool.** `model_pool` (or `SDE_MODEL_POOL`) holds the models; `graph.planner.assign_models`
  stamps one `model_key` per task (round-robin or smooth weighted round-robin) and stores it in
  `job_config.json`. `WorkerServices.router_for(key)` puts that model first and the rest of the chain
  as fallbacks. Routers share token buckets and per-model semaphores (`ConcurrencyPlan.llm_slots`).
* **Failed-task redistribution.** `collect_results` splits a failed task into sub-batches
  (`workers/task_splitter.py`: `plan_split` → manifest → `part_items`). A `WorkItem.part > 0` writes
  `batch_NNNN_pPP` and owns claims as `task*1000+part`. `consolidate` merges finished parts in order
  into `batch_NNNN`, then `registry.adopt_parts`, a new status and checksum, and deletes the part files.
  Downstream code only ever sees whole batches. `latest_by_task` is keyed by `(task_id, part)`.
* **Error handling.** Catch the narrowest type, wrap it into the `EngineError` hierarchy with context,
  `redact()` anything that may echo provider responses, and never let one task crash a queue or the job.
* **Whole-record dedup** uses `DatasetSchema.record_fields` (non-`dedup_exclude` fields). `check_record_space`
  (validate node) sets `record_dedup=False` when those fields allow fewer than 2x records combinations; then
  `record_fields` is empty and guardrails, merger and the split merge skip the whole-record key.
* **Pooled text diversity** (`distinct_text_ratio`) is reported only: excluded from blocking checks and from the
  quality score when `services.pool_filler` is set.
* **Semantic dedup only on free-text (`llm`) fields.** Structured near-duplicates are legitimate.
* **Recovery manifest** = `metadata/tasks/task_NNNN.json` with the batch SHA-256. A batch is
  "done" only if the status and the checksum agree (`verified_completion`).
* **Output layout** (user preference): `output/Jobs_<YYYY-MM-DD>/<schema_name>/<job_id>/`, with
  job_id `<schema_name>_<YYYYMMDD_HHMMSS>_<hex6>`. The schema name comes from `schema_name`, then the
  CLI config file stem, then the rich schema's `entity`, then `dataset`. Look jobs up with
  `storage.file_manager.find_job_dir`, which also finds the legacy `jobs/job_<id>` folders.
* **Split runs** (`app/split_runner.py`): requests above `split_size` / `SDE_MAX_RECORDS_PER_JOB`
  become sequential sub-jobs (`GenerationRequest.sub_job`: global `record_offset`, shared
  `registry_path` with namespace = part, profiles from part 1, per-part seed). A `split_run.json`
  manifest records progress. The final streamed merge checks cross-sub-job duplicates and references.
* **Limit failover** (`ModelRouter`): when another healthy model exists, `RateLimitError`, `ProviderTimeoutError`
  and `ContextLengthError` are not retried on the same model. Rate limits and timeouts put the model in a shared
  cool-down (`WorkerServices.cooldowns`); context overflow only moves that request. The last remaining model is
  retried normally.
* **Unique regex capacity**: `schema_parser.check_unique_capacity` rejects patterns smaller than 2x the record
  count and warns above ~100 expected collisions. Sample patterns are sized for 1M records (tested).
* **Permanent model errors** (`ModelUnavailableError`, incl. `QuotaExceededError` for HTTP 402) are
  never retried. The router disables the model for the job via a set shared by all routers.
  `check_models` probes cloud models with one tiny call when an LLM is needed.
* **No unbounded waits.** Slot acquisition polls with `llm_slot_timeout`. Process-pool results poll every 1 s
  with `task_timeout` and cancellation, then `_terminate_pool`. The tracker watchdog (`stall_timeout`) cancels
  stalled jobs. `engine._invoke` runs the graph in a daemon thread and joins in 0.5 s steps (signals stay
  responsive on Windows). `EngineContext.close(force)` releases everything, including unloading Ollama models.
  The CLI adds SIGTERM/SIGBREAK handlers, `atexit`, and `os._exit(130)` on hard abort.
* **Runtime config over schema config** (user preference): sample schemas hold only the data definition.
  Batch size comes from `planner.auto_batch_size` unless the request or `SDE_BATCH_SIZE` sets it. Models:
  request `model_pool` > explicit `model_name` > `SDE_MODEL_POOL` > default. `check_models` skips all checks
  when no LLM is needed, uses one `/api/tags` call for all local models, probes the others in parallel,
  and caches results across the sub-jobs of a split run (`engine.availability_cache`).
* **LLM preflight** (`generation/preflight.py`, called from `nodes._prepare_llm` in `allocate_tasks`):
  for jobs of 1,000+ records with LLM fields, one 1-record request per model. Failed models are dropped;
  without a GPU only the fastest local model is kept. `plan_llm` picks `per_record` or `pooled`
  (`SDE_LLM_MODE`, auto from `SDE_POOLED_MIN_RECORDS` when all LLM fields are strings) and raises
  `FeasibilityError` (exit 4) above `SDE_MAX_ETA_HOURS` unless `force`. The plan is cached in
  `availability_cache["llm_plan"]` for split runs.
* **Pooled mode** (`generation/llm_pool.py`): builds `llm_pool.json` once (job metadata dir, or the split-run
  root; format 2 stores the plan too, so resume and later sub-jobs skip the speed test). Answers are stored as
  templates (`templatize`: the sample record's own values -> `{field[:fmt]}`), `relabel` files them under the
  category they name, and `PooledFiller` renders them with each record's values and checks every composed text
  (`check_generated_text`, min words) before use. `WorkerServices.use_pool` turns off the embedding near-dup
  check for pooled text, and global guardrails treat `distinct_text_ratio` as non-blocking then. `PoolGrower`
  adds answers in the background and saves them in `services.close`.
* **Rescue rounds** (`generation_worker._generate_rows`): when at most `max(min_rescue_records,
  max_invalid_fraction x task)` records are still invalid after the regeneration rounds, they get
  `rescue_rounds` more rounds (pooled filler with `widen=True`) instead of failing the task. Same positions,
  so counts and IDs stay exact.
* **Timeouts shrink requests**: the enricher halves a group on `ProviderTimeoutError` (and after a call
  slower than 60 s) and lowers `chunk_size`. The router disables a model after 2 consecutive timeouts when
  others remain; when every model timed out it raises `ProviderTimeoutError`, not `ModelUnavailableError`.
* **Merging is pure Python** (`storage/merger.py`). It refuses to merge rather than drop records.
* Hashes must be process-stable (`utils/hashing.py`, BLAKE2b). Never use built-in `hash()` for
  anything stored or shared across processes.

## Workflows
* Add a provider: subclass `BaseLLMAdapter` (+ `HTTPAdapterMixin`), register it with
  `@register_provider`, add entries to `app/config/models.json`. No graph changes.
* Add a generator: extend `GeneratorName` in `schemas/input_schema.py`, implement it in
  `RecordBuilder._generate`, validate it in `schema_parser._validate_field`, and add any
  realism checks.
* Tests use the `mock` provider (`MockLLMAdapter.behaviors[...]` injects failures). Real-Ollama
  tests auto-skip when the model is absent. Cloud tests need `SDE_RUN_CLOUD_TESTS=1`.

## Facts worth remembering
* LangGraph 1.2.x: `from langgraph.types import Send`, `from langgraph.runtime import Runtime`,
  `StateGraph(State, context_schema=Ctx)`, `graph.invoke(..., context=ctx, config={"max_concurrency": N})`.
* Ollama Cloud models do not support schema-constrained `format`, so the registry uses `json_mode`.
* "Beam" = Reflection AI Beam 1.0 (`beam:cloud` on Ollama). "Iceman" could not be identified (Nov 2026).
