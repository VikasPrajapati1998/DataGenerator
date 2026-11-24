# Architecture map

Entry points: `app/main.py` (CLI), `app/engine.py` (`SyntheticDataEngine.run/resume/cancel`).
Diagrams and research notes: `docs/architecture.md`.

## Module relationships
`main` → `engine` → `graph.graph_builder` (LangGraph) → `graph.nodes` →
{`schemas`, `models`, `generation`, `scheduler`, `workers`, `guardrails`, `quality`, `storage`, `monitoring`}.
`workers.generation_worker.run_task` is the single task implementation used in both thread and
process modes. `workers.services.WorkerServices` bundles everything a task needs and is built
from `metadata/job_config.json`.

## Directory tree
```text
app/
  __init__.py                 package version
  main.py                     argparse CLI: generate, resume, cancel, plan, models
  engine.py                   SyntheticDataEngine: builds context, invokes graph, JobResult
  split_runner.py             sequential sub-jobs for very large requests + streamed cross-sub-job merge
  core/exceptions.py          EngineError hierarchy (Schema/Config/Resource/LLM/Storage/...)
  core/cancellation.py        CancellationToken (thread event + CANCEL marker file)
  config/settings.py          Settings from env/.env (paths, Ollama host, secrets, defaults)
  config/model_registry.py    ModelConfig + ModelRegistry (resolve, on-the-fly models)
  config/models.json          shipped model definitions (Ollama local/cloud, Beam, Nemotron, mock)
  config/guardrail_config.py  GuardrailConfig thresholds (overridable per request)
  schemas/input_schema.py     GenerationRequest, DatasetSchema, FieldSpec, RuleSpec, ProfileRange...
  schemas/schema_parser.py    shorthand/rich → DatasetSchema, validation, dependency order
  schemas/validation_schema.py dynamic Pydantic model + RecordValidator (rules)
  graph/state.py              GraphState TypedDict, CoreWorkerInput, EngineContext
  graph/reducers.py           concat / merge_dicts reducers, latest_by_task
  graph/planner.py            build_plan, generation_items, repair_items
  graph/dispatcher.py         prepare_dispatch (queues), dispatch_core_queues (Send fan-out)
  graph/nodes.py              the 14 graph nodes
  graph/graph_builder.py      StateGraph wiring and routing
  scheduler/cpu_detector.py   cores/RAM/GPU detection (psutil, nvidia-smi)
  scheduler/task_allocator.py calculate_dynamic_nodes, plan_tasks, allocate_to_cores
  scheduler/core_queue.py     CoreQueue (one-active-task invariant)
  scheduler/work_stealing.py  CoreQueueScheduler (acquire/release/run_queue, stealing)
  scheduler/concurrency_manager.py plan_concurrency (queues vs LLM concurrency)
  scheduler/affinity.py       optional per-thread CPU pinning (Windows/Linux)
  workers/services.py         JobSpec, WorkerServices (per-process service bundle)
  workers/generation_worker.py WorkItem, run_task (generate/repair, regeneration loop, persist)
  workers/validation_worker.py streaming re-validation of one batch for global checks
  workers/worker_manager.py   TaskExecutor (thread/process pool), run_core_queue
  generation/record_builder.py deterministic fields (sequence, Faker, distributions, profiles, geo, refs)
  generation/llm_enricher.py  batched LLM fill of `llm` fields with output guardrails; halves requests on timeout
  generation/preflight.py     LLM speed test, model choice (CPU-only: one local model), mode + ETA check
  generation/llm_pool.py      pooled mode: templated llm_pool.json, PooledFiller (valid texts per record), PoolGrower
  generation/profile_calibrator.py one-off LLM calibration of category ranges
  generation/geo_data.py      curated city/state tables (en_IN, en_US, en_GB)
  generation/regex_generator.py strings matching a regex subset
  models/base_adapter.py      BaseLLMAdapter, HTTPAdapterMixin, extract_json
  models/ollama_local.py      /api/chat, /api/tags, /api/ps, /api/embed
  models/ollama_cloud.py      ollama.com with Bearer key
  models/openai_compatible.py /chat/completions (NVIDIA NIM, OpenRouter, vLLM...)
  models/mock_adapter.py      deterministic offline adapter with failure injection
  models/model_factory.py     provider → adapter registry (register_provider)
  models/model_router.py      fallback chain, retries, circuit breaker, LLMUsage
  models/rate_limiter.py      TokenBucket, Limiter protocol
  guardrails/node_guardrails.py per-batch pipeline + BatchGuardState
  guardrails/global_guardrails.py streaming dataset checks + GlobalReport
  guardrails/uniqueness.py    SQLite UniquenessRegistry (claims per task)
  guardrails/semantic_deduplication.py hashing/Ollama embedders, near-dup search, SimHash/LSH
  guardrails/prompt_safety.py instruction sanitising, output text checks
  quality/realism_validator.py profile ranges, geo, email, age vs DOB, text plausibility
  quality/distribution_validator.py streaming stats + statistical checks
  quality/quality_scorer.py   weighted 0-100 score
  storage/file_manager.py     JobPaths layout, atomic writer, disk check, temp cleanup
  storage/formats.py          CSV/JSON/JSONL read/write/count
  storage/batch_writer.py     batch, rejected report, task status, verified_completion
  storage/merger.py           deterministic merge with integrity/uniqueness checks
  monitoring/logger.py        central logging config + secret redaction
  monitoring/execution_tracker.py counters, resource sampling, progress, summary
  utils/retry.py              RetryPolicy, retry_call (exponential backoff + jitter)
  utils/hashing.py            stable value/record hashes, seeds, file SHA-256
  utils/safe_eval.py          sandboxed rule expressions
  utils/helpers.py            job ids, chunking, formatting
tests/                        pytest suite (see README "Testing")
samples/schemas/              employees(.json/_mock), ecommerce_orders, customer_profiles
samples/outputs/              generated by scripts/make_samples.py
benchmarks/run_benchmarks.py  benchmark matrix → benchmarks/results/
scripts/make_samples.py       regenerates sample outputs
docs/architecture.md          diagrams, design decisions, research notes
output/                       runtime: Jobs_<YYYY-MM-DD>/<schema_name>/<job_id>/ (legacy: jobs/job_<id>/)
logs/                         log files (runtime)
```
