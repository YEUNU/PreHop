# Execution and Measurement Protocol

Use this guide to launch targets, preserve existing evidence, and report
comparable costs. Runtime setup belongs in
[RUNTIME_REQUIREMENTS](RUNTIME_REQUIREMENTS.md); completed numbers belong in
[RESULTS](RESULTS.md). Keep live counters, queue snapshots, and temporary
investigations in generated campaign artifacts outside `docs`.

## Completion and continuation

A finished benchmark receives a completion receipt from
`scripts/record_paper_completion.py`, without final paper-policy validation.
The legacy verifier entry forwards to that recorder. Runtime readiness,
index-reuse, corpus integrity, and checkpoint checks remain separate.
An index-only completion is not a completed query benchmark.

Source edits alone do not block index or rolling dispatch. Loaded Python code
may remain unchanged in an already-running process; new segments record their
actual launch provenance. Preserve old results and settings rather than
relabelling them with current defaults. Paper generation omits the LLM seed;
evaluation/sampling seed 42 is separate from generation.

## Execution contract

`core/execution_profile.py` binds the selected transport and producer settings
to provenance. Set `RAG_EXECUTION_PROFILE` to an absolute JSON path before Python
starts. Version 3 uses one combined generation/embedding request pool;
legacy versions retain their historical per-kind limits.

`core/inference_queue.py` limits upstream requests across child processes,
forwards request bodies and streaming responses, and does not add retries.
The default measurement limit is one target. `RAG_MEASUREMENT_MAX_TARGETS`
accepts 1–4; multiple targets require the owned queue and imply shared-resource
measurement. Local ownership locks cannot establish exclusivity against remote
clients or other users.

### Shared indexing profile

`configs/execution_profiles/index-shared-120.json` contains:

| Field | Value |
|---|---:|
| `inference_concurrency` | 120 |
| `embedding_batch_size` | 16 |
| `benchmark_concurrency` | 8 |
| `index_document_concurrency` | 60 |
| `index_prefetch_documents` | 120 |
| `prehop_chunk_concurrency` | 8 |
| `lightrag_document_concurrency` | 32 |
| `youtu_document_concurrency` | 60, inactive compatibility field |

Generation and embedding compete for the same 120 slots, with no per-kind
reservation. Queue snapshots expose the measured aggregate peak plus per-kind
observations; do not add peaks measured at different times. These settings
specify request bounds, not GPU capacity, sustained utilization, or an optimum.
Native producers and local graph/model work can limit demand.

Without an execution profile, registry transport defaults are generation
concurrency 30, embedding batch/concurrency 16/1, and benchmark concurrency one.
Changing the profile requires compatible new timing evidence. A small pilot
can check integration and request throughput but does not establish full-run
quality, batch invariance, or optimal query concurrency.

## Foreground target

Prepare `.env`, full corpus/query manifests, and the required runtimes first.
Use a fresh target ID and the selected main interpreter:

```bash
export PYTHON_BIN=/absolute/path/to/prepared/main-venv/bin/python
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/prepared/main-venv

"$PYTHON_BIN" scripts/run_with_inference_queue.py \
  --profile "$PWD/configs/execution_profiles/index-shared-120.json" \
  --metrics /absolute/path/fresh-queue-metrics.json \
  -- bash scripts/run_paper_target.sh multihoprag prehop fresh-target-id
```

The wrapper owns the queue until its foreground child exits. Do not put a
background launcher inside it. `run_paper_target.sh ... --check` performs
readiness checks without launching the target. The real target uses fresh
run-scoped storage and disables in-repository generation/embedding caches.
It never invokes the global graph clear operation.

A compatible completed index can be reused; a compatible partial deterministic
benchmark can resume missing queries. Completed benchmark status is recorded
without another final policy check. Existing incompatible artifacts are
preserved and require a fresh run ID. Source and query provenance remain bound
to the actual execution.

## Index-only batch

With a prepared environment and a selected profile:

```bash
export RAG_EXECUTION_PROFILE="$PWD/configs/execution_profiles/index-shared-120.json"
"$PYTHON_BIN" scripts/index_matrix.py launch indexing-01
```

The supervisor owns a detached session and writes
`data/results/<campaign>/index-supervisor/{plan,status,queue-metrics}.json`.
It performs smoke builds and full indexes for the registry's seven methods on
two datasets: 14 default targets. Smoke failure isolates that target; unrelated
targets continue. It reports completion only when every planned full index
succeeds and does not itself run full query benchmarks.

Full index IDs use `<campaign>-index-<dataset>-<strategy>`. Failed attempts and
original phase costs remain recorded. The index-only runner does not implicitly
retry failed attempts; a rolling controller may enqueue explicit retries with
fresh IDs. A later benchmark can use a version-2 index-reuse link to a complete
index-supervisor receipt without repeating index construction.

## Persistent ownership and recovery

Read generated status and the named stage logs to identify the owning process
and actual target outcome. A supervisor heartbeat proves liveness, not source
completion. Stage-boundary queue snapshots are not live utilization readings.
Low inference activity can reflect parsing, local models, graph writes, or
other dependent stages; it does not by itself justify increasing concurrency.

Supervisors use identity-bound process ownership and retain status after the
initiating shell exits. Cleanup targets their own verified descendants with
TERM and a bounded wait, without KILL escalation. Surviving owned processes
block restart. Reboot recovery and automatic chat notifications are not claimed.

`paper_campaign.py` and `run_paper_matrix.sh` also retain a separate legacy
full-matrix evidence-ledger workflow. Its explicit prerequisites are distinct
from the index-only and rolling dispatcher, and its legacy stage names do not
change the registry's target count. It is not an automatic final validation step.

### Cancelling queued work

Supported clients attach `X-Prehop-Run-ID`. An authenticated
`POST /v1/queue-cancel` for that ID removes waiting requests and rejects later
requests for it. Forwarded requests keep their slot until the upstream response
or timeout; cancellation does not prove the upstream model stopped inference.
Use a new run ID when restarting. If queues are layered, their counters overlap
and must not be summed as independent work.

## Final tables and measurement definitions

Report datasets separately and state each metric's population.

| Dataset | Quality population and measures | Cost normalization |
|---|---|---|
| MultiHop-RAG | 2,255 non-null queries: official Hits@4/10, MRR@10, MAP@10; all 2,556 queries: official QA Accuracy | 609 source documents; 2,556 queries |
| MuSiQue-Ans dev | 2,417 queries: official answer EM/F1 scoring; global paragraph Support P/R/F1 as a task adaptation | 21,099 prepared source files; 2,417 queries |

The 301 MultiHop-RAG null questions are excluded from successful retrieval
rows. Terminal failures have zero quality scores; failed null rows can alter
the failure-inclusive retrieval denominator and must be reported separately.
Global MuSiQue support must not be labelled as the official question-local
candidate protocol. AllFacts and literal fact recall are additional diagnostics,
not official leaderboard metrics.

- **Index wall time:** original successful index-pipeline wall seconds, including
  waiting, retries, and native work. `amortized_indexing_cost` divides this by
  manifest source count. Post-timer capacity/reporting work is separate.
- **Query batch wall time:** dispatch of the full batch through the last answer
  or terminal failure. `amortized_query_cost` divides this by full query count.
  It includes delays from queues, retries, and interleaved checkpoint work but
  excludes initialization and trailing reports. It is inverse throughput.
- **Query latency:** individual query response time, including applicable
  worker/queue waits and synthesis. It is not batch wall time divided by queries.
- **Benchmark segment wall time:** cumulative checkpointed execution segments,
  including setup and checkpoint work. Resume retains each segment once.
  It is distinct from the continuous query-batch timer.
- **Reuse/clone preparation:** separate from original index construction.
  Copying body vectors or a native index does not establish cold indexing cost.
- **Storage:** Neo4j values are logical-payload estimates; file-backed values
  are physical artifact bytes. Label the method and do not rank them as equal
  physical database-size measurements.

Partial, integrity-failed, or resumed query batches do not supply uninterrupted
throughput. Fully executed batches include terminal failures. Missing native
usage and monetary costs remain unavailable. Preserve failed-attempt costs
separately from successful index costs. Prehop tracing contributes to measured
phase wall time; trace files are excluded from retrieval-index storage size.

Paired quality analyses require matched query IDs and explicit method controls.
Query bootstrap intervals do not capture generation/index-build variability or
selection bias. Latency comparisons require a declared common serving/load
window. The [ablation specification](PAPER_ABLATION_DESIGN.md) defines the two
representation comparisons and their unequal initial search budgets.
