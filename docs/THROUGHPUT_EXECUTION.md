# Execution and Measurement Protocol

Use this guide to launch targets, preserve existing evidence, and report
comparable costs. Runtime setup belongs in
[RUNTIME_REQUIREMENTS](RUNTIME_REQUIREMENTS.md). Completed benchmark results
retain their source artifacts. Keep live counters and temporary
investigations in generated campaign artifacts outside
`docs`.

HotpotQA (HippoRAG corpus) preparation and its sentence-output evaluation policy are
documented in [HOTPOTQA](HOTPOTQA.md).

## Completion and continuation

`scripts/run_paper_target.sh` calls `scripts/record_paper_completion.py` after
benchmark execution. The recorder accepts result states `completed_unadmitted`,
`completed` and `admitted`; other states return an incomplete result. It emits
the legacy receipt label `admitted` with `verification=disabled_by_user`, and
preserves an existing receipt.
This is a status-based execution receipt. It does not recount queries, rescore
answers, validate configuration or certify publication eligibility. To report
completed evidence, inspect the owning result's population, terminal failures,
index source and measurement scope. Dispatch and completion have no additional
runtime, index-reuse, corpus-integrity, configuration or checkpoint validation
gates. JSON decoding and actual execution errors remain observable. An
index-only completion is not a completed query benchmark.

LLM judging is disabled in the paper configuration. If explicitly enabled for
a separate run, judging is synchronous. `RAG_JUDGE_BATCH=true` is rejected when
judging is enabled; asynchronous Batch submission and reconciliation are not
supported. Unjudged fields remain unavailable, not quality scores.

Source edits alone do not block index or rolling dispatch. Loaded Python code
may remain unchanged in an already-running process; new segments record their
actual launch provenance. Preserve old results and settings rather than
relabelling them with current defaults. Paper generation omits the LLM seed;
evaluation/sampling seed 42 is separate from generation.

## Execution contract

Clients send generation and embedding requests directly to the configured
LiteLLM gateway. `core/execution_profile.py` records the selected operational
settings and their content hash. Set `RAG_EXECUTION_PROFILE` to an absolute JSON
path before starting Python. The profile sets client and producer limits;
there is no local HTTP proxy or cross-process request semaphore.

Run dataset/system targets sequentially. Within each target, benchmark workers
may process up to eight questions concurrently. A client generation limit is
per client, not a gateway-wide ceiling; starting independent jobs or nested
native workers can increase aggregate requests. Request limits do not describe
hardware capacity or guarantee that an upstream service will accept every call.

### Direct-request profile

`configs/execution_profiles/direct-8.json` contains:

| Field | Value |
|---|---:|
| `generation_concurrency` | 8 |
| `embedding_batch_size` | 16 |
| `embedding_concurrency` | 1 |
| `benchmark_concurrency` | 8 |
| `index_document_concurrency` | 8 |
| `index_prefetch_documents` | 16 |
| `prehop_chunk_concurrency` | 8 |
| `lightrag_document_concurrency` | 8 |

Without a profile, registry defaults remain generation concurrency 30,
embedding batch/concurrency 16/1, and benchmark concurrency one. An explicitly
selected profile overrides matching environment values. Preserve the effective
settings with measurements; changing concurrency does not update prior timing
results.

### Configuration ownership and precedence

`core/strategy_registry.py` declares static method and request defaults. Core
retrieval settings and canonical index/query descriptions read those same method
defaults. The registry never reads the selected profile during import.
`core/execution_profile.py::resolved_execution_environment` combines operational
defaults, caller values, and the selected profile. `core/inference_transport.py`
then resolves one effective request contract, including embedding controls, for
clients, native child processes, probes, and recorded provenance.

All Python runners load `.env` through `scripts/runner_environment.py`; shell
runners use `scripts/lib.sh` to load the file and select Python, then call the
same resolver. Priority is registry defaults < `.env` < exported environment <
the profile's matching throughput fields. `RAG_SKIP_PROJECT_ENV=true` skips only
the file, not profile resolution. Isolated external workers can additionally
override embedding batch/concurrency/retries with their existing
`RAG_<STRATEGY>_EMBEDDING_*` variables; these take precedence over global controls
and are recorded in the effective contract.

`core/paper_policy.py::configure_target_environment` owns paper run identity,
namespace, output/cache paths, generation seed omission, and method-native query
instruction. Shell and Python target launchers call this function. It does not
validate index compatibility. Read-only configuration inspection uses
`resolved_target_environment` and passes the copied settings through policy and
runtime resolution without temporarily modifying `os.environ`.
`configs/paper_runtime_requirements.json` remains
the installation contract for separate native runtimes and local models; it is
not a second set of request defaults.

## Foreground target

Prepare `.env`, corpus/query manifests, and the required runtimes first. Use a
fresh target ID and the selected main interpreter:

```bash
export PYTHON_BIN=/absolute/path/to/prepared/main-venv/bin/python
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/prepared/main-venv
export RAG_EXECUTION_PROFILE="$PWD/configs/execution_profiles/direct-8.json"
bash scripts/run_paper_target.sh multihoprag prehop fresh-target-id
```

The command indexes and benchmarks one target in the foreground.
`run_paper_target.sh ... --check` resolves launch settings without running the
target; it does not verify runtime readiness. The target uses run-scoped storage
and disables in-repository generation/embedding caches. It never invokes the
global graph clear operation.

The launcher skips indexing when the run's index-statistics file exists and
resumes missing query IDs when its partial result exists. These file-presence
checks do not establish semantic compatibility. Use fresh IDs for changed
inputs or settings and retain provenance. See
[checkpoint behavior](ARCHITECTURE.md#evaluation-output-contract).

## Index-only batch

With a prepared environment and a selected profile:

```bash
export RAG_EXECUTION_PROFILE="$PWD/configs/execution_profiles/direct-8.json"
"$PYTHON_BIN" scripts/index_matrix.py launch indexing-01
```

The supervisor owns a detached session and writes
`data/results/<campaign>/index-supervisor/{plan,status}.json`.
It performs smoke builds and full indexes sequentially for the registry's seven
methods on two datasets: 14 default targets. The link-experiment controller below
is a separate workflow. Smoke failure isolates that target; unrelated targets
continue. It reports completion only when every planned full index
succeeds and does not itself run full query benchmarks.

Full index IDs use `<campaign>-index-<dataset>-<strategy>`. Failed attempts and
original phase costs remain recorded. The index-only runner does not implicitly
retry failed attempts; a controller may schedule explicit retries with
fresh IDs. A later benchmark can use a version-2 index-reuse link to a complete
index-supervisor receipt without repeating index construction.

## Persistent ownership and recovery

Read generated status and the named stage logs to identify the owning process
and actual target outcome. A supervisor heartbeat proves liveness, not source
completion.
Low inference activity can reflect parsing, local models, graph writes, or
other dependent stages; it does not by itself justify increasing concurrency.

Supervisors use identity-bound process ownership and retain status after the
initiating shell exits. Cleanup targets their own verified descendants with
TERM and a bounded wait, without KILL escalation. Surviving owned processes
block restart. Reboot recovery and automatic chat notifications are not claimed.

`paper_campaign.py` and `run_paper_matrix.sh` also retain a separate legacy
full-matrix evidence-ledger workflow. Its ledger records executed stages without
prerequisite approval checks. The `resume_continuation` stage exercises interruption
and continuation of a saved checkpoint; it does not test configuration rejection.
Legacy stage names in historical receipts do not change the registry's target
count. This workflow is not an automatic final validation step.

## Final tables and measurement definitions

Report datasets separately and state each metric's population.

| Dataset | Quality population and measures | Cost normalization |
|---|---|---|
| MultiHop-RAG | 2,255 non-null queries: official Hits@4/10, MRR@10, MAP@10; all 2,556 queries: official QA Accuracy | 609 source documents; 2,556 queries |
| HotpotQA (HippoRAG corpus) | All 1,000 released rows (944 original questions): official answer, supporting-fact and joint EM/F1/precision/recall | 9,221 released passages; row-weighted metrics and original-question cluster intervals |

The 301 MultiHop-RAG null questions are excluded from successful retrieval
rows. Terminal failures have zero quality scores; failed null rows can alter
the failure-inclusive retrieval denominator and must be reported separately.
HotpotQA supporting-fact scores use a shared, gold-independent projection of
complete returned corpus sentences to original title/index pairs. Report mapping
coverage; distinguish these predictions from native sentence selection and from
additional passage-coverage diagnostics. Saved-output research comparisons
use the four MultiHop-RAG rank measures plus distinct-gold Recall@10 on both
datasets; the HotpotQA adaptation is defined in
[HOTPOTQA](HOTPOTQA.md#common-passage-retrieval-metrics). These offline measures
do not replace native benchmark output fields or become official HotpotQA metrics.

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
window. [Representation comparisons](REPRODUCING.md#compare-link-and-search-representations)
describe their controls and unequal initial search budgets.

## Controlled link experiments

`scripts/link_experiment_campaign.py` executes the explicit dependency graph
from `scripts/plan_link_experiments.py`. New plans run one task at a time. A ready
HopRAG phase takes priority when the slot is free. Plans can explicitly set a
larger `max_active`, in which case
at most one HopRAG task runs across datasets.
A plan with `max_active: 1` and `execution_filter: "hoprag"` runs only
one HopRAG task at a time and leaves other models pending.
`--resume` preserves recorded completed/failed tasks and adopts running tasks
from the saved state instead of restarting the task graph.
Exclusive timing tasks wait for an idle campaign, while ready non-timing work
can continue. Primary benchmark tasks follow their completed index. Requests go
directly to the configured gateway; connection replays process one query at a time.
The connection estimator does not dispatch online end-to-end benchmarks. Reference
traces supply matched starting passages for connection-only measurements;
offline analysis adds their per-query deltas to existing measured full-query
latencies. Estimated latency is separate from measured latency and cannot
replace measured batch wall time or throughput.
See [connection timing](REPRODUCING.md#measure-stored-versus-online-connection-processing)
for commands and measurement scope, and
[HOTPOTQA](HOTPOTQA.md) for the active corpus. The JSON timing-store file points
to Neo4j HOP_TIMING relationships and contains no destination table.

The controller starts only tasks whose dependencies have completed. Ready
ablation benchmarks have dispatch priority over preparation and indexing;
primary benchmarks follow their own index. A failed task is recorded as
`failed`, and dependent tasks become `dependency_failed`; the controller does
not silently retry or substitute results. Adopted processes occupy the same
slots as newly launched jobs. Timing exclusivity applies to this campaign,
not to unrelated clients of the LiteLLM server.

Use the generated campaign `status.json` for task state and the owning result
or log for progress. Estimate remaining time from observed completions over a
recent interval. Report the measured phase: HopRAG edge-scoring progress does
not include later persistence or query benchmarking. Keep ETA and partial
scores outside this document and the final result tables. A pending task has
no measured duration yet, so a campaign-wide completion time may be unavailable.

Post-hoc analysis reads completed artifacts without restarting benchmarks.
The artifact formats and analysis modules are documented in
[Architecture](ARCHITECTURE.md#post-hoc-connection-analysis).

### HopRAG edge blocks

`RAG_HOP_EDGE_BLOCK_SIZE=128` controls adapter scoring memory, not candidate
selection. `scripts/benchmark_hoprag_edges.py` measures blocks 8, 32, 64 and 128
against the pinned native dense function using existing per-document caches.
It records dtype, exact score equality, throughput and process peak RSS; these
are implementation measurements, not completed corpus indexing costs.

The HopRAG indexer preserves per-document node/question/embedding caches and
stage completion sets. An interrupted, uncommitted edge group is scored again;
completed groups remain recorded. Preserve previous index statistics before
recovery and report interrupted edge-attempt costs separately from its duration.
