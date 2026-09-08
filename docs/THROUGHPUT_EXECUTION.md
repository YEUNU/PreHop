# Throughput execution and paper cost protocol

Use this protocol to select request and document concurrency, launch indexing
or benchmark work, and report comparable per-document and per-query costs.
The eight primary methods run on both complete prepared datasets. Run
one target at a time on the same remote replica/GPU and declared local resource
budget. Queueing changes request admission only, not native prompts, retrieval
budgets, ordering dependencies, seeds or retry policies. Producer parallelism uses
native APIs through adapters; adaptive Youtu schema visibility is declared
as a distinct synchronized parallel policy. GPU utilization is observational; 99% is not an
admission criterion. An external workload on the same backend invalidates a
claim of exclusive resources even though local target locks pass.

## Execution contract

`core/execution_profile.py` binds the selected settings to index and query
provenance. `core/inference_queue.py` applies shared upstream request limits
across native worker processes without changing the approved gateway identity.
The queue has bounded connection handlers, preserves request bodies and
streaming responses, and performs no retries. It does not log request bodies
or credentials.

Evidence contract v3 binds the profile to gate and admission checks. Older
contexts must be revalidated. Checkpoint recovery can retain effectiveness
results, but interrupted query runs cannot supply continuous-run throughput.
For module ownership and producer behavior, see
[ARCHITECTURE](ARCHITECTURE.md#amortized-throughput-cost-evidence-v3).

## Profiles and pilot

`RAG_EXECUTION_PROFILE` must be an absolute JSON path exported before Python
starts. Version 3 uses one `inference_concurrency` limit shared by generation
and embedding requests. It retains embedding batch size, query workers and
document producer settings; it has no per-kind request quotas. Client ceilings
resolve to the shared limit so an old embedding ceiling of two cannot prevent
use of the pool. Native dependencies and producer counts still bound demand.

Legacy version 1 declares four transport settings. Version 2 additionally
binds active document workers, document prefetch, LightRAG insertion workers
and Youtu construction workers, with optional bounded Prehop chunk lookahead
(`prehop_chunk_concurrency`, default 1). Adapter producer settings supersede ambient
defaults. Youtu concurrency and schema synchronization also enter semantic policy. The profile's canonical JSON hash enters both
index and query provenance, gate context, and admission. The included
`configs/execution_profiles/throughput-pilot.json` (generation 30, embedding
16/2, queries 8) is an unvalidated starting point, not a measured optimum.
Without a profile, the registry's existing defaults remain available. See
[Adapter producer parallelism](ARCHITECTURE.md#adapter-producer-parallelism)
for per-method behavior.

### Shared indexing profile

`configs/execution_profiles/index-shared-120.json` sets a combined limit of
120 active inference requests. Generation and embedding compete for the same
semaphore, with no reserved slots for either kind. For example, 90 generation
requests leave room for 30 embedding requests; either kind alone can use all
120 slots. Requests above the active limit wait within the bounded handler
capacity. This does not promise FIFO ordering or 120 GPU sequences.

Embedding batches remain at 16 inputs and producer settings retain the values
below. Queue snapshots expose `limits.shared` and aggregate `metrics.shared`,
alongside per-kind observational metrics. The aggregate peak is measured
directly; do not add the two per-kind peaks or add aggregate counts to their
components. Protocol-level concurrency tests do not establish the best remote
throughput. Gateway and client probes verify the selected embedding response
contract, and matrix preflight verifies runtime/configuration readiness. Neither
check builds full indexes or establishes sustained throughput for every method.
Validate sustained load separately before treating an execution profile as optimal.

### Previous indexing profile

`configs/execution_profiles/index-throughput-tested.json` contains the tested
indexing configuration:

| Profile field | Value | Meaning |
|---|---:|---|
| `generation_concurrency` | 60 | Shared active generation request limit |
| `embedding_batch_size` | 16 | Remote embedding inputs per batch |
| `embedding_concurrency` | 2 | Shared active embedding request limit |
| `benchmark_concurrency` | 8 | Query worker bound; separate validation remains required |
| `index_document_concurrency` | 60 | Active Prehop document limit |
| `index_prefetch_documents` | 120 | Prehop document scheduling window |
| `prehop_chunk_concurrency` | 8 | Prehop per-document chunk lookahead |
| `lightrag_document_concurrency` | 32 | Native LightRAG insertion limit |
| `youtu_document_concurrency` | 60 | Adapter Youtu document worker limit |

Indexing pilots and native smoke builds validate execution at these settings.
They do not establish a global optimum, query throughput, downstream quality
equivalence, or publication admission. Embedding limits remain conservative.
Prehop trace I/O is enabled by default and contributes to measured phase wall
time; record its trace reference when comparing costs.
For the reported server sequence limits, see
[Serving capacity](RUNTIME_REQUIREMENTS.md#serving-capacity).

### Compare candidate profiles

Capture a small, fixed representative request workload without gold answers as
JSONL rows with `endpoint` (`chat/completions` or `embeddings`) and the original
`body`. Use the configured remote models and non-streaming responses for pilot
replay. Avoid confidential material in files intended for publication. Give
candidate profiles the same workload, resource budget and tuning opportunities.

```bash
.venv/bin/python scripts/pilot_inference_queue.py \
  --requests /absolute/path/representative-requests.jsonl \
  --profiles /absolute/path/profile-4.json /absolute/path/profile-8.json \
  --output /absolute/path/pilot-results.json \
  --selected-profile /absolute/path/selected-profile.json
```

The pilot chooses the highest successful request throughput among candidates
with no HTTP/envelope failures. It preserves the workload hash and all candidate
measurements, not the request bodies. Repeat with reversed candidate order to
assess warmup/cache effects. Separately compare low/high load extraction validity,
output changes, failure/retry rates and downstream metrics. Envelope success is
not proof of algorithmic correctness or batch invariance. Request replay does
not tune native query worker concurrency: use end-to-end small canaries for that.
Freeze the chosen profile before full runs and repeat the ordered gates.

### Validate before full indexing

1. Test profile binding, bounded scheduling, assembly order and failure
   propagation. For Youtu, verify schema updates retain all types while LLM
   calls remain outside the lock.
2. Run relevant tests with the main and actual pinned native interpreters.
   Verify original source hashes and checkout integrity before and after.
3. Run fixed real-document pilots for changed producer paths. Record coverage,
   elapsed time, errors, retries and achieved concurrency. Small synthetic
   smoke builds establish integration, not saturation or an optimum.
4. Freeze code and profile before full indexing. Retain target run IDs, original
   costs and effective settings. A profile change requires a fresh measurement.

Local model work and dependent graph stages can leave the remote LLM idle.
Validate query concurrency separately: native query workers can duplicate model
and index memory. Indexing pilots do not establish downstream quality or query
throughput.

## Execution

For a foreground target after its prerequisites are ready:

```bash
.venv/bin/python scripts/run_with_inference_queue.py \
  --profile /absolute/path/selected-profile.json \
  --metrics /absolute/path/fresh-queue-metrics.json \
  -- bash scripts/run_paper_target.sh multihoprag prehop fresh-target-id
```

The wrapper waits for the child and owns the queue until it exits. Do not put a
background-launch command inside this wrapper. For persistent campaigns instead
export `RAG_EXECUTION_PROFILE` before creating a new plan and use the
[paper campaign plan/launch workflow](RUNTIME_REQUIREMENTS.md#durable-campaign-ownership). The supervisor starts its own queue,
passes the private route to native children, writes `inference-queue-*.json`
after stages and on shutdown, drains outstanding requests before the next stage,
and retains its existing sequential gate protocol.
Use the same selected profile for later read-only admission/verification.

Local locks reject concurrent queue owners and overlapping measured throughput
targets. They cannot establish exclusivity against other users or remote clients.
Queue metrics record request/error counts, peak active requests, summed upstream
and queue seconds. Summed request seconds are not campaign wall time. Client
connection/worker limits may restrict actual concurrency below the configured cap.

## Final tables and measurement definitions

MultiHop-RAG: Hits@4, Hits@10, MRR@10 and MAP@10 use its 2,255 answerable
retrieval questions; 301 null questions remain separate diagnostics. Cost uses
all 2,556 processed questions and all 609 source documents. MuSiQue uses Answer
EM/F1 and Support precision/recall/F1 over 2,417 questions; cost normalizes by
21,099 manifest source documents. Keep datasets in separate tables.

- `amortized_indexing_cost`: existing index-pipeline wall seconds divided by
  manifest source count, including internal waiting/retries and native work.
  Existing adapter boundaries remain explicit; storage/reporting measurements
  after the frozen timer are excluded. Record total time alongside s/doc.
- `amortized_query_cost`: wall seconds from dispatching the complete query batch
  through the last answer or terminal query failure, divided by the full query count. Internal
  queues, retry time, and any interleaved evaluation/checkpoint work that delays
  later answers are included. Initialization and trailing report generation are
  excluded. This is inverse throughput, not mean response latency.
- Incomplete or integrity-failed runs have null normalized performance values.
  Fully executed batches include terminal query failures in measured query cost. Resumed query
  runs retain separate segment timing but have null s/query and throughput:
  they are not uninterrupted performance measurements. A fresh benchmark on the
  same immutable index can supply query cost without repeating indexing.
- Reused indexes retain the original complete index's measured time and profile;
  source costs are not replaced by clone/reuse preparation time.
- Keep request latency, queue waiting and stage shares separately labeled. A
  single execution supplies no estimate of run-to-run performance variance.
- Storage is method-labeled: Neo4j logical payload estimate versus physical
  retrieval-file bytes. Do not rank them as equivalent physical database sizes.
- Missing native token or monetary telemetry stays unavailable. Record failed
  attempts and pilot budget separately from the successful target's cost.

MuSiQue graph/refinement/selection controls reuse the same immutable index and
profile. Fixed-candidate ranking/order controls remain separate analyses. Paired
bootstrap uses 10,000 resamples and seed 42; query resampling does not measure
LLM/index-build variability. Broader comparative latency claims require a
separate, declared fixed-load measurement.

## Index-only batch

After testing a profile and validating all runtimes, select the prepared main
environment and launch with a fresh campaign ID from the repository root:

```bash
export PYTHON_BIN=/absolute/path/to/prepared/main-venv/bin/python
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/prepared/main-venv
export RAG_EXECUTION_PROFILE="$PWD/configs/execution_profiles/index-shared-120.json"
"$PYTHON_BIN" scripts/index_matrix.py launch indexing-01
```

The launcher returns a receipt with the supervisor PID, plan path and status
path. The supervisor owns a nohup session, first performs a small native smoke build for each planned target (sixteen
in the default matrix), then processes the
complete corpora for targets whose smoke build passed. Failed targets are
recorded independently; unrelated targets continue. The run writes
`data/results/<campaign>/index-supervisor/{plan,status,queue-metrics}.json`
and per-stage logs. Full source indexes use `<campaign>-index-<dataset>-<strategy>`
run IDs, so a later exact-target benchmark can use their original indexing
artifacts. Code/configuration drift stops execution. Existing artifacts are
never overwritten and the runner does not restart failed attempts implicitly.

Read the status file to verify the active `stage`, per-target outcomes and
supervisor state. `smoke/...` identifies a bounded integration build;
`index/...` identifies full-corpus indexing. `index_complete` describes one
index. The supervisor reports `completed` only when every target in the frozen plan completes its full index;
otherwise it reports `failed`. These process states are separate from the
publication ledger in [RESULTS](RESULTS.md).

### Check progress and stalls

Read `status.json` and the per-stage log paths it names. A fresh supervisor
heartbeat proves that the supervisor is alive; it does not prove that the child
is completing documents. Compare successive `Indexing progress` records and,
for Prehop, the trace event timestamps. The persisted `queue-metrics.json` is a
stage-boundary snapshot, not a live utilization reading.

Low remote GPU utilization can be expected during parsing, local model work,
graph writes, and final graph construction. If document counts and trace events
also stop advancing, investigate the child instead of increasing the queue
limit or extrapolating an ETA. Prehop HTTP trace writes must remain independent
of the default executor used by embedding permit waiters; see the
[tracing contract](ARCHITECTURE.md#prehop-tracing).

Preserve interrupted logs and traces as diagnostic evidence. A corrected build
uses a fresh campaign ID and repeats validation; an interrupted index cannot
supply continuous-run indexing cost. Keep current run IDs and status pointers
in local run artifacts rather than copying them into this guide.

The process continues after the initiating shell or task exits. This runner
does not start the separate paper campaign monitor or send chat notifications,
and it does not provide automatic reboot recovery. It writes
`benchmark_admitted: false`; publication still requires the complete query
benchmark and applicable independent review, live gates and admission checks.

## Documentation during an active campaign

The index and benchmark supervisors bind executable source content and effective
configuration. Editing an executed `.py` or `.sh` file can stop scheduling at
the next target boundary; an already running query process may continue with
its loaded code. Markdown-only documentation edits do not change that source
digest. This is not permission to edit configuration files during measurement.

Record timestamped progress and provisional metrics in the
[result evidence register](RESULTS.md). Keep the exact query
population, metric denominator and admission state visible. Do not copy live
counters into setup instructions, implementation contracts or manuscript claims.
A verifier failure and a query failure are different states; completed query
artifacts must remain available for verification after a verifier repair.


## LinearRAG query batches

With benchmark concurrency eight, the adapter submits up to eight questions to
the original batch QA API. This permits concurrent generation after the native
retrieval phase. It does not change native parameters or promise continuous
GPU saturation: MPNet and graph retrieval use local resources. The shared
inference limit remains 120. Record actual batch sizes and report batch wall
time/query separately from request latency. Restart interrupted serial runs
under a fresh ID when switching to this execution path.
