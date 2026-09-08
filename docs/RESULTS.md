# Result Evidence Register

This register separates retained admitted results from current execution state.
Five MultiHop-RAG runs have passing admission receipts. The 16-target comparison
is incomplete; no MuSiQue quality result or repaired HopRAG/HippoRAG2 result is
claimed here. Strategy order comes from `core/strategy_registry.py`.

## Retained full-run results — 2026-09-09 06:07 KST

Each row contains all 2,556 MultiHop-RAG queries with zero terminal query
failures. Retrieval metrics use the same 2,255 eligible questions; official QA
accuracy and mean query latency use all 2,556 questions. Scores below are
percentages. Admission receipts were checked against the retained result, detail and
original index-stat hashes at this snapshot. This is preservation of their original admission, not a claim
that they were rerun under the new unseeded-generation or repair profiles.

| Method | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Official QA accuracy | Mean query latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Naive RAG | 69.36 | 83.90 | 55.20 | 26.35 | 28.91 | 3.1811 |
| Prehop | 92.42 | 94.90 | 82.00 | 45.22 | 34.23 | 35.1713 |
| LinearRAG | 81.91 | 85.28 | 67.30 | 34.16 | 55.48 | 33.0635 |
| MS GraphRAG | 58.27 | 62.48 | 37.82 | 18.68 | 37.56 | 22.8464 |
| LightRAG | 72.73 | 87.54 | 63.65 | 30.39 | 11.19 | 97.8781 |

### Recorded indexing and query costs

Index times cover each original successful 609-document index. Query batch
wall time covers the full 2,556-query batch. Wall time per unit is inverse
throughput; it is distinct from mean individual query latency. These are
recorded execution costs, not controlled estimates of isolated serving latency.
Prehop includes enabled trace I/O; local method backbones remain declared.

| Method | Index wall (s) | Index wall / document (s) | Query batch wall (s) | Batch wall / query (s) |
|---|---:|---:|---:|---:|
| Naive RAG | 119.1448 | 0.1956 | 1047.6934 | 0.4099 |
| Prehop | 2504.8409 | 4.1130 | 11309.8792 | 4.4248 |
| LinearRAG | 638.5990 | 1.0486 | 10787.6445 | 4.2205 |
| MS GraphRAG | 6861.4718 | 11.2668 | 7617.4921 | 2.9802 |
| LightRAG | 6562.6064 | 10.7760 | 31578.0235 | 12.3545 |

### Evidence locations

Each directory below contains `admission.json`; its bindings identify the exact
result, detail rows and original index statistics. The result JSON may retain
`completed_unadmitted`; the passing bound receipt owns admission status.

- Naive RAG: `data/results/multihop-benchmark-first-20260908-multihoprag-naive/`
- Prehop: `data/results/multihop-benchmark-first-20260908-multihoprag-prehop/`
- LinearRAG: `data/results/linear-native-batch-v2-20260908-multihoprag-linear_rag/`
- MS GraphRAG: `data/results/rolling-two-models-20260908-multihoprag-ms_graphrag/`
- LightRAG: `data/results/rolling-two-models-20260908-multihoprag-lightrag/`

`data/maintenance/documentation-evidence-20260909.json` records the snapshot
paths, metrics and admission checks for result, detail and index-stat hashes. The retained Prehop/Naive paired
analysis remains in `tmp/paper-draft-20260908/admitted-results.json` with its
reproduction script. Its 10,000 seed-42 paired bootstrap resamples describe
query-sample uncertainty, not run-to-run variation or new comparative tests.

## Execution update — 2026-09-09 07:35 KST

GFM-RAG and HopRAG were restarted as `gfm-cancellable-20260909` and
`hoprag-cancellable-20260909` behind the cancellable front queue. The previous
600-second GFM attempt and HopRAG canary are superseded, not final results.
MS GraphRAG retains its original process. Legacy in-flight requests may still
drain through the original queue; this restart does not establish full recovery.

### Superseded execution update — 2026-09-09 07:20 KST

The 60-second GFM-RAG benchmark attempt was discarded at the user’s request.
Its partial results and run-local artifact copy were removed; the original
completed index is retained. The replacement run uses
`gfm-timeout600-20260909-multihoprag-gfm_rag` and applies the common 600-second
QA transport deadline. It starts from the first query. LightRAG and MS GraphRAG
indexing continue, and the rolling capacity remains three.

### Superseded execution update — 2026-09-09 06:28 KST

The rolling controller now permits three active jobs. It retained both MuSiQue
index processes and launched GFM-RAG/MultiHop-RAG from the completed index. By 06:29 KST,
its benchmark worker was writing answer audit records. The shared request limit remains 120. This launch is not
a completed benchmark result or a measured speedup; four jobs remain untested.

### Earlier execution snapshot — 2026-09-09 06:07 KST

The shared queue continues to serve two active MuSiQue indexing jobs:
LightRAG and MS GraphRAG. The replacement controller adopted their existing
processes. It dispatches GFM-RAG/MultiHop-RAG, HopRAG/MultiHop-RAG,
HippoRAG2/MuSiQue, LinearRAG/MuSiQue and GFM-RAG/MuSiQue as slots become free.
This is a timestamped queue plan, not evidence that these retries have started.

- GFM-RAG previously failed before benchmark startup because a prior job's LLM
  seed reached index validation. Target initialization and seed handling are
  repaired; the new benchmark is pending.
- HopRAG's previous full index failed on two documents after native response
  retries returned three values to a two-value caller. The repaired adapter is
  integrated into the main repository; its new full run is pending.
- HippoRAG2/MuSiQue previously failed NER on 20 chunks. The structured extraction
  adapter is enabled for the next fresh run; no successful full result follows
  from unit tests alone.
- The repair suite passed 108 tests. HopRAG's 57,716 saved responses parsed as
  JSON in offline replay. These are engineering checks, not indexing completion
  or benchmark admission.

Read `tmp/current_benchmark_campaign.txt` and then
`data/results/rolling-two-models-20260908/slot-fill-status.json` for live state.
This controller can mix indexing and benchmark jobs. Older supervisor target
labels and an earlier failed run's status are historical evidence. Check the
current controller's active jobs, logs and target receipts before reporting ETA.
HTTP 200 counts do not establish successful native extraction, and completed
futures can include failed documents. Do not extrapolate full index time from
extraction alone when graph construction is still pending.

## Primary matrix

| Method | MultiHop-RAG benchmark | MuSiQue benchmark |
|---|---|---|
| Prehop | Retained admitted result | No admitted result |
| Naive RAG | Retained admitted result | No admitted result |
| HopRAG | Failed index; repaired run pending | No admitted result |
| MS GraphRAG | Retained admitted result | Indexing |
| LightRAG | Retained admitted result | Indexing |
| HippoRAG2 | No admitted result | Failed index; repaired run pending |
| GFM-RAG | Repaired benchmark pending | No admitted result |
| LinearRAG | Retained admitted result | No admitted result |

An index or canary completion is not a full benchmark admission. Retained
results keep their original code, seed, profile and cost evidence. New profiles
must not relabel old artifacts. Source edits no longer block index dispatch,
but corpus, artifact, semantic compatibility and result-completeness checks
remain required for reuse and admission.

## Prepared dataset identities

Each admitted artifact must bind the prepared corpus and full query file.
The query and metric denominators are defined in
[the cost and metric protocol](THROUGHPUT_EXECUTION.md#final-tables-and-measurement-definitions).
A different corpus fingerprint identifies different inputs and requires fresh
compatible evidence.

| Dataset | Source documents | Full queries | Corpus fingerprint |
|---|---:|---:|---|
| MultiHop-RAG | 609 | 2,556 | `c11b84f626c08d06d6dbc938512275824567aaffdc77a0f0b5424ad94f13a8ee` |
| MuSiQue answerable dev | 21,099 | 2,417 | `63562ceaf17343507b305b152af93245959f458412be321662cfbc8fde9f2a34` |

## Evaluation configuration

| Setting | Value |
|---|---|
| Remote generation model | `gemma-4-31b-it` |
| Controlled remote embedding model | `qwen3-embedding-4b`, 2,560 dimensions |
| Remote embedding batch/concurrency | Serial default 16 / 1; explicit profile for throughput campaigns |
| Query concurrency | Serial default 1; explicit content-bound throughput profile for new campaigns |
| Evaluation / sampling seed | 42 |
| LLM generation seed | Omitted in new paper runs from 2026-09-09; retained runs keep their recorded values |
| LLM judge | disabled |

LinearRAG instead uses pinned
`sentence-transformers/all-mpnet-base-v2` embeddings at 768 dimensions in the
primary official-faithful mode. GFM-RAG retains its checkpoint-defined local
components. HopRAG uses the pinned upstream runtime with the registered
`adapter-json-recovery-v1` response intervention. HippoRAG2 uses
`strict-extraction-v1` for newly launched repaired indexes. These profiles
describe changed response handling and are not claims of unmodified upstream
execution. See [runtime requirements](RUNTIME_REQUIREMENTS.md#native-output-handling-and-artifact-validation).

## Admission checks

`scripts/verify_paper_target.py` and
`scripts/verify_submission_consistency.py` must establish all of the following
before a cell becomes `admitted`:

1. Every query reached an answer or a terminal query failure: 2,556 ordered
   rows for MultiHop-RAG or 2,417 for MuSiQue. Terminal failures follow the
   versioned zero-score policy below; source-mapping or integrity failures
   block admission.
2. Detail rows have unique, gap-free indices in input order. Query IDs,
   query-record digests, ground-truth identities, eligible counts, and all
   aggregates recompute exactly from those rows. Primary per-query metrics also
   recompute from the saved answer/retrieved evidence and authoritative query
   manifest; expected evidence must match that manifest. Primary averages must
   be finite numbers in [0, 1], with ineligible rows excluded by the dataset rule.
3. Corpus manifests and index statistics bind the full source-ID set, source
   count, content digest, query-ID set, and query-record digest. MuSiQue schema
   v2 keeps paragraph IDs distinct from source filenames.
4. The completed index has exact source coverage derived from its stored
   retrieval artifacts, plus a content-addressed artifact inventory. A staged
   input directory alone is not proof of coverage.
5. The semantic configuration ID and hash match the checked-in per-strategy
   specification, including the upstream revision and method-defining model,
   checkpoint, schema, and retrieval settings. Operational throughput settings
   are recorded separately.
6. The generation and embedding transports match the strategy's declared
   policy. Remote calls use the single fail-closed LiteLLM gateway; pinned
   local method components match their exact revisions and dimensions.
7. Dataset metrics remain separate, and indexing cost, query service latency,
   worker-queue delay, and end-to-end latency retain their distinct meanings.
   Missing upstream token or cost telemetry is marked incomplete rather than
   estimated.
8. External adapters report their executed observation or recovery profile and bind the
   index-time audit prefix. Missing or changed audit bytes and stale profiles
   prevent admission. Recorded native fallback events do not independently
   invalidate a completed native run; extraction quality is evaluated as produced. Later query appends do
   not replace the index-time evidence; the final inventory binds them separately.

## Failure handling

The `terminal-query-failure-zero-v1` policy keeps terminal query failures in
primary quality aggregation. Terminal query exceptions receive zero answer and
retrieval/support scores, an incorrect-answer label, and a durable error trace.
Failure counts and rates accompany the scores. Optional judge metrics remain
unjudged, not fabricated zeros. Latency and available usage include failed calls;
missing usage remains unavailable. Paired bootstrap retains failed queries as
zero-score observations. Successful rows retain metric applicability filtering.
The current `metric_value` helper gives failed quality rows zero even when a
successful null query would be ineligible for retrieval metrics. Report failed
null-query counts separately so this failure-inclusive denominator is visible.
The admitted Naive/Prehop pair has no terminal failures, so its retrieval
denominator remains 2,255 in both columns.

A batch that executed every query can complete with query failures. Source
mapping/integrity errors stop subsequent queries for that target and block
admission. Resuming preserves terminal error rows and only executes missing
queries; rerunning a failed query requires a separately recorded experiment.

An index failure has unavailable quality, not zero quality. The index supervisor
continues other eligible targets and writes `outcomes.json` and `outcomes.md`
with states, attempt durations and diagnostic references. A smoke failure blocks
only that target's full index. Failed index time is attempt cost, not successful
per-document indexing performance. Index and query completeness, source identity,
audit hashes and cost telemetry are still verified independently of quality.

## Artifact retention

Keep the complete evidence chain for every admitted result and active run.
The [maintainer policy](../CLAUDE.md#execution-and-repository-hygiene) governs
generated artifacts and explicit cleanup. Results from removed strategies cannot supply primary matrix cells.

## Publication synchronization

- Public summaries and presentation sources may copy numbers only from
  `admitted` rows in this file.
- Relative changes, uncertainty intervals, chart dimensions, and latency
  summaries must be recomputed from the admitted detail artifacts.
- A dirty tracked worktree is recorded in provenance; it is not silently
  described as clean. Semantic compatibility is decided by the recorded
  configuration and artifact identities; project commit numbers are provenance only.
- Target failures remain visible even when a scheduler finishes dispatching all
  jobs. A scheduler state of `completed` is not proof that every target passed;
  inspect target exit codes and admission receipts.

## Throughput cost profile

New evidence uses contract v3 and the protocol in [THROUGHPUT_EXECUTION](THROUGHPUT_EXECUTION.md).
Indexing s/doc and query s/query are total measured phase wall time divided by
manifest source count and full query count, respectively; they are not mean
request latency. Resumed query runs remain ineligible for continuous-run query
cost. Transport and producer pilots are engineering validation and cannot
supply a full benchmark result.
Prehop phase wall time includes its enabled trace I/O. Retain the trace reference
with the result and disclose instrumentation when comparing costs. Diagnostic
trace files are excluded from retrieval-index storage measurements.
