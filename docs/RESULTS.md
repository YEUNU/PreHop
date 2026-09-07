# Result Evidence Register

This file is the canonical register for reportable full-run results. At this
revision no cell in the primary paper matrix has passed admission, so this
document intentionally contains no primary numerical result. Literature
citations elsewhere describe prior work; they are not evidence for a local
result.

Admission binds the current corpus bytes, semantic configuration, execution
profile and result artifacts. A changed identity requires compatible evidence
and current verification. Index reuse retains the original indexing cost and
requires a verified index link plus complete query evaluation.

This register tracks publication admission. For live indexing progress, read
the selected campaign's `index-supervisor/status.json`; smoke/index process
states do not change a publication cell automatically.

## Primary matrix

The primary matrix has 16 independent targets: eight methods on MultiHop-RAG
and MuSiQue. Its order comes from `core/strategy_registry.py`, which is the
single source of truth used by the Python CLI and the shell runners.

| Strategy | MultiHop-RAG | MuSiQue | Backbone policy |
|---|---|---|---|
| Prehop | `planned` | `planned` | controlled remote generation and embedding |
| Naive RAG | `planned` | `planned` | controlled remote generation and embedding |
| MS GraphRAG | `planned` | `planned` | official pipeline with controlled remote backbone |
| LightRAG | `planned` | `planned` | official method with controlled remote backbone |
| HippoRAG2 | `planned` | `planned` | official method with controlled remote backbone |
| GFM-RAG | `planned` | `planned` | official method with checkpoint-defined local components |
| LinearRAG | `planned` | `planned` | official-faithful pinned MPNet mode |
| Youtu-GraphRAG | `planned` | `planned` | controlled native agent API with pinned MiniLM and NER |

The publication experiment ledger uses only these status values. The table
above records that ledger; it does not mirror indexing-only smoke completion:

- `planned`: the target has not passed a canary.
- `canary_passed`: the strategy contract passed a small non-reportable canary.
- `in_progress`: the complete target is running or has a resumable checkpoint.
- `completed_unadmitted`: the complete target produced artifacts, but admission
  has not passed.
- `admitted`: the complete target passed every check below and may supply
  numerical results.
- `failed`: the target or its admission check failed.

A canary is not target completion, and target completion is not admission.
Only `admitted` artifacts may supply numbers to this register or presentation
material. A successful benchmark artifact has execution
status `completed_unadmitted` until `scripts/verify_paper_target.py` writes a passing
`data/results/<run-id>/admission.json` record with status `admitted`. That
record binds the result JSON, complete detail JSONL, exact index-stats path and
bytes, canonical index-policy digest, runtime freeze and constraints,
post-query retrieval-artifact inventory and versioned effective model
configuration. Git and verifier-source hashes remain separate provenance;
only semantic/evidence contract changes affect compatibility.
Every strict skip reruns current verification. An unchanged admitted binding
keeps the original ledger bytes; a changed or invalid binding is rejected and
preserved, requiring a fresh namespace. Failed probes write separate receipts
and do not reserve the eventual successful admission ledger.

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
| Seed | 42 |
| LLM judge | disabled |

LinearRAG instead uses pinned
`sentence-transformers/all-mpnet-base-v2` embeddings at 768 dimensions in the
primary official-faithful mode. Youtu-GraphRAG uses pinned
`sentence-transformers/all-MiniLM-L6-v2`
embeddings at 384 dimensions and its pinned NER dependency. GFM-RAG uses the
embedding and graph components identified by its validated checkpoint and
configuration files. These declared method components are not mislabeled as
the controlled remote embedding backbone.

Youtu runs the pinned agent loop. Adapter observers capture its answer and
final evidence order without changing retrieval or deduplication; the common
benchmark performs post-answer evaluation. The target remains labelled
`controlled_adapter` for its declared transport, format and concurrency changes.

## Admission checks

`scripts/verify_paper_target.py` and
`scripts/verify_submission_consistency.py` must establish all of the following
before a cell becomes `admitted`:

1. The full target completed with zero error rows: 2,556 ordered query rows for
   MultiHop-RAG or 2,417 for MuSiQue.
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
   input directory alone is not proof of coverage. For Youtu, admission binds
   exact staged-source-to-native-chunk coverage separately from observational
   native graph reachability. Native duplicate deduplication may make the
   latter incomplete; the ledger records that method-native limitation without
   modifying retrieval or mislabelling input coverage.
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
8. External adapters report their native observation profile and bind the
   index-time audit prefix. Missing or changed audit bytes and stale profiles
   prevent admission. Recorded native fallback events do not independently
   invalidate a completed native run; extraction quality is evaluated as produced. Later query appends do
   not replace the index-time evidence; the final inventory binds them separately.

## Failure handling

The `terminal-query-failure-zero-v1` policy keeps every executed query in the
primary quality denominator. Terminal query exceptions receive zero answer and
retrieval/support scores, an incorrect-answer label, and a durable error trace.
Failure counts and rates accompany the scores. Optional judge metrics remain
unjudged, not fabricated zeros. Latency and available usage include failed calls;
missing usage remains unavailable. Paired bootstrap retains failed queries as
zero-score observations, subject to the existing gold-evidence applicability rule.

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
generated artifacts and explicit cleanup. Legacy adapter results cannot supply
primary matrix cells.

## Publication synchronization

- Public summaries and presentation sources may copy numbers only from
  `admitted` rows in this file.
- Relative changes, uncertainty intervals, chart dimensions, and latency
  summaries must be recomputed from the admitted detail artifacts.
- A dirty tracked worktree is recorded in provenance; it is not silently
  described as clean. Semantic compatibility is decided by the recorded
  configuration and artifact identities; project commit numbers are provenance only.
- The matrix continues after independent target failures, reports every failed
  target, and exits nonzero when any target failed.

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
