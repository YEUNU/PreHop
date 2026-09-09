# Result Evidence Register

This register preserves completed benchmark evidence, dataset identities, and
measurement definitions. Live counters and queue state belong in generated
campaign artifacts, not this document.

## Completed full-run results

Each row contains all 2,556 MultiHop-RAG queries with zero terminal query
failures. Retrieval metrics use the same 2,255 eligible questions; official QA
accuracy and mean query latency use all 2,556 questions. Scores below are
percentages. Existing runs retain their original execution settings and evidence.
The first four runs record generation seed 42; the GFM-RAG query run omits
the generation seed and retains checkpoint-defined local retrieval components.
All five use benchmark concurrency eight. Completion does not imply identical
backbones, decoding settings, or native answer prompts. QA is preserved here
as recorded evidence, not as an isolated retrieval-quality comparison.

| Method | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Official QA accuracy | Mean query latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Naive RAG | 69.36 | 83.90 | 55.20 | 26.35 | 28.91 | 3.1811 |
| Prehop | 92.42 | 94.90 | 82.00 | 45.22 | 34.23 | 35.1713 |
| MS GraphRAG | 58.27 | 62.48 | 37.82 | 18.68 | 37.56 | 22.8464 |
| LightRAG | 72.73 | 87.54 | 63.65 | 30.39 | 11.19 | 97.8781 |
| GFM-RAG | 49.62 | 55.65 | 35.97 | 18.92 | 22.46 | 37.7325 |

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
| LinearRAG (index only) | 638.5990 | 1.0486 | — | — |
| MS GraphRAG | 6861.4718 | 11.2668 | 7617.4921 | 2.9802 |
| LightRAG | 6562.6064 | 10.7760 | 31578.0235 | 12.3545 |
| GFM-RAG | 1033.6809 | 1.6973 | 12300.5139 | 4.8124 |

### Evidence locations

These links open the final result JSON directly. Detail rows, traces and index
cost evidence remain alongside each run. No seed-exception record is required.

- [Naive RAG](../data/results/multihop-benchmark-first-20260908-multihoprag-naive/naive/multihoprag/seed_42/naive_multihoprag.json)
- [Prehop](../data/results/multihop-benchmark-first-20260908-multihoprag-prehop/prehop/multihoprag/seed_42/prehop_multihoprag.json)
- [MS GraphRAG](../data/results/rolling-two-models-20260908-multihoprag-ms_graphrag/ms_graphrag/multihoprag/seed_42/ms_graphrag_multihoprag.json)
- [LightRAG](../data/results/rolling-two-models-20260908-multihoprag-lightrag/lightrag/multihoprag/seed_42/lightrag_multihoprag.json)
- [GFM-RAG](../data/results/benchmark-first-two-20260909-multihoprag-gfm_rag/gfm_rag/multihoprag/seed_42/gfm_rag_multihoprag.json)

## Derived evidence

The manuscript's four-system retained-output analysis uses Prehop, Naive RAG,
LightRAG, and MS GraphRAG. Its input hashes and per-query data are recorded in
`artifacts/paper_revision_20260909/analysis.json` and `*.per_query.jsonl`.
`coverage_partition.json` records complete, partial, and absent fact coverage.
These are analyses of saved outputs, not new retrieval runs.

Retrieval diagnostics use 2,255 evidence-bearing questions. Exact-fact recall
and AllFacts use the official case-sensitive, space/newline-stripped matcher;
they are additional measures, not official leaderboard metrics. Normalized/fuzzy
recall is a separate matching-rule sensitivity analysis. These diagnostics do not establish reasoning correctness.

## Primary matrix

The supported methods and their identities are defined in
`core/strategy_registry.py`. Only completed full-run artifacts can supply
benchmark scores. An index or canary alone is not a completed benchmark.
Runtime readiness, index reuse, and checkpoint checks still apply before
execution or resume. Consult generated campaign artifacts for queue state.

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
`adapter-json-recovery-v1` response intervention. This profile
describes changed response handling and is not a claim of unmodified upstream
execution. See [runtime requirements](RUNTIME_REQUIREMENTS.md#native-output-handling-and-artifact-validation).

## Completion records

`scripts/record_paper_completion.py` records a finished benchmark without a
final paper-policy check. It accepts a completed execution status and writes
`admission.json` with `status=admitted` and `verification=disabled_by_user`.
The legacy status name supports existing launchers; it does not certify an
independent validation. In-progress and failed executions are not completed.
The result JSON retains its original `completed_unadmitted` execution label.

`scripts/verify_paper_target.py` is only a compatibility entry for already-running
launchers. The standalone `verify_submission_consistency.py` remains an optional
offline audit, outside automatic completion and dispatch. Seed differences do
not require an exception record. Raw results, detail rows and traces are retained.

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
