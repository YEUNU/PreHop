# Result Evidence Register

The revised Prehop design removes initial query rewriting and evidence-conditioned
query regeneration/re-search. The original query searches all three channels at
every input length. This single-pass policy is implemented. Earlier Prehop
results are excluded from the current paper and this register. The new MultiHop-RAG Prehop run completes all 2,556 questions with no terminal
failures. It is the only Prehop run used in the current result tables. Original
artifacts retain their recorded settings.

This register preserves completed benchmark evidence, dataset identities, and
measurement definitions. Live counters and queue state belong in generated
campaign artifacts, not this document.

## Completed full-run results

Each row contains all 2,556 MultiHop-RAG queries with zero terminal query
failures. Retrieval metrics use the same 2,255 eligible questions; official QA
accuracy and mean query latency use all 2,556 questions. Hits and QA scores are percentages; MRR and MAP are on a 0–1 scale. Existing runs retain their original execution settings and evidence.
Naive RAG, MS GraphRAG, and LightRAG record generation seed 42; Prehop, GFM-RAG, and LinearRAG omit the generation seed and retain their method-specific
local retrieval components.
All six use benchmark concurrency eight. The replacement LinearRAG run uses
the restored native query parameters (0.4 expansion threshold, passage weight 2,
and three expansion sentences). Completion does not imply identical
backbones, decoding settings, or native answer prompts. QA is preserved here
as recorded evidence, not as an isolated retrieval-quality comparison.

| System | Hits@4 (%) | Hits@10 (%) | MRR@10 | MAP@10 | Official QA accuracy (%) | Mean query latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Prehop | 92.11 | 94.50 | 0.8166 | 0.4485 | 34.19 | 32.87 |
| MS GraphRAG | 58.27 | 62.48 | 0.3782 | 0.1868 | 37.56 | 22.85 |
| LightRAG | 72.73 | 87.54 | 0.6365 | 0.3039 | 11.19 | 97.88 |
| GFM-RAG | 49.62 | 55.65 | 0.3597 | 0.1892 | 22.46 | 37.73 |
| LinearRAG | 84.48 | 87.49 | 0.6457 | 0.3367 | 56.81 | 32.40 |
| Naive RAG | 69.36 | 83.90 | 0.5520 | 0.2635 | 28.91 | 3.18 |

### Recorded indexing and query costs

Index times cover each original successful 609-document index. Query batch
wall time covers the full 2,556-query batch. Wall time per unit is inverse
throughput; it is distinct from mean individual query latency. These are
recorded execution costs, not controlled estimates of isolated serving latency.
Local method backbones remain declared.

| System | Index wall (s) | Index wall / document (s) | Query batch wall (s) | Batch wall / query (s) |
|---|---:|---:|---:|---:|
| Prehop | 2,504.84 | 4.11 | 10,607.27 | 4.15 |
| MS GraphRAG | 6,861.47 | 11.27 | 7,617.49 | 2.98 |
| LightRAG | 6,562.61 | 10.78 | 31,578.02 | 12.35 |
| GFM-RAG | 1,033.68 | 1.70 | 12,300.51 | 4.81 |
| LinearRAG | 638.60 | 1.05 | 10,583.70 | 4.14 |
| Naive RAG | 119.14 | 0.20 | 1,047.69 | 0.41 |

The Prehop query batch is continuous and not resumed. Index construction and
query execution are measured as separate phases.

### Evidence locations

These links open the final result JSON directly. Detail rows, traces and index
cost evidence remain alongside each run. No seed-exception record is required.

- [Prehop: original-query single-pass result](../data/results/prehop-original-query-20260910-multihoprag-prehop/prehop/multihoprag/seed_42/prehop_multihoprag.json)
- [Naive RAG](../data/results/multihop-benchmark-first-20260908-multihoprag-naive/naive/multihoprag/seed_42/naive_multihoprag.json)
- [MS GraphRAG](../data/results/rolling-two-models-20260908-multihoprag-ms_graphrag/ms_graphrag/multihoprag/seed_42/ms_graphrag_multihoprag.json)
- [LightRAG](../data/results/rolling-two-models-20260908-multihoprag-lightrag/lightrag/multihoprag/seed_42/lightrag_multihoprag.json)
- [GFM-RAG](../data/results/benchmark-first-two-20260909-multihoprag-gfm_rag/gfm_rag/multihoprag/seed_42/gfm_rag_multihoprag.json)
- [LinearRAG](../data/results/benchmark-first-two-20260909-multihoprag-linear_rag/linear_rag/multihoprag/seed_42/linear_rag_multihoprag.json)


## Derived evidence

The manuscript's output-size, literal-coverage, evidence-count, and QA-group
analyses cover all six completed systems, including the replacement LinearRAG
run. Their source paths, SHA-256 hashes,
calculation definitions, and derived values are recorded in
[completed metrics](../artifacts/paper_revision_20260909/completed_metrics.json).
These are calculations from saved outputs, not new retrieval runs.

| System | Records: min / median / max | Median words | Exact-fact Recall@10 (%) | AllFacts@10 (%) |
|---|---|---|---|---|
| Prehop | 12 / 12 / 12 | 1,706.5 | 69.73 | 41.64 |
| MS GraphRAG | 1 / 5 / 8 | 4,139.5 | 34.07 | 10.78 |
| LightRAG | 4 / 12 / 20 | 9,205.5 | 54.19 | 20.75 |
| GFM-RAG | 5 / 5 / 5 | 8,275.5 | 32.42 | 13.75 |
| LinearRAG | 5 / 5 / 5 | 7,930.0 | 55.28 | 25.14 |
| Naive RAG | 12 / 12 / 12 | 1,632.5 | 50.16 | 20.22 |

Size statistics use all 2,556 questions. Retrieval diagnostics use the same
2,255 evidence-bearing questions and the official case-sensitive,
space/newline-stripped matcher. These coverage measures are additional metrics,
not official leaderboard metrics or assessments of reasoning correctness.

Earlier Prehop-derived differences and coverage partitions are excluded from
current reporting. Retained analysis files are historical artifacts, not a
source for current Prehop comparisons. The Prehop entry in `completed_metrics.json`
now derives only from the new single-pass result and records its SHA-256 hash.
No earlier paired confidence interval is carried forward.

## HotpotQA (HippoRAG corpus)

No completed HotpotQA (HippoRAG corpus) results are recorded here. Table 4 in the manuscript reserves all seven
system rows for official Answer EM/F1, Supporting Fact EM/F1, Joint EM/F1, and
query latency. Use the declared 1,000-row / 9,221-passage release and completed artifacts before filling these cells. No fullwiki outcome is relabelled as a reduced-corpus result.

## Primary matrix

The supported methods and their identities are defined in
`core/strategy_registry.py`. Only completed full-run artifacts can supply
benchmark scores. An index or canary alone is not a completed benchmark.
Execution records errors and provenance without additional admission gates. Consult generated campaign artifacts for queue state.

## Prepared dataset identities

Each admitted artifact must bind the prepared corpus and full query file.
The query and metric denominators are defined in
[the cost and metric protocol](THROUGHPUT_EXECUTION.md#final-tables-and-measurement-definitions).
A different corpus fingerprint identifies different inputs and requires fresh
compatible evidence.

| Dataset | Source documents | Full queries | Corpus fingerprint |
|---|---:|---:|---|
| MultiHop-RAG | 609 | 2,556 | `c11b84f626c08d06d6dbc938512275824567aaffdc77a0f0b5424ad94f13a8ee` |
| HotpotQA (HippoRAG corpus) | 9,221 | 1,000 rows / 944 original IDs | `64595f5f19ef92699d584b42243ca3206028b830a1c85fde1e85ffcf805fecf1` |

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
The six retained runs have no terminal query failures; their
retrieval denominator is 2,255.

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

## Manuscript and presentation mapping

The manuscript reports MultiHop-RAG retrieval in Table 2, MultiHop-RAG costs
in Table 3, and unmeasured HotpotQA (HippoRAG corpus) outcomes in Table 4. Table 5 defines A/B/C; Tables 6–7 contain unmeasured result slots.
Tables 8–9 reserve HOP-usefulness and connection-timing outcomes; Tables 10–11
reserve paired timing summaries and automatic co-evidence connectivity. Tables 12–14 contain
fact-count, answer-type, and returned-length diagnostics; Table 15 reports generation settings.
Each system table includes the seven primary methods, with `—` for missing
outcomes. This register's completed-result tables include only finished runs.

The Markdown manuscript is the source for reported claims and result tables.
The manuscript PDF is generated from that source using the official ACL
review template; [build instructions](acl/README.md) describe regeneration.
The 27-slide presentation includes the completed single-pass Prehop results and
prepared HotpotQA corpus counts. The figures illustrate the method and do not
provide additional measured evidence.
Figure and export checks are maintained in [the checklist](PAPER_CHECKLIST.md).

## Publication conventions and result insertion

Use the fixed display order **Prehop, HopRAG, MS GraphRAG, LightRAG, GFM-RAG,
LinearRAG, Naive RAG**, retaining only applicable systems in completed-only
register tables. Do not reorder by score, speed, or completion status. Strategy
identifiers in code are unchanged. Use these display names without shortening
Naive RAG to Naive or substituting Microsoft GraphRAG for MS GraphRAG.

In paper tables, percentages and seconds use two decimals, MRR/MAP use four
on a 0–1 scale, and counts are integers unless explicitly averaged. Added
coverage and percentage-score differences are in percentage points (pp).
`—` means unmeasured, `N/A` means non-applicable, and measured zero remains
numeric. Source artifacts retain their original numeric precision.

| Incoming result | Paper destination | Dependent updates |
|---|---|---|
| Completed MultiHop-RAG system | Tables 1–3 and 12–14 | Sections 4.2–4.3, 5.1–5.2, Discussion, abstract/conclusion where affected |
| Completed HotpotQA (HippoRAG corpus) system | Table 4 | Section 5.3, corpus/support mapping checks, Discussion and scope statements |
| Representation A/B/C | Tables 6–7 | Section 5.4, paired differences and intervals, candidate counts |
| Query-conditioned HOP usefulness | Table 8 | Section 5.4, eligible/nonempty/failed-query counts and NEXT overlap in A.3 |
| Precomputed versus online matching | Tables 9–10 | Section 5.4 and A.3, destination agreement, paired intervals and preparation cost |
| Blinded stored-link labels | Table 11 | Section 5.4 and A.4, sample sizes, unclear cases and adjudicated labels |

Before insertion, match the result's method, dataset, query IDs, evaluated
configuration, completion/failure accounting, and source artifact. Do not use
primary Prehop scores as condition A, or mix partial-query results with full-run
rows. Replace the matching row, update every dependent interpretation and
completion statement, then rebuild the PDF and inspect its page budget and
numeric column widths. The benchmark execution environment is not part of this
publication workflow.

## Paired uncertainty for current completed MultiHop-RAG results

The current Prehop output was aligned with each completed comparator by query
ID, original question, and gold facts (2,255 evidence-bearing queries). The
analysis uses 10,000 paired percentile bootstrap resamples with seed 42.
AllFacts@10 is recomputed from the first ten returned passages with the declared
case-sensitive space/newline-stripped matcher. Intervals are unadjusted and
conditional on these saved outputs; they do not measure run-to-run variation
or isolate the effect of precomputed links.

| Comparator | Prehop minus comparator MAP@10 [95% CI] | AllFacts@10 difference, pp [95% CI] |
|---|---|---|
| MS GraphRAG | 0.2617 [0.2509, 0.2726] | 30.86 [28.82, 32.95] |
| LightRAG | 0.1446 [0.1338, 0.1556] | 20.89 [18.76, 23.02] |
| GFM-RAG | 0.2593 [0.2475, 0.2708] | 27.89 [25.76, 30.02] |
| LinearRAG | 0.1118 [0.1008, 0.1232] | 16.50 [14.28, 18.76] |
| Naive RAG | 0.1850 [0.1765, 0.1934] | 21.42 [19.56, 23.33] |

Derived values and source hashes: `artifacts/paper_revision_20260909/current_paired_bootstrap.json`.

## Primary-run link utility from saved candidates

The complete pre-selection candidate traces of the current primary Prehop run
were joined to final returned passage IDs. Payload hashes and candidate IDs
were verified. These diagnostics do not fill the A/B/C result slots.

| Measure | Current primary Prehop | Population |
|---|---|---|
| Mean direct candidates | 23.16 | 2,255 evidence-bearing queries |
| Mean HOP destinations | 24.70 | 2,255 evidence-bearing queries |
| HOP destination relevance (%) | 2.14 | 2,255 queries with nonempty HOP sets |
| Added gold coverage (pp) | 3.54 | 2,255 evidence-bearing queries, including zero gain |
| Retained added coverage (pp) | 2.72 | Same population; final selection up to 12 passages |
| Retained gain also reachable through NEXT (pp) | 0.52 | Same population |
| Queries gaining gold evidence | 215 (9.53%) | 2,255 evidence-bearing queries |
| Queries retaining additional gold evidence | 164 (7.27%) | 2,255 evidence-bearing queries |

HOP sets include direct overlap; added coverage excludes facts already found by
direct retrieval. All fractions are query-macro, with the declared literal
matcher and distinct gold facts. These are not top-ten AllFacts scores or
causal estimates of disabling HOP. Low evidence relevance does not by itself classify a link as semantically incorrect. Automatic co-evidence analysis measures structural reachability; precomputation's latency effect requires the matched online arm.

Source hashes, populations, and query-level values:
`artifacts/paper_revision_20260909/current_link_utility.json`.

## Controlled connection experiments

A/B/C, automatic co-evidence/null analysis, and Neo4j stored-versus-online
comparisons have no completed paper outcome yet. The experiment contract is
[PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md). Common-input preparation and
service probes are implementation evidence, not speedup or quality results.
Use campaign artifacts for current dispatch state. Frozen-prefix A/B/C latency
is downstream-only; it must not replace primary end-to-end latency.
