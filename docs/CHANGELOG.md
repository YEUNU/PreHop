# Changelog

This file is the chronological engineering record. `ARCHITECTURE.md` defines
current behavior, `README.md` is the user-facing guide, `CLAUDE.md` defines
maintenance policy, and the local `prehop_paper.md` owns paper prose and claims.
Entries are newest first. Older entries describe behavior at that point in
development and may be superseded; they are not a current configuration guide
or a source of paper claims.

## 2026-08-26 — Paper and architecture synchronized to the fixed configuration

The manuscript now describes the selected configuration without promoting its
sample-200 development result to a paper claim: legacy Q−/Q+, owner-wide
activation, materialized reciprocal provenance, full depth-one NEXT/HOP
traversal, and no rewrite. The method is fixed pending the
pre-declared confirmatory evaluation. The primary hypothesis explicitly
requires improvement over representation-matched NEXT-only/no-HOP controls,
not only over external systems.

The approved main ablation is now the cumulative A0–A5 ladder: body only,
questions without graph, NEXT only, raw HOP, reciprocal HOP, and exact
activation. Adjacent paired contrasts isolate each addition; online/offline
reciprocal is classified separately as an implementation equivalence and
efficiency study. `ARCHITECTURE.md` now expresses the same current defaults in
implementation terms and removes P1/P2 experimental shorthand. No benchmark,
index, or code behavior changed in this documentation-only update.

A repository-wide Markdown audit also removed duplicated operating details,
corrected stale Exact-Q+ default descriptions, distinguished hypotheses from
established claims, and reduced private submission history. The vendored
HopRAG guide was otherwise preserved; only its local supported-corpus count was
corrected.

`CLAUDE.md` now requires each document to retain its audience-specific tone and
prohibits unnecessary proper nouns, invented management labels, repeated
slogans, and unexplained experimental shorthand. Actual dataset, system, code,
configuration, and established research names remain available where needed
for precision and reproducibility.

## 2026-08-26 — Development-winning P2 default and traversal specialization

The current operational defaults reproduce the selected legacy sample-200
query-time result: `RAG_QUESTION_SCHEMA=legacy`, owner-wide Q+
activation, materialized reciprocal filtering, full NEXT/HOP traversal, and no query
rewrite. Both `core/config.py` and the project `.env` specify that contract;
P1 exact activation, the unfiltered graph, and P1+P2 remain explicit
ablations. This default was selected on the development sample and is not a
held-out paper result.

Traversal now emits only the configured NEXT/HOP and HOP-filter Cypher
branches. The previous query sent inactive `none`, `reciprocal`, and
`reciprocal_offline` branches together and guarded them with runtime
parameters, increasing query text and planner work. Candidate construction,
ranking, reciprocal semantics, and the stored graph remain unchanged.

The legacy graph was augmented in place without changing its 21,843 HOP edge
count. Materialization stored 2,772 reciprocal provenance pairs in 284.7 s;
fresh indexes now enable this step by default. The online implementation
remains available for equivalence and ablation checks.

The optimized offline default completed the same sample-200 with zero errors.
Every per-query deterministic retrieval metric matched online P2. Average
traversal fell from 478.0 ms to 167.1 ms (-65.0%) and end-to-end latency from
3.681 s to 3.408 s (-7.4%). Answer F1 is shown below for completeness but is
not used to establish implementation equivalence because synthesis was not
seeded; retrieval/evidence metrics are deterministic.

All rows below are completed artifacts with the same 200-query digest
`d5facf87...`. Official ranking and evidence aggregates have 150 eligible
non-null queries. Prehop and Naive use top-k 12; HopRAG retains official top-k
20 and MS GraphRAG retains its official LocalSearch context budget, so latency
and resource-cost comparisons are descriptive rather than equal-budget.

| System | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Doc P | Doc R | Doc F1 | Fact R@10 | Answer F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Prehop P2 offline** | **0.7267** | **0.8600** | **0.5679** | **0.2770** | **0.4331** | 0.7000 | **0.5122** | **0.5822** | 0.1472 | 3.408 s |
| Naive RAG | 0.6400 | 0.8533 | 0.5307 | 0.2525 | 0.4247 | 0.7117 | 0.5013 | 0.5444 | 0.1421 | **2.929 s** |
| HopRAG | 0.6533 | 0.8200 | 0.5366 | 0.2715 | 0.3357 | **0.7900** | 0.4442 | 0.5278 | **0.1551** | 153.163 s |
| MS GraphRAG | 0.4267 | 0.4867 | 0.2841 | 0.1395 | 0.2390 | 0.3967 | 0.2885 | 0.5722 | 0.0105 | 17.652 s |

Benchmark index-manifest lookup now checks the artifact payload's exact corpus
tag. Previously the `multihoprag` filename glob could select a newer
`multihoprag_grounded_v1` artifact because one tag prefixes the other; this did
not change the queried Neo4j labels, but it recorded the wrong index policy in
the result provenance.

The complete suite passes with 232 tests and 4 skips; Ruff and whitespace
checks also pass.

## 2026-08-26 — Grounded schema result and reciprocal query-path cleanup

Added the opt-in `grounded_v1` question schema without changing the legacy
default or existing `multihoprag` index. Q− stores a verbatim answer, source
quote, and anchors; Q+ stores a source quote, anchors, and missing information.
Every stored field is checked against source text. An invalid generated record
is discarded while valid siblings are retained, so one bad record cannot fail
an entire document. The integrated gate verifies all stored grounding fields.

A cold full-corpus build under the separate `multihoprag_grounded_v1` tag
completed 609/609 documents with zero errors and all integrity checks passing.
It stored 8,529 chunks, 22,088 Q−, 18,999 Q+, and 18,277 HOP edges. Compared
with the legacy graph's 21,843 HOP edges, strict grounding reduced HOP density
by 16.3%. Indexing took 5,459 s, including 454 s for HOP construction.

The reciprocal Q−→Q+ nearest-neighbor check can now be materialized once at
index time. This build stored 2,455 reciprocal provenance pairs;
`reciprocal_offline` reads those IDs without query-time reverse ANN. A live
200-pair comparison matched the original rule 200/200 with zero mismatches.
Graph search also skips a pre-expansion scoring pass that was always superseded
after traversal. Both changes preserve ranking semantics.

The grounded sample-200 ablation used identical query IDs, models, top-k 12,
disabled rewrite/judge, and zero runtime errors:

| Variant | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Doc P | Doc F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 exact | 0.7133 | 0.8267 | 0.5406 | 0.2640 | 0.3861 | 0.4821 | 3.81 s |
| P2 owner + reciprocal offline | 0.7000 | 0.8400 | 0.5415 | 0.2647 | 0.4146 | 0.4990 | 3.74 s |
| P1+P2 exact + reciprocal offline | 0.7000 | 0.8400 | 0.5414 | 0.2647 | 0.4157 | 0.4999 | 3.72 s |

Relative to grounded P1, P1+P2 significantly improved document precision by
0.0296 (95% CI [0.0142, 0.0460]) and document F1 by 0.0178
([0.0048, 0.0314]); ranking-metric intervals included zero. P1+P2 and P2 were
effectively identical. All grounded variants were below the legacy leading
reciprocal result (0.7267/0.8600/0.5679/0.2770), so `grounded_v1` is rejected
as Primary and remains an auditable development ablation. These are sample
results without a complete corpus manifest, so the final paper configuration
remains on hold rather than being tuned on this exploratory sample.

The paired-bootstrap tool now identifies the dataset from the artifact's
`dataset` field rather than assuming a custom corpus tag is the dataset name.
The full suite completed with 231 passed and 4 skipped.

## 2026-08-26 — Exact Q+ activation and Primary-selection hold

Query-time Q+ retrieval now preserves the exact question-node IDs returned by
vector and full-text search when results collapse to owner chunks. Traversal
admits a stored `HOP_ANSWER` only when those IDs intersect the edge's
`source_question_ids`; bridge embeddings and emitted path provenance use the
same intersection. Reciprocal filtering applies after this exact activation.
This is a read-only provenance correction: the full existing index, indexes,
and all 21,843 stored HOP edges remained unchanged. Static checks and the full
suite completed with 221 passed and 4 skipped.

All following runs used the same MultiHop-RAG sample-200 query digest, full
existing corpus index, top-k 12, generation and embedding models, disabled
query rewrite, disabled judge, and zero runtime-error rows. They are
exploratory development evidence, not results on the complete prepared split.
Exact Q+ alone
(`prehop_exact_qplus_p1_20260826`) selected 189 HOP paths and produced
Hits@4 0.7267, Hits@10 0.8467, MRR@10 0.5657, and MAP@10 0.2755. Exact Q+ plus
reciprocal filtering (`prehop_exact_qplus_reciprocal_p2_20260826`) selected 23
HOP paths and produced 0.7267, 0.8533, 0.5672, and 0.2768 respectively.
Relative to Exact alone, paired bootstrap differences were +0.0015 for MRR
(95% CI [0.0001, 0.0035]), +0.0013 for MAP ([0.0003, 0.0024]), +0.0375 for
document precision ([0.0234, 0.0530]), and +0.0267 for document F1
([0.0161, 0.0382]). Exact plus reciprocal is therefore the best HOP-enabled
query-time candidate tested here.

Causal controls prevent promoting that candidate to the final Primary. With
the same code and index, `NEXT`-only produced Hits@4 0.7267, Hits@10 0.8533,
MRR 0.5673, MAP 0.2766, and latency 3.39 s, while the HOP-enabled candidate
produced 3.69 s latency. Their official-ranking differences all had intervals
spanning zero, but HOP-enabled document precision and F1 were lower by 0.0097
and 0.0098 with intervals excluding zero. The approved no-graph control
produced Hits@4 0.7267, Hits@10 0.8600, MRR 0.5630, MAP 0.2795, and higher
fact recall; none of its official-ranking differences versus the HOP-enabled
candidate excluded zero. The sample therefore does not establish a causal HOP
benefit. Final Primary selection remains on hold until the pre-declared
grounded Q−/Q+ schema is rebuilt over the full corpus and evaluated without
adding sample-tuned thresholds or heuristics.

## 2026-08-26 — Deterministic modality ranks and rejected graph corroboration

An additional read-only reciprocal-link ablation was evaluated without
rebuilding or mutating the existing Prehop index. For an activated stored
Q+→Q− provenance pair, traversal retained the HOP only when the target Q−
reverse-retrieved that exact source Q+ as its best cross-document Q+
representation. The reverse ANN pool is determined by target-document Q+
cardinality plus one; the ablation has no similarity threshold or fitted
candidate width. The sample-200 run (`prehop_reciprocal_20260826`, 200/200,
zero errors, judge disabled) reduced selected HOP occurrences from 220 to 27.
Relative to `prehop_order_only_20260826`, Hits@4 was unchanged, Hits@10 changed
by +0.0133 (95% CI [0.0000, 0.0333]), MRR@10 by +0.0021
([0.0003, 0.0046]), MAP@10 by +0.0016 ([0.0005, 0.0029]), document precision
by +0.0359 ([0.0229, 0.0503]), and document F1 by +0.0246
([0.0144, 0.0355]); document recall changed by -0.0006 with an interval
spanning zero. This is retained only as an explicit query-time ablation because
it was identified on the same exploratory sample and the artifacts lack a
complete corpus manifest. It is not included in the paper manuscript.

Prehop now explicitly sorts vector and full-text rows by descending
within-modality backend score with a stable node-ID tie break before applying
equal reciprocal-rank fusion. Neo4j procedure, aggregation, or `UNION ALL`
return order is no longer an implicit ranking input. Backend scores remain
isolated within each modality and are not interpolated, so this is a
correctness/reproducibility fix rather than a new weight or tuned parameter.

An isolated sample-200 check (`prehop_order_only_20260826`, 200/200 rows,
zero errors, judge disabled) reproduced the accepted checkpoint's Hits@4,
Hits@10, MRR@10, fact-recall, and document diagnostics exactly. MAP@10 changed
by -0.0003 with a paired 95% interval spanning zero. The run is exploratory
and was produced from a dirty development tree, so it is validation evidence,
not a replacement paper result.

A second development variant allowed an already-direct candidate to receive
additional NEXT/HOP rank evidence. Relative to the accepted checkpoint it
increased document precision by 0.0494 (95% CI [0.0321, 0.0674]) and document
F1 by 0.0355 ([0.0195, 0.0517]), but decreased MAP@10 by 0.0116
([-0.0209, -0.0027]). Hits@4 changed by +0.0133 with an interval spanning
zero. Because the primary official ranking metric regressed significantly,
the corroboration behavior was rejected and is not present in the current
query path.

A subsequent read-only graph diagnosis separated routing from endpoint
quality on the same 150 non-null sample queries. All gold facts resolved to
indexed chunks. Although at least one gold-document pair had a HOP edge for
114/150 queries, at least one exact gold-fact-owner pair had a HOP edge for
only 21/150. A query-only variant therefore treated HOP as a document route
and selected one target chunk per offline edge inside Neo4j. It increased
selected HOP-path occurrences from 220 to 1,013, but changed neither Hits@4,
Hits@10, nor MRR@10; MAP@10 changed by only +0.00004 with an interval spanning
zero. Among eligible queries, 118 selected at least one localized HOP target,
44 selected a gold target document, and only two selected an exact gold-fact
chunk through HOP. The variant was rejected and removed. In this sample,
query-activated edge precision was a more plausible bottleneck than
target-document chunk localization; this was not a general error decomposition.

## 2026-08-26 — Parameter-free representation fusion and baseline integrity

Prehop now preserves reciprocal-rank evidence when Q−, body, and Q+ owner
lists are merged. Final selection combines the resulting representation order
with the body/bridge semantic order using equal reciprocal ranks; it does not
interpolate backend scores or introduce a fitted channel weight. NEXT targets
inherit total source representation evidence and HOP targets inherit only Q+
evidence, both attenuated by reciprocal path length. With the required
one-edge traversal this is a structural factor of one half, not a swept
hyperparameter. This prevents graph-discovered nodes from tying directly
retrieved owners solely through inheritance.

The independent MultiHop-RAG sample-200 development check completed once for
each of Prehop, Naive, official HopRAG, and official MS GraphRAG on the same
query-ID digest, with the optional judge disabled and no runtime-error rows.
The checkpoint is exploratory because its corpus manifest is absent and it is
not the complete 2,556-query split. It therefore validates implementation
direction but is not a submission result. Paired analysis showed that the
then-current Prehop revision improved the pre-fusion checkpoint on Hits@4, MRR@10,
MAP@10, and diagnostic fact recall@4, while diagnostic document F1 remained
lower; the paper claim remains scoped to official evidence ranking and
discloses the document-level tradeoff.

HopRAG provenance recovery now marks exact text shared by multiple source
documents as ambiguous instead of awarding an arbitrary title, and its sync
adapter reuses one event loop per worker thread to prevent descriptor growth.
MS GraphRAG provenance continues to follow official text-unit IDs through
document IDs to staged filenames. Its adapter-owned keyword router was removed:
the baseline now consistently invokes the official LocalSearch API. Full tests
cover both provenance paths and the fixed API selection.

## 2026-08-25 — Independent runs and answer-owner HOP construction

Removed the aggregate matrix runner and its merge/accounting tests. Dataset
and strategy targets now run independently with isolated logs and explicit run
identities. Index artifacts record code provenance and resolved semantic
settings, while throughput controls remain separate runtime controls.

Prehop HOP construction now resolves each Q+ through its nearest
cross-document Q− and follows that question's owner relation to the evidence
chunk. This removes the redundant body ANN pass, top-1 channel intersection,
constant aggregate edge score, and all HOP width/weight/threshold settings.
The no-Q− indexing ablation resolves Q+ directly to a body chunk. Query-time
selection uses global cosine order by default; source round-robin remains an
explicit ablation. The integrated index gate verifies the corresponding
question-level provenance before publishing a completed snapshot.

Generation request limits are shared by clients using the same endpoint and
event loop, with in-flight/peak logging. Prehop bounds generation fan-out by
processing chunks in source order within each concurrently active file. Graph
files now enter a bounded rolling scheduling window, whose completed slots are
reused immediately. This removes the full-batch wait caused by a single long
document without creating an unbounded task queue.

## 2026-08-24 — Matrix barrier, capacity, and continuation correction

The paper matrix runner now executes strict strategy barriers in the requested
order `ms_graphrag → hoprag → naive → prehop`; all selected datasets in a
strategy phase finish before the next strategy begins. Adapter-specific
generation settings are clamped to one shared per-target budget, so MS
GraphRAG's request semaphore and HopRAG's document-worker/thread product are
included in the aggregate `max_num_seqs=120` guard.

An interrupted matrix can be continued in the same run directory with
`--resume --run-id <run-id>`. Progress ETA includes pending datasets and future
strategy barriers, and the default human-readable watch interval is one hour.
Attempt wording is explicit: `--target-attempts 2` means two total pipeline
attempts, while `RAG_HOP_INTERNAL_RETRIES=2` means two total official-call
attempts. The vendored HopRAG history is retained; this correction does not
rewrite historical third-party changes.

The continuation path now records SIGINT/SIGTERM interruptions as resumable
attempt fragments, uses strategy-specific durations from prior matrix results
for cold-start ETA estimates, and records MS GraphRAG's dropped-relationship
integrity warnings in each target result. HopRAG's batched edge writes use
sequential node matches to avoid Neo4j cartesian-product warnings. Naive timing
is documented as aggregate-only because its adapter does not expose finer
phase boundaries.

## 2026-08-24 — Dataset suite selection and resumable matrix accounting

The paper indexing matrix uses MultiHop-RAG and MuSiQue. The larger auxiliary
QA corpus was removed from the repository and active experiment because its
closed-corpus indexing cost was disproportionate to the comparison.

The measured runner now enforces a 120-sequence capacity, records live
`progress.json` snapshots with phase and ETA information, persists interrupted
attempt fragments in the run directory, and provides
`scripts/merge_index_matrix_runs.py` for cumulative timing across stopped and
resumed runs. The matrix adapter changes in this release do not rewrite the
vendored third-party tree; historical vendored changes remain part of the
repository history and are documented separately.

## 2026-08-23 — Query-time latency cleanup and candidate-pool retuning

A multi-angle pass over the query path (hybrid.py/retrieve.py/scoring.py/
traversal.py, prompted by "is there more to fix here") found three pieces of
pure wasted work, unrelated to the diversity-cap fix above. Stage 1 of
`retrieve.py` scored and selected its own candidates every query even though,
in the default `full` variant, Stage 2 always runs next and its own final
scoring pass immediately superseded that result — an entire embedding-
similarity round trip over up to 72 candidates, discarded unused, every
single query. `core/vllm_client.py` had no embedding cache, so the same query
string was independently re-embedded up to seven times per retrieval call
across hybrid.py's RRF channels and scoring.py's body/bridge passes; a
client-instance-scoped cache (keyed by exact text, gated to single-item
`encoding_type="query"` calls only, so document batches are never touched)
collapses these to one real network call per distinct string. Independent
channel fetches inside one stage (Q-/body, Q+/Q- support, and hybrid.py's own
vector/fulltext pair, the last moved onto two Neo4j sessions) were sequential
`await`s and now run concurrently via `asyncio.gather`. None of this changes
ranking or selected results — confirmed by the existing test suite passing
unchanged and by these being pure-function/dead-code fixes, not algorithm
changes.

Candidate-pool widths that fed Stage 1/Stage 2/traversal (`top_k * 6/8/4`
literals) were config-driven into `RAG_CANDIDATE_LIMIT_MULTIPLIER`,
`RAG_SUPPORT_POOL_MULTIPLIER`, `RAG_STAGE1_POOL_MULTIPLIER`, and
`RAG_WIDE_POOL_MULTIPLIER` (the last shared by `retrieve.py`'s Stage 2 cap and
`traversal.py`'s `candidate_budget`, both representing the final wide pool
handed to scoring) and re-swept with the same 60-query multihoprag fact_recall
methodology used for the diversity cap. `RAG_WIDE_POOL_MULTIPLIER` moved from
8 to 6: fact_recall improved (0.611 → 0.619, matching the 12x result) while
avg_traversal_ms dropped about 25% (1026ms → 774ms) — a pool that wide was
adding latency without adding evidence quality. The other three multipliers
showed no change beyond already-observed run-to-run noise and kept their
original values.

These fixes are query-time only and do not touch or affect the concurrently
running full-matrix indexing job (`full_20260823_routed`): that process
already had the pre-fix code loaded in memory when it started, indexing never
imports the retrieval-package mixins these changes touch, and the one shared
file (`vllm_client.py`) only gained a cache that returns numerically identical
embeddings for repeated text, never a value change.

## 2026-08-23 — Per-source diversity cap in final evidence selection

The incremental single-seed traversal (previous entry) had two live bugs.
`_expand_frontier`'s Cypher put a bare `WHERE` directly after a `CALL (src)
{ ... }` subquery block, which Neo4j's grammar rejects; it needed an explicit
`WITH src, related, path_type, edge_score` in between. The seed loop also
skipped a candidate seed once it appeared in `discovered_ids`, not just
`expanded_ids` — so a seed only passively swept up as another (earlier,
same-document) seed's NEXT-neighbor never got its own expansion turn and lost
its own HOP_ANSWER edges, even when it was the one chunk carrying a path to a
second gold document. Fixed by gating the skip on `expanded_ids` alone.

Neither bug fix alone changed retrieval outcomes on a 15-query dev check.
Manual inspection of the two tracked loss cases found the real driver was
downstream: `_score_and_select`'s final top-k was pure global score order, so
several near-duplicate high-scoring chunks from one strongly-relevant
document routinely filled most of the evidence slots and crowded out the
only chunk from a second, lower-scoring gold document. `scoring.py` now caps
each source at `RAG_MAX_CHUNKS_PER_SOURCE_FRACTION` (default 0.34) of top_k
in final selection (`floor(top_k * fraction)`, minimum 1); capped-out
candidates still backfill by score if too few distinct sources exist to fill
top_k. A dataset-structure check found that all 60 development queries had
gold evidence spanning at least two documents (mean 2.4). This motivated the
experiment but did not rule out sample-specific tuning.

Tuning used `fact_recall` (does retrieved chunk text actually contain the
gold evidence fact), not `doc_recall` (does the gold title merely appear
anywhere in results) — doc_recall turned out gameable: capping at 1 chunk
per document pushed doc_recall to 0.903 but dropped fact_recall to 0.531,
below the 0.589 earlier-baseline (pre-fix, pre-incremental-traversal) figure,
because forcing maximal spread away from the strongest passages reduces the
odds of landing on the specific relevant one. 0.34 gave both doc_recall
0.787 and the best fact_recall 0.619. A follow-up sweep of
`MAX_CHUNKS_PER_SOURCE_FRACTION`, `GRAPH_HOP_DEPTH`, `GRAPH_SEARCH_LIMIT`,
`HOP_LINK_LIMIT`, and `HOP_SAME_NEED_WEIGHT` produced no better setting on
that development sample, so no further change was made. A 10,000-resample
paired bootstrap over the 60-query fact_recall
delta (+0.031, 95% CI [-0.014, +0.076]) does not yet exclude zero — the
improvement is directional, not proven significant at this sample size.

A 200-query development benchmark on MultiHop-RAG with the fix and the
0.34 default (LLM judge excluded from this figure set — same served model
as generation, so self-judged and unreliable): avg_doc_match 0.735,
avg_evidence_doc_recall 0.582, avg_hits@10 0.432, avg_hits@4 0.304,
avg_map@10 0.261, avg_mrr@10 0.454, avg_answer_attempted 0.64. Latency
breakdown: avg_retrieve_ms 2598, avg_traversal_ms 1716, avg_synthesis_ms
9211, avg_latency 13525.

## 2026-08-23 — Explicit routed-server capacity and retry backoff

The matrix runner no longer assumes that identical OpenAI-compatible URLs mean
generation and embedding share one accelerator scheduler. An explicit
`RAG_INFERENCE_CAPACITY_MODE` records routed-server topology, with separate
generation and embedding `max_num_seqs` budgets. Generation capacity now uses
the generation-heavy width rather than all matrix slots. Resource retries also
halve the target's global LLM semaphore in addition to Prehop file, HopRAG
adapter, and MS GraphRAG adapter concurrency. The 128-sequence full-run profile
therefore starts at 120 generation calls with eight sequences of headroom; no
official baseline implementation is changed. Cold graph reset now drops stale
application schema first and deletes nodes in bounded transactions, halving
the batch on Neo4j transaction-memory exhaustion.

## 2026-08-22 — Individual question graph and evidence-directed HOP/NEXT

Q-/Q+ are now individual nodes with document-scoped embeddings rather than
concatenated chunk properties. Every Q+ searches cross-document Q-, body, and
Q+ channels; Q-/body matches are required for `HOP_ANSWER`, while Q+ same-need
similarity is provenance/support only. Rank fusion replaces the obsolete 0.82
cross-encoder-era cosine threshold. `NEXT` remains stored in document order and
is traversed bidirectionally; `HOP_ANSWER` is traversed only toward evidence.
Re-indexing a document atomically replaces its old chunk/question subgraph.
Files removed from a corpus snapshot are pruned before a new run, and Naive
now atomically replaces all chunks belonging to a changed/shortened source.
Prehop no longer creates an empty document record before its embeddings have
passed count/dimension validation.

ANN over-fetch is now sized per source (`own representations + 15`, floor 50)
instead of using the longest document for every request. Generation concurrency
is enforced once per target event loop, embeddings use loop-local semaphores for
HopRAG's threaded hooks, and the `max_num_seqs=128` capacity is validated. The
unused threshold sweep, local-model/reranker loaders, and upstream end-to-end
reranker script were removed.

The matrix capacity planner now treats identical generation/embedding URLs as
one shared `max_num_seqs` budget and separate URLs as independent budgets. It
samples both queues and automatically reduces effective client concurrency
before launch; official HopRAG's synchronous embedding hook now obeys the same
batch/request limits and validates every returned vector.
After observing Prehop throughput drop from about 7 to 2–3 documents/minute
when HopRAG joined the same generation endpoint, the matrix scheduler now caps
generation-heavy targets at one and fills spare width with Naive targets from
later datasets.
Naive now embeds and writes 32 documents at once. A live 64-document HotpotQA
probe improved throughput from roughly 3 to 47.1 documents/s with exact source,
chunk, and embedding-dimension integrity. The Prehop file in-flight default is
16 (still bounded by the 30-request generation semaphore) to avoid
under-utilizing the endpoint on one-chunk corpora.
Because generation-heavy targets are now serialized, HopRAG's validated
document-worker setting is restored to 10 (4 chunk threads, upper bound 40)
and the matrix uses 32 MS GraphRAG requests instead of its former conservative
8; both stay within the shared max-seq 128 budget.

Judge hallucination is no longer inferred from an incorrect-answer score.
OpenAI Batch reconciliation requires both explicit fields for every row and
keeps the manifest/result untouched when either field is missing.

Graph traversal now begins with the same 12 seeds as depth-0 retrieval. The
old `top_k-1` combined with a graph-search cap of 10 silently removed two flat
candidates before any edge was evaluated, which made the depth comparison
invalid and caused avoidable gold-document losses.

## 2026-08-22 — Dataset-neutral Q⁺ storage

Removed the post-generation Q⁺ heuristic gate entirely. All benchmark
datasets now store the non-empty, deduplicated Q⁺ strings produced by the
same shared prompt and validated JSON schema; no domain-specific metric,
period, statement, or keyword rule affects graph construction.

## 2026-08-22 — HopRAG empty-output recovery

The full cold matrix exposed an official HopRAG question-generation response
with a valid JSON shape but an empty question list. That case now receives
three bounded retries of the same official generation step. A still-empty,
blank, or malformed paragraph result uses the upstream empty-list skip with an
explicit warning and a measured skip count; if every paragraph is skipped,
the resulting empty document still fails the target. No alternative indexing
or retrieval method is substituted.

## End-to-end consistency and fail-loud completion (2026-08-22)

Completed the repository-wide audit beyond Prehop's mixins. Missing dataset
or query inputs now raise and exit non-zero; the MultiHop-RAG wrapper now
points at the real sample200 file; the deleted report-tool invocation and
unneeded reranker/Neo4j server startups were removed. Indexing restores a
failed Neo4j batch for idempotent replay, persists its failure artifact, and
then exits non-zero on any file/flush/HOP failure. HopRAG no longer substitutes
plain vector retrieval when official traversal fails, and shared embedding /
reranker paths reject missing vectors or malformed score responses instead of
producing zero scores. The same validation now covers Prehop's sparse Q-/Q+
embedding batches, HopRAG's required JSON keys, and the optional MS GraphRAG
dense-retrieval helper.

The unused MS GraphRAG dense-retrieval helper was subsequently removed: the
benchmark adapter now exposes only the official local/global GraphRAG search
APIs, preventing its old `top_k=5` auxiliary default from being mistaken for
an experimental retrieval setting.

LLM JSON handling now distinguishes transport failures from parse failures and
validates required keys/types at every Prehop call site. Prehop, Naive, and
HopRAG share one synthesis prompt and one empty-context abstain policy; MS
GraphRAG retains synthesis inside its official API. The configured judge model
is now fixed for a run (no per-row fallback model), partial Batch API outputs
retain their reconciliation manifest, benchmark aggregates use the union of
numeric fields across rows, and any per-query runtime error marks the artifact
`completed_with_errors` and makes the command exit non-zero. `benchmark_all`
still attempts every strategy before reporting their combined failure, and the
main result JSON remains slim while full traces stay in the dedicated report
artifacts. Optional debug output now respects `--save-intermediate`, and
source-sweep smoke tests exclude the local virtualenv/third-party/data trees.
Dataset wrappers validate query files only for stages that actually benchmark,
so an index-only run is not coupled to preparation of an unused query sample.

Run diagnostics are now isolated by `RAG_RUN_ID`: intermediate JSON uses
`data/debug/<run>/<strategy>/<corpus>/<source>/`, index logs use
`logs/index/<run>/<strategy>.log`, and failure/stat filenames carry the same
run ID. `--clear-graph` executes once rather than racing in every parallel
strategy or wiping a strategy indexed earlier by a dataset wrapper. Removed
the old shared-layout debug files, logs, obsolete source-less chunk cache,
Python/test/linter caches, stale exception log, and generated egg metadata
(about 309 MB total); the removal was moved to the desktop trash for recovery.
Also deleted unreferenced remnants of the previous agent/OCR architecture
(`core/schemas.py`, Prehop schemas/trace helpers, tool definitions), the old
HypoReflect migration cleanup script, and the duplicate dedicated MultiHop
sampler now covered by `data/make_sample.py`.

## Full-corpus exhaustive scan (not sampling) after a deep content audit (2026-08-22)

This audit inspected real Q-/Q+/chunk content in addition to counting known
patterns. Two outcomes:

- A manual check of roughly 20 records across the then-active datasets found
  the expected Q−/Q+ roles in those examples. This spot check was diagnostic,
  not evidence of corpus-wide generation quality.
- **A full-corpus frequency scan (not sampling) found more residual
  HotpotQA/MuSiQue markup** the earlier wiki-markup fix's narrow
  `<nowiki>`/`<br>`/`[[...]]`/`{{...}}` patterns missed: ruby-annotation
  tags (`<ruby>`/`<rb>`/`<rt>`/`<rp>`, Chinese/Japanese pronunciation
  guides), `<ref>`, `<a href="...">`, `<onlyinclude>`, `<section begin=...
  />`, `<small>` (293 occurrences across 90 hotpotqa files) — generalized
  `clean_wiki_markup()`'s tag-handling from an enumerated list to a single
  catch-all `<[^>]+>` strip (tag removed, inner/surrounding text kept;
  `<br>` still special-cased to a space so words don't get joined). Also
  added stripping of bare citation markers (`[12]`) and stray `[[`/`]]`
  left over from malformed/mismatched-bracket wikilinks the primary
  wikilink regex doesn't match. Applied to both `prepare_hotpotqa.py` and
  `prepare_musique.py` identically. musique_corpus was already fully clean
  even before this fix (0 occurrences) — applied there anyway since it
  shares the same Wikipedia source and could hit the same issue on a
  different data slice later.
- One thing investigated and *not* fixed: ~30 hotpotqa paragraphs are
  genuine Wikipedia disambiguation-page stubs ("X may refer to: ..." with
  no real facts) — this is the source page's actual content, not
  corruption, and is exactly the kind of low-information distractor
  paragraph HotpotQA's own distractor-config design already expects.
- The ongoing 15-minute background quality monitor was upgraded from
  `ORDER BY rand() LIMIT 30` sampling to a full-population scan (`MATCH
  (c:...) RETURN count(...)` with no LIMIT) across all three corpus tags,
  per the same "sampling misses things" lesson — a 70-count full-population
  hit on multihoprag right after this change was manually verified as a
  false positive (live-blog-format articles like "MLB Winter Meetings
  tracker" genuinely cite "Source: X" inline per update, as normal
  journalism convention — not a scraper artifact, left alone).

## MultiHop-RAG scraper boilerplate in article bodies, found live (2026-08-22)

Third data-quality bug caught by the same periodic Neo4j quality sampler,
right after the two below — ~9% of MultiHop-RAG articles (54/609) had
newsletter-subscription UI text scraped into the article `body` field
itself in the source `corpus.json`, predating any of this project's own
code. Two structurally distinct patterns by outlet:
- Independent-sourced articles: a Mustache-template signup widget
  (`{{ #verifyErrors }} ... {{ /verifyErrors }} ... {{ /verifyErrors }}`,
  closing tag always exactly twice) at the very start of body, preceded by
  section-specific opening copy that varies too much to pattern-match
  directly ("Sign up to Simon Calder's...", "Stay ahead of the trend...",
  etc. depending on section) — so the fix anchors on the position (start of
  body) and the two `{{ /verifyErrors }}` closing tags instead, not the
  opening phrase.
- Guardian-sourced articles: text between `skip past newsletter promotion`
  and `after newsletter promotion` (an accessibility skip-link pair
  wrapping an embedded newsletter widget) — can appear anywhere in the
  body, stripped wherever it occurs.
`_strip_scraper_boilerplate()` added to `prepare_multihoprag.py`, applied to
`article.get("body")` only (the separate `evidence_facts` field in
`MultiHopRAG.json` comes from a different, already-clean `fact` field, not
the scraped body). Caught and fixed twice in quick succession — first
attempt anchored on the opening phrase and missed the Lifestyle-section
variant, which starts differently; broadened to anchor on the fixed closing
markers instead. Verified against the full corpus after each attempt
(byte-size distribution, zero remaining marker occurrences) before
resuming indexing.

## Two more corpus-prep leak bugs, found live during the full reindex (2026-08-22)

Caught during the from-scratch full-dataset reindex by a periodic quality
monitor that samples real indexed chunks from Neo4j (not just watching
progress/error counts) — both stopped and fixed within ~1 minute of the
run starting, minimal cost sunk.

- **HTML entities in HotpotQA/MuSiQue titles**: `prepare_hotpotqa.py`/
  `prepare_musique.py` never called `html.unescape` on article titles
  (`prepare_multihoprag.py` already did), so entity-named articles' titles
  and filenames carried literal `&quot;`/`&amp;`/etc. (e.g.
  `&quot;J&quot; Is for Judgment`). Body text itself was already clean.
  Gold `evidence_docs` happened to use the same escaped form, so automated
  title-matching wasn't silently broken — but citations would have read the
  escaped form forever. Folded `html.unescape` into both scripts'
  `clean_wiki_markup()` and applied it to titles too (previously only body
  text/evidence_facts went through it).
- **`Category:` header leaking into MultiHop-RAG's first chunk**: the
  earlier "News-metadata injection removed" pass (see below) trimmed the
  corpus-file header from 4 lines to 2 (`Title:`/`Category:`), but the
  chunker's header-skip logic only ever strips exactly one line (`Title:`)
  — so `Category: <cat>` became parts of `sent_id=0`'s body text for every
  single document, undetected because this 2-line header format had never
  actually gone through a real indexing run until this reindex. Checked
  whether `category` is read anywhere downstream first (it isn't — the
  benchmark's own `category`/`question_type` comes from `queries.json`, a
  completely different field) and removed the header line entirely rather
  than teaching the chunker about a second header line, so MultiHop-RAG's
  corpus header format now matches HotpotQA/MuSiQue's (`Title:` only).

## Baseline top-k provenance and alignment (2026-08-22)

Audited the unequal synthesis-context counts across strategies against their
upstream implementations. HopRAG now keeps the official repository's
end-to-end `--topk 20` setting (the local adapter previously returned 10 after
reranking). Naive RAG has no upstream fixed value, so its prior local default
of 5 was removed and it now shares Prehop's domain-aware
`RAGConfig.DEFAULT_TOP_K` (news=12, financial=8). MS GraphRAG was left alone:
the benchmark uses its official `local_search` / `global_search` APIs and does
not call the adapter's separate dense `retrieve(top_k=5)` helper. This is a
benchmark-only change and requires no reindexing.

## Fail-loud sweep across indexing/retrieval (2026-08-22)

Removed 9 "keep-system-alive" fallbacks from `models/prehop/indexing/` and
`models/prehop/retrieval/` — places where a real failure (bad JSON from the
LLM, an empty query embedding, an empty candidate pool) was being silently
absorbed into a default/empty value instead of surfacing. Explicitly scoped
to prehop's own indexing/retrieval code, not `core/vllm_client.py` (shared
by every strategy — its own retry-then-degrade behavior stays) and not the
baseline adapters.

- `graph_writer.py`: found and fixed a real correctness bug the fallback was
  masking, not just a style issue. `embedding` (the primary/body field) was
  silently substituted with `q_minus_embedding` whenever Q⁻ existed. Since
  `body_vector_index` is built on the `embedding` property, this meant the
  "body" search channel was actually searching Q⁻ content for any chunk
  with a non-empty Q⁻. Fixed: `embedding` is now always the true body
  embedding; missing body embedding raises instead of falling back.
- `knowledge_mapping.py`, `traversal.py`, `rewrite.py`, `rerank.py`: each
  had a `generate_json(...)` call whose failure (after `core/vllm_client.py`
  exhausts its own retries and returns `{}`) was silently treated as "empty
  result" (empty Q-/Q+, "INSUFFICIENT" continuation decision, no query
  rewrites, unsimplified rerank query) rather than raising. All four now
  raise, so a genuine failure shows up as a failed chunk/query
  (`data/index_failures/` at indexing time, a 0-score `runtime_error` result
  at query time — `cli/benchmark.py` already isolates failures per query) instead
  of silently degrading quality with no signal.
- `hop_edges.py`: `_find_hop_candidates` fell back to querying the Q⁻
  vector index (with a Q⁺ embedding as the query vector — a semantically
  odd cross-channel search) whenever the primary Q⁺↔Q⁺ ANN search came back
  empty. Undocumented anywhere in the module's own docstring. Removed —
  empty candidates now just mean no HOP edge for that chunk.
- `hybrid.py`: an empty query embedding was silently treated as "this
  channel has no candidates" rather than raising.
- `retrieve.py`: Stage 2 (Q⁺ expansion) falling back to Stage 1's
  *un-thresholded* reranked candidates when Stage 2 rerank produced nothing
  above `RERANKER_THRESHOLD` — silently returning content that had already
  failed the quality gate. Removed; the query now correctly falls through to
  "Insufficient evidence" instead.
- One test (`test_retrieve_prefers_company_matched_candidate`) turned out to
  depend on this exact failure mode: its query is >80 chars, which triggers
  `_simplified_rerank_query`'s LLM call, and the test never mocked
  `generate_json` — it was silently relying on the network call failing and
  falling back to the original query. Fixed by mocking `generate_json`
  properly.

## Shared-module refactor (2026-08-22)

Immediately followed the fail-loud sweep, which had mechanically introduced
the same 3-line "call generate_json, raise if empty" pattern at 4 call
sites — a clear signal a shared helper was overdue.

- Deleted `rerank.py::hybrid_search` — confirmed zero callers anywhere in
  the repo, fully superseded by `retrieve.py`'s `RetrieveMixin.retrieve()`.
- Added `TextUtilsMixin._rrf_accumulate` (`text_utils.py`) unifying the two
  independent reciprocal-rank-fusion implementations in `hybrid.py`
  (`update_rrf`) and `retrieve.py` (`_accumulate`).
- Deleted `traversal.py`'s local `_rerank_and_gate` closure and pointed both
  of its call sites at `rerank.py`'s existing `_rerank_and_select` instead.
  This fixed a real, previously undocumented inconsistency as a side effect:
  `_rerank_and_select` ran the query through `_simplified_rerank_query`
  (strips verbose/role-played preludes — empirically verified elsewhere to
  collapse a rerank score from 0.94 to 0.03) before scoring;
  `_rerank_and_gate` did not, scoring `graph_search`'s often-long synthetic
  query raw. Decided to unify on always simplifying.
- Added `models/prehop/llm_json.py` (`generate_json_or_raise`) — the shared
  fail-loud wrapper the 4 call sites above now use, instead of each
  duplicating the same raise block.
- Hard constraint respected throughout: no changes to `models/naive/`,
  `models/hoprag/`, `models/ms_graphrag/`, or `core/`/`cli/`.

## Dead-field cleanup

- `Document.summary` (a document-level LLM-generated summary, written at
  indexing time) removed after an actual quality A/B experiment — not a
  cost argument. Same retrieval, two synthesis-context conditions (with vs.
  without the summary injected), same judge. Result: no measurable benefit,
  plus one concrete case where the summary actively misled the synthesis
  answer. Removed `graph_writer.py::summarize_document`, `cli/index.py`'s
  summarization pass, and the now-unused `GLOBAL_SUMMARY_PROMPT`/
  `GLOBAL_SUMMARY_FORMAT_INSTRUCTION` prompts. (Distinct from `chunk_summary` —
  the per-chunk summary produced alongside Q-/Q+ — which is still generated
  and still feeds the body fulltext index; not touched.)
- Following the same "is this actually read anywhere" question generally
  (not just for the summary): `body_embedding` (computed but never queried —
  the primary `embedding` field already serves body search) and `corpus`
  (redundant with the Neo4j label itself, which already namespaces every
  query by corpus tag) were both confirmed dead by direct code inspection
  and removed from `graph_writer.py`'s Chunk node writes. Two more
  never-referenced dict keys (`q_plus`, `q_plus_embed`) were found in the
  same batch-data payload and dropped.
- The identical pattern was found in `models/naive/naive_rag.py` — a
  completely separate indexing implementation from prehop's own
  `graph_writer.py` — which independently wrote both a `corpus` and a
  `branch` property on its Chunk nodes. Same root cause: `self.chunk_label`
  (built from `corpus_tag` + `ablation_profile`) already namespaces every
  `MATCH`/`MERGE` via the node label, so both properties were pure
  unread redundancy. Removed there too.

## Data-quality bugs found via code-intent audit

An audit compared indexed Neo4j data with the source code's documented
contracts across the three then-active corpora.

- MultiHop-RAG: 665/8,442 chunks carried raw source-file header boilerplate
  (`Title:`/`Source:`/`Category:`/`Published:` lines) leaking into chunk
  text because `prepare_multihoprag.py`'s article header format didn't match
  what the chunker expected to strip. Fixed at the source (`prepare_multihoprag.py`),
  corpus regenerated, affected corpus re-indexed.
- HotpotQA (caught by the audit) and MuSiQue (fixed proactively, same
  pattern, before the audit reached it): unstripped MediaWiki markup
  (`<nowiki>`, `<br>`, `[[wikilink|display]]`, `{{template}}`) leaking into
  both corpus body text and `evidence_facts`. Added a shared
  `clean_wiki_markup()` function (duplicated per prep script, matching the
  established per-script-duplication convention in `data/prepare_*.py`) and
  applied it at both corpus-build and query-build time.

## Full-dataset regeneration

`prepare_hotpotqa.py`/`prepare_musique.py` default to `--limit 2000`
(questions, not documents) — an early full-corpus indexing run turned out to
have been built from that default subset, not the true full datasets.
Regenerated with `--limit 0` (true full: HotpotQA 7,405 / MuSiQue 2,417
questions) for a real from-scratch reindex.

## News-metadata injection and comparison-query decomposition removed entirely

Two MultiHop-RAG-specific engineering features — injecting `Source:`/
`Published:` metadata lines into the synthesis context, and decomposing "A
vs B" comparison queries into per-entity retrieval variants — were removed
entirely from the code (not just disabled behind a flag), after deciding
they sat outside the paper's three main claims and broke the
dataset-general/domain-agnostic parity the benchmark suite is built around.
`_build_context_from_nodes` now always uses the single `[[title, Page N,
Chunk N]]` format; `QUERY_REWRITE_PROMPT` always does plain paraphrase
rewriting.

## Reranker replaced: cross-encoder → embedding cosine-similarity

An earlier iteration used a dedicated reranker model
(`Qwen3-Reranker-0.6B`, served separately) for two purposes: query-time
retrieval reranking, and — critically — scoring candidate HOP edges at
indexing time. That model isn't part of the current inference setup, so
both uses were replaced with embedding cosine-similarity: HOP-edge scoring
now reuses the score Neo4j's own `db.index.vector.queryNodes` ANN query
already returns (no extra model call at all), and query-time reranking uses
a shared `_embedding_rerank_scores(query, texts)` helper. Both
`HOP_THRESHOLD`/`RERANKER_THRESHOLD` were originally calibrated for
cross-encoder classifier scores (roughly a 0-1 probability) and now gate raw
bi-encoder cosine similarity instead — flagged as likely needing empirical
re-tuning once real benchmark numbers exist; not yet done.

Separately, the Q⁺ quality gate (`_is_high_quality_q_plus`, requires an
entity/period/metric/source-anchor signal) was originally a strict 4-of-4
AND — this collapsed acceptance to ~2.8% of generated Q⁺ (~1,640/47k
chunks), leaving the HOP graph effectively empty. Relaxed to "at least 2 of
4" after observing that bridge questions about the same metric across
periods, or a related metric in the same period, naturally drop one signal.

## Model infra migrated to LiteLLM proxy

Generation and embeddings now route through configured OpenAI-compatible
endpoints rather than repository-managed local model processes. The migration
used `gemma-4-31b-it` (chat) and
`qwen3-embedding-4b` (embeddings, dim=2560) — a dimension change from
whatever was configured before, requiring `NEO4J_VECTOR_DIMENSIONS` to be
updated and the vector indexes rebuilt from scratch (dimension is fixed at
index-creation time).

Live smoke test (2-doc toy corpus, before any real-scale benchmark existed)
showed retrieval — not traversal — dominating total query latency (LiteLLM
proxy round-trips for query rewrite + per-channel hybrid RRF embedding
calls). Useful context before assuming the graph-traversal step is a
latency risk; not re-verified at real scale yet.

## hoprag/ms_graphrag baseline dependency fixes

Neither baseline had been exercised even once before this pass. hoprag
crashed on a missing `pandas` import — initially misdiagnosed as needing
`paddlenlp`/`modelscope` too, but those are already stub-injected via
`sys.modules` before the vendored `third_party/HopRAG/tool.py` imports them,
so only `pandas` was a genuine gap. Both baselines' endpoint/auth defaults
were hardcoded to dead local ports and a hardcoded `"EMPTY"` API key from an
earlier local-serving setup; fixed to fall back through the same
`VLLM_URL`/`VLLM_EMBED_URL`/`VLLM_API_KEY` env vars prehop's own pipeline
uses. ms_graphrag additionally had a hardcoded `vector_size=1024`, stale
from before the embedding-model switch — now reads
`NEO4J_VECTOR_DIMENSIONS`. A known remaining limitation: hoprag's official
(unmodified, verified byte-identical to upstream) edge-construction code can
throw `KeyError` on an imbalanced pending/answerable question-category
cross-join at small corpus scale — not yet re-verified at real scale.

## Full rename: prehypo → prehop

Package directory, strategy id, Neo4j label prefix, docker container name,
and loggers all renamed from the project's original working name
(`prehypo`) to `prehop` — `hypohop` is the paper's title only, not used
anywhere in code/config.
