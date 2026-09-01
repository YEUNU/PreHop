# Changelog

This file is the chronological engineering record. `ARCHITECTURE.md` defines
current behavior, `README.md` is the user-facing guide, `CLAUDE.md` defines
maintenance policy, and the local `prehop_paper.md` owns paper prose and claims.
Entries are newest first. Older entries describe behavior at that point in
development and may be superseded; they are not a current configuration guide
or a source of paper claims.

## 2026-09-01 — MS GraphRAG MuSiQue answer-boundary correction

MS GraphRAG LocalSearch now requests one short answer span with an explicit
`Final Answer:` prefix. The response contract is stored in each query trace.
This aligns its generated answer boundary with MuSiQue normalized EM/F1 while
retaining the official LocalSearch retrieval and context construction path.
Because a provider may still omit the requested label, the adapter now also
attaches the shared answer boundary after both LocalSearch and GlobalSearch
return. This post-processing changes no answer text.

The shared answer extractor now recognizes explicit `Final Answer:`,
`@@ANSWER:`, and line-leading `Answer:` markers. An unmarked response is kept
as the complete prediction. This prevents long citation-bearing responses
from losing their leading answer span and prevents the ordinary word
`answer` inside a sentence from being treated as an output marker.

The same check covered every retained full-result row for Naive, Prehop, and
HopRAG on MuSiQue and MultiHop-RAG. The corrected boundary changed the
prediction passed to metrics for 3 Naive and 8 HopRAG MuSiQue rows, and 8
Naive, 1 Prehop, and 60 HopRAG MultiHop-RAG rows. Answer EM did not change for
any method. Mean answer-F1 changes were −0.00002 and −0.00003 on MuSiQue, and
+0.00003, −0.000001, and +0.00025 on MultiHop-RAG in the same method order.
Naive and HopRAG now attach the same explicit answer boundary as Prehop,
including abstentions, without changing the generated text. Explicitly marked
responses are no longer cut to 400 characters during metric calculation.

## 2026-09-01 — 발표·논문 문서의 용어와 실험 흐름 통일

발표자료와 현재형 문서를 색인 단계, 질의 단계, 같은 후보 재평가 순서로
맞췄다. 내부 옵션명은 재현 명령에만 두고, 본문에서는 한 단계 이웃 확장,
근거 기반 질문 보충, 질문 역할별 후보 선택, 같은 후보의 점수 재계산처럼
실험에서 실제로 바꾼 대상을 사용한다.

구성요소 실험은 MuSiQue 2,417개 전체 결과만 사용한다. Ablation 1은 같은
색인에서 그래프 확장을 켜고 끈 비교, Ablation 2는 반복 질문 보충 비교,
Ablation 3은 후보 선택 정책 비교, Ablation 4는 같은 후보에서 점수식을
바꾼 비교다. 후보 입력 순서와 단계별 시간도 2,417개 전체 결과로 제시한다.
MultiHop-RAG와 MuSiQue의 전체 성능표와 분모는 각각 분리해 유지했다.

발표자료의 방법·실험 도식에는 코드 식별자 대신 연구 개념어를 사용했다.
결론 슬라이드는 전체 성능, 구성요소 기여, 단계별 시간의 측정 결과만
요약하도록 정리했다.

## 2026-09-01 — Full-split Prehop controls replace subset diagnostics

The final documentation now maps every control to its actual pipeline
location. Stored-edge coverage is a read-only inspection of the completed
index. Graph on/off, refinement removal, and candidate-ordering removal reuse
that index and change only query behavior. Candidate presentation and score
formula analyses freeze the retrieved pool, and the timing study changes no
retrieval behavior. `RESULTS.md`, `ABLATION_STUDY.md`, and
`CONSISTENCY_AUDIT.md` register these boundaries, commands, artifacts, and
accepted claim wording. A mechanical submission audit verifies both full
eligibility gates, complete control counts, canonical values, and briefing
terminology.

The graph-on/off artifact now includes fixed MuSiQue hop-depth strata. This
depth 1/0 control disables NEXT and HOP together and is not a HOP-only
contrast. For
2-hop, 3-hop, and 4-hop questions, graph-on minus graph-off Answer EM was
+0.00240, +0.00921, and +0.00741; every 95% interval included zero. Retrieval
passes changed by +0.00559, +0.00789, and −0.00988; every interval also
included zero. Support F1 increased by +0.00596 for 2-hop and +0.00543 for
3-hop, but changed by −0.00268 for 4-hop. Together with 1.32% full structural
connectivity at 3 hops and 0% at 4 hops, these results reject a complete-path
compression claim and retain only a small local evidence-selection effect.

The briefing now presents the stage map in the main sequence and the
hop-stratified graph effect beside structural coverage in the appendix. Draft
status language, the separate-reranker implication, future-work defenses, and
cross-dataset component attribution were removed. External systems remain
official-setting point-estimate references because their budgets differ and
their outputs are not synchronized paired counterfactuals.

The completed `global` candidate-selection control is also scoped to the
broader policy: it removes the final generation-model ordering call and changes
the deterministic refinement preview from role/body rounds to global fused
order. Its answer/support effect is not attributed to the final call alone.
Only the frozen-candidate replay isolates that call's input-order sensitivity.

The presentation evidence scope is now fixed. External systems are retained
only as reference results under their own official settings. The incomplete
attempt to regenerate answers from an equal twelve-passage evidence budget was
stopped and is excluded from the briefing, paper claims, and comparative
tables. It is not an experiment result. MuSiQue-only Prehop controls do not
support a claim about MultiHop-RAG.

Two Prehop component-removal controls completed the full 2,417-question
MuSiQue split with zero errors. Removing only evidence-conditioned iterative
question refinement, while retaining the initial role-aligned questions,
changed answer EM by −0.12784 (paired 95% bootstrap interval −0.14398 to
−0.11212), answer F1 by −0.14000 (−0.15574 to −0.12448), paragraph-
support F1 by −0.04811 (−0.05162 to −0.04466), and support recall by
−0.12657 (−0.13509 to −0.11802). Omitting the generation model's single
complete-list ordering call and using the deterministic fused top twelve
changed answer EM by −0.08854 (−0.10343 to −0.07365), answer F1 by
−0.09898 (−0.11427 to −0.08434), paragraph-support F1 by −0.04952
(−0.05301 to −0.04618), and support recall by −0.11481 (−0.12350 to
−0.10630). Prehop has no dedicated reranker; the latter control removes one
list-ordering call made by the same configured generation model. Because the
controls used separate generation calls and were resumed under mixed load,
their latency fields are excluded.

The fixed-concurrency full benchmark then completed all 2,417 questions with
zero errors and captured every candidate pool. Its deterministic frozen-pool
analysis found support F1 0.27791 under the default equal reciprocal-rank
fusion and graph decay 0.5. Semantic-only ranking changed support F1 by
−0.03269 (paired 95% bootstrap interval −0.03595 to −0.02943),
representation-only by −0.00359 (−0.00582 to −0.00142), decay 0 by +0.00457
(+0.00327 to +0.00592), decay 1 by −0.00870 (−0.01027 to −0.00721),
body-only semantics by −0.00028 (−0.00072 to +0.00015), and bridge-only
semantics by −0.00464 (−0.00567 to −0.00364). The full result supersedes the
former subset impression of formula insensitivity and leaves the 0.5 decay as
a consequential heuristic rather than an optimum.

The same run separated query-stage time at concurrency 32. Mean accounted
time was 131.69 seconds: rewrite/refinement 57.79 s, retrieval 17.61 s, graph
expansion 2.25 s, deterministic scoring 0.09 s, candidate ordering 41.83 s,
and synthesis 12.12 s. Generation-model stages occupied 84.9% and graph
expansion 1.7%. Absolute values are service-load specific and are not compared
with the official 23.86-second run; the eligible conclusion is the within-run
decomposition.

The full frozen candidate-order replay also completed all 2,417 questions
with zero errors. A same-input-order replay reproduced the recorded selected
set with mean Jaccard 0.96760 (95% interval 0.96363 to 0.97129). A
query-specific fixed shuffle of the identical content had Jaccard 0.63909
against the same-order replay (0.63090 to 0.64717), and the paired difference
after same-order variability calibration was −0.32851 (−0.33705 to −0.31997).
The shuffle changed support F1 by −0.00368 (−0.00535 to −0.00201). Its
first-input selection rate exceeded the random-position expectation by only
2.8 percentage points (+1.2 to +4.6 points), so the call is order-sensitive
without merely copying the first item. Retrieval and answer synthesis were
not repeated; answer EM/F1 is therefore out of scope.

The former 201-question candidate-order and score diagnostics are superseded
for presentation claims. Their historical entries remain below only as an
engineering chronology. The presentation now uses only the completed frozen
candidate input-order replay, deterministic score-formula sensitivity, and
separated query-stage timing from all 2,417 MuSiQue questions.

## 2026-08-31 — Presentation-defense controls and claim narrowing

The professor briefing was audited against the implementation and immutable
MuSiQue artifacts. The terminology now distinguishes deterministic score
fusion from the one generation-model candidate-ordering call. Prehop has no
separate learned reranker: the configured generation model produces query
views, orders one complete candidate list, and writes the final answer. The
briefing therefore uses “candidate ordering” rather than implying a dedicated
reranking model.

The “path compression” interpretation was also narrowed. On all 2,417
MuSiQue questions, at least one stored HOP edge joined gold paragraphs for
23.6% of 2-hop, 22.0% of 3-hop, and 25.7% of 4-hop questions. Gold-only paths
were fully connected for 23.6%, 1.3%, and 0.0%, respectively. Three-hop
questions still averaged 3.37 retrieval passes and four-hop questions 3.59.
The current graph is therefore described only as a possible local shortcut;
it does not compress a complete multi-hop answer path into one global hop.
The complete-system performance tables remain valid, but they are no longer
used as causal evidence for stored HOP edges.

The graph-on/off diagnostic then completed all 2,417 questions. Graph-on minus
graph-off was +0.00538 answer EM (95% interval −0.00455 to +0.01489), so no
answer-accuracy gain was detected. Paragraph-support F1 increased by +0.00435
(+0.00269 to +0.00603), while support recall changed by +0.00186 (−0.00190 to
+0.00569). The run used separate generation calls and the graph-off side was
resumed across behavior-equivalent code segments. Its mixed-load latency is
explicitly ineligible; `scripts.analyze_presentation_controls.py` now accepts
`--exclude-latency` so the derived comparison artifact cannot accidentally be
used for a timing claim.

The full pair was also stratified by the precomputed structural labels. Among
566 questions with at least one stored edge between gold paragraphs, graph-on
minus graph-off was −0.00177 answer EM (−0.02473 to +0.02120), +0.00728
support F1 (+0.00398 to +0.01068), +0.01075 support recall (+0.00280 to
+0.01914), and −0.00177 inferred retrieval passes (−0.02827 to +0.02650).
Among 305 questions with a connected gold-only subgraph, support F1 increased
by +0.00895 (+0.00534 to +0.01284) and support recall by +0.01858 (+0.00765 to
+0.03060), but answer EM and retrieval passes did not improve. This supports
only a small local evidence-recovery effect. It does not establish answer-path
compression or fewer online retrieval rounds.

A balanced 201-query MuSiQue control then separated three online components.
Disabling stored graph expansion changed answer EM by +0.005 relative to the
full condition, with a paired 95% bootstrap interval of −0.030 to +0.040.
Disabling evidence-conditioned refinement reduced answer EM by 0.100
(−0.149 to −0.055). Removing the generation-model candidate-ordering call left
answer EM unchanged and reduced paragraph-support F1 by 0.025
(−0.035 to −0.016). These are explicitly labelled exploratory because the
fixed sample was used during development.

`scripts/audit_hop_question_quality.py` added a deterministic 200-path audit
with source text, Q+, matched Q−, and target text. The configured generation
model's non-independent diagnostic labelled 97.0% of Q+ as source-grounded and
100% as entity-explicit, but only 5.0% of targets as answering Q+ (Wilson 95%
interval 2.7–9.0%) and 0.5% of matched Q− as semantically equivalent. The
dominant failure is not pronoun-only Q+: similarly worded relations about
different named entities are connected. The output CSV deliberately leaves
human decisions blank. The briefing includes this negative result and treats
an entity-and-relation-bound index rebuild plus blinded human review as a
prerequisite for a factual-edge precision claim.
The earlier `linked_v2` answer-anchor continuation control also failed as a
simple repair: enabling it changed paragraph-support F1 by −0.0082 on 201
MuSiQue questions (paired 95% interval −0.0157 to −0.0010). Because that run
used a separately built index and the global deterministic selector, it is
reported only as evidence that an entity label alone is insufficient; the
next schema must bind both the referenced entity and requested relation.

The diagnostic implementation now supports:

- `RAG_CANDIDATE_ORDER_TRACE_PATH`,
  `RAG_CANDIDATE_ORDER_INPUT_ORDER`, and
  checkpoint-resumable `scripts/replay_frozen_candidate_order.py` for exact
  candidate-pool replay;
- `RAG_FINAL_RANK_VARIANT` for fused, semantic-only, and
  representation-only final orders;
- `RAG_HOP_SEMANTIC_VARIANT` values `body_only` and `bridge_only` beside the
  default body/bridge minimum;
- `RAG_GRAPH_PATH_DECAY` for the declared zero, one-half, and one propagation
  sensitivity;
- `RAG_QUERY_REFINEMENT_MAX_ROUNDS`, where zero preserves evidence-driven
  stopping and a positive value imposes an auditable operational cap;
- separate `graph_expand_ms`, `deterministic_score_ms`, and
  `candidate_order_ms` fields, avoiding the former graph/score/model aggregate;
- `scripts/resynthesize_equal_evidence_budget.py` and its paired analyzer for
  a partial equal-evidence external comparison.

All six score sensitivities completed on the balanced 201-query set. Replacing
the final equal-rank fusion by semantic-only or representation-only order,
setting graph propagation to zero or one, and using body-only or bridge-only
HOP semantics produced answer-EM point estimates from 0.2985 to 0.3184 and
paragraph-support F1 from 0.3172 to 0.3214. Every paired 95% interval for
answer EM/F1 and paragraph-support F1/recall included zero. The result narrows
the criticism to lack of a theoretical derivation rather than observed
sample-level brittleness; it does not establish optimality. Absolute latency
from these heavily concurrent runs is explicitly ineligible.

The equal-evidence control completed on the balanced 201 MuSiQue questions.
All methods retained their first 12 frozen passages and used the same context
format, answer prompt, generation model, seed, and sampling settings. Prehop
reached 0.3085 answer EM and 0.3840 answer F1; Naive reached 0.1493/0.1795,
HopRAG 0.1741/0.2227, and MS GraphRAG 0.0945/0.1475. The three paired
answer-EM differences from Prehop had 95% bootstrap intervals below zero.
Upstream retrieval calls were not equalized and mean retained text ranged
from 6,393 to 7,289 characters, so the result remains a partial fairness
control rather than a complete compute-budget comparison. An initial timing
field included local semaphore queue time; the artifact marks those synthesis
timings ineligible, and no latency claim uses them.

A clean low-load run on the same balanced sample split the historical
`traversal_ms` aggregate into its actual components. Mean end-to-end latency
was 39.81 s: query rewriting/refinement 18.02 s, candidate ordering 12.80 s,
answer synthesis 3.51 s, retrieval 4.03 s, graph expansion 1.35 s, and
deterministic scoring 0.08 s. Generation therefore occupied 34.33 s (86.2%)
of the observed mean, while graph expansion occupied 3.4%. The result corrects
the earlier interpretation of traversal time and explicitly prevents an
end-to-end latency claim from the offline-link design.

Paired refinement-cap runs also completed. A one-call cap reduced latency by
5.69 s (95% interval −6.92 to −4.51) but reduced paragraph-support recall by
0.0108 (−0.0220 to −0.0004); its answer-EM difference was −0.0149 with an
interval including zero. A two-call cap reduced latency by 1.09 s (−2.13 to
−0.07), while answer EM changed by +0.0100 (−0.0149 to +0.0348) and support
recall by −0.0062 (−0.0137 to 0.0000). The evidence-driven stop remains the
full-result default; two calls are retained only as a possible operational
trade-off.

The checkpoint-resumable frozen candidate replay completed all 201 pools with
zero errors. Repeating the captured search order produced mean top-12 set
Jaccard 0.966 (95% interval 0.950 to 0.980) and exact-set agreement on 88.1%
of queries. Reverse order versus the search-order replay produced Jaccard
0.584 (0.557 to 0.614) and 4.0% exact agreement; deterministic hash shuffle
produced 0.635 (0.607 to 0.663) and 8.0%. The first presented candidate was
selected on 93.0% of search-order calls, 11.4% of reverse calls, and 16.9% of
shuffled calls. The result establishes input-order dependence beyond
same-order generation variability, but not a simple top-item copying rule.
Because this replay does not regenerate answers, the mitigation decision
remains a complete-split comparison between deterministic fused order and any
multi-order consensus design.

Tests cover the new configuration guards, score-signal variants, propagation
factor, frozen candidate trace, input-order controls, and refinement cap.
The briefing's canned Solaris/Tarkovsky example was removed because it was not
an indexed execution case. It now uses an actual stored MuSiQue path concerning
the 1941 declarations of war, and explicitly notes where the matched Q− is not
semantically identical to Q+.

## 2026-08-31 — Full-split gate passed and final defaults frozen

The final Prehop indexes and benchmarks completed on new, strategy-specific
tags without clearing or modifying the existing `musique` and `multihoprag`
indexes. Both indexes were built with project caches disabled, seed 42, the
legacy Q−/Q+ schema, six-sentence chunks, and top-k 12. MuSiQue indexed all
21,099 sources into 23,280 chunks with zero failures and passed every graph
integrity check. Its cold wall time was 5,676.81 seconds and its estimated
logical property payload was 4.290 GiB. MultiHop-RAG indexed all 609 sources
into 8,529 chunks with zero failures and passed every integrity check. Its
cold wall time was 2,145.63 seconds and its estimated logical payload was
1.545 GiB. The immutable statistics are
`data/index_stats/prehop_musique_final_20260830_final-musique-prehop-index-20260830.json`
and
`data/index_stats/prehop_multihoprag_final_20260830_final-multihoprag-prehop-index-20260830.json`.

The complete 2,417-query MuSiQue result reached 0.4150 official answer EM,
0.5115 answer F1, 0.2034 support precision, 0.8840 support recall, and 0.3267
support F1, with 23.86 seconds mean end-to-end latency. The complete
2,556-query MultiHop-RAG result reached 0.9268 Hits@4, 0.9494 Hits@10, 0.8232
MRR@10, and 0.4550 MAP@10, with 27.63 seconds mean latency. Both artifacts
have completed status, the exact full-query digest, a matched corpus
fingerprint, a matched active-source snapshot, and no error row.

Before launching comparison work, an unnecessary new HopRAG MultiHop-RAG
index was started under a fresh tag. It was stopped after about 12 minutes
because neither code, source data, model policy, nor official metric code had
changed. No HopRAG nodes had been written and no existing index was touched.
Its failed run record remains at
`data/index_stats/hoprag_multihoprag_hoprag_final_20260831_final-multihoprag-hoprag-index-20260831.json`;
it is not a benchmark artifact. This corrected the execution policy: an
unchanged full comparison is verified from its immutable per-query result,
not rebuilt merely because the older summary predates the current manifest
field.

`scripts/verify_existing_baseline.py` implements that read-only verification.
It requires an exact current query-record match, current corpus-manifest and
content match, the historical full-size active snapshot, membership of every
retrieved passage in the current corpus, and exact recomputation of every
per-row and aggregate official metric. It writes a new derived summary and
does not alter the original result. This is stricter than copying historical
aggregates and resolves the earlier manifest-absent rejection without relaxing
the gate. MultiHop-RAG HopRAG and MS GraphRAG passed all checks; their verified
summaries are under `data/results/verified-multihoprag-baselines-20260831/`.

MuSiQue required one additional compatibility step because the shared final
answer prompt changed after the matched HopRAG retrieval run. The complete
answer-only rerun already made under the selected prompt was paired by exact
query ID with the frozen full retrieval rows. `scripts/combine_frozen_synthesis.py`
recomputed answer EM/F1 from the former and support metrics from the latter,
recording both file hashes and leaving both inputs untouched. A ten-row direct
prompt replay produced nine byte-identical answers and one longer answer with
the same content. The combined full summary is
`data/results/verified-musique-baselines-20260831/hoprag_musique.summary.json`.
MS GraphRAG's official LocalSearch answer path did not use the changed shared
prompt, so its existing matched full artifact was reused directly.

No compatible full MuSiQue result existed for the restored controlled Naive
configuration, so this was the only comparison that required new work. Its
new tag indexed all 21,099 sources into 23,280 chunks in 227.82 seconds with
zero failures and a 0.456 GiB estimated logical payload. The complete run
reached 0.2106 answer EM, 0.2726 answer F1, 0.1364 support precision, 0.6241
support recall, and 0.2215 support F1, with 1.88 seconds mean latency.

The predeclared 10% gate then passed every official metric. On MultiHop-RAG,
the strongest comparison row was HopRAG for all four metrics; Prehop's relative
gains were 35.45% Hits@4, 12.27% Hits@10, 45.44% MRR@10, and 62.47% MAP@10.
On MuSiQue, HopRAG was strongest for answer EM/F1 and support recall, while
Naive was strongest for support precision/F1. Prehop's relative gains were
58.45%, 50.71%, 49.12%, 26.60%, and 47.50%, respectively. The paper-eligible
gate artifacts are
`data/results/final-multihoprag-performance-gate-20260831.json` and
`data/results/final-musique-performance-gate-20260831.json`.

Only after both gates passed were the operational defaults changed to the
frozen result policy: unfiltered stored HOP edges
(`RAG_HOP_EDGE_FILTER=none`), owner activation, evidence-conditioned iterative
role rewriting, complete candidate-list ranking, and top-k 12. Reciprocal
filtering, linked answer-continuation edges, sentence nodes, increased output
width, and document-round selection remain rejected experiments rather than
defaults. The two new verification utilities, the performance gate, and their
tests preserve the final comparison procedure as executable code.
The final repository check passed 299 tests with four external-integration
skips; repository-wide Ruff, formatting, and whitespace checks also passed.

## 2026-08-30 — Fixed policy grid and structural retrieval candidates

A fresh 2³ comparison was run on the fixed MuSiQue development IDs with one
active index, seed 42, top-k 12, the shared synthesis prompt, and no judge.
The three discrete policies were bounded role rewriting, offline reciprocal
HOP filtering, and exact matched-Q+ activation. This was a structural policy
comparison; no score weight, threshold, retrieval budget, or dataset-specific
rule was fitted.

| Rewrite | HOP filter | Activation | Answer EM | Answer F1 | Support P | Support R | Support F1 |
|---|---|---|---:|---:|---:|---:|---:|
| none | raw | owner | 0.1542 | 0.2006 | 0.1498 | 0.6032 | 0.2371 |
| none | raw | exact | 0.1542 | 0.1978 | 0.1498 | 0.6024 | 0.2370 |
| none | reciprocal | owner | 0.1393 | 0.1830 | 0.1514 | 0.5991 | 0.2387 |
| none | reciprocal | exact | 0.1443 | 0.1913 | 0.1509 | 0.5978 | 0.2381 |
| role-aligned | raw | owner | **0.1841** | **0.2385** | **0.1602** | **0.6443** | **0.2534** |
| role-aligned | raw | exact | 0.1841 | 0.2338 | 0.1596 | 0.6405 | 0.2524 |
| role-aligned | reciprocal | owner | 0.1542 | 0.2000 | 0.1578 | 0.6256 | 0.2489 |
| role-aligned | reciprocal | exact | 0.1592 | 0.2050 | 0.1580 | 0.6256 | 0.2491 |

The role-aligned, raw-HOP, owner-activation row strictly dominated both
reciprocal rows and the exact-activation row on every listed MuSiQue metric.
Reciprocal filtering and exact activation are therefore rejected as primary
defaults at this gate. The full-split A3/A4 comparison was retained only as a
paper ablation and was not used for this selection. On the development-excluded
remainder, raw HOP nevertheless independently showed higher answer EM and F1
than reciprocal owner activation, while its paragraph-support F1 interval
included zero and evidence-document F1 decreased. The paired artifact is
`data/results/musique-a3-vs-a4-heldout-20260830`.

Trace review then identified a role-routing omission. A generated Q+ states
the information another passage must answer, but the established query path
searched that view only against Q+ dependency seeds. The
`dependency_to_answer` ablation also sends the same Q+ view to the Q- answer
index, while retaining Q+ seed search and fusing every destination role into
one unweighted ranked list. This rule follows the dataset-independent Q-/Q+
definitions and introduces no learned value, fitted parameter, or
dataset-specific branch. The isolated run was recorded as
`musique-dev-crossrole-raw-owner-20260830`.

That isolated run completed at 0.1791 answer EM, 0.2350 answer F1, 0.1599
support precision, 0.6410 support recall, and 0.2527 support F1. The aligned
control was 0.1841, 0.2385, 0.1602, 0.6443, and 0.2534 respectively. Every
paired interval included zero, and all five target aggregates moved downward.
The route was rejected at the first dataset gate rather than adjusted for
MultiHop-RAG. Its experimental code was removed; this record and the artifacts
under `data/results/musique-dev-crossrole-*` preserve the decision evidence.

The next dataset-independent candidate addresses the established one-edge
query path. `fixed_point` traversal expands NEXT and raw HOP from newly
selected graph evidence, rescoring the accumulated candidates after each
round. It stops when the top-k identity set no longer changes or the selected
frontier has no unseen nodes; there is no fitted maximum depth, beam width, or
dataset-derived stopping threshold. Every additional edge applies the same
reciprocal one-edge rank inheritance already used by the single-step path.
This differs from the rejected fixed depth-two diagnostic, which imposed a
depth parameter and used the reciprocal-filtered graph before the current
rewrite and raw-HOP selection.

A five-query termination smoke run completed without errors. Relative to the
same five rows from the single-step control, answer F1 and support recall were
unchanged at 0.1143 and 0.8000; support F1 changed from 0.2352 to 0.2330 and
latency from 3.17 s to 3.56 s. The fixed development run
`musique-dev-fixedpoint-raw-owner-20260830` was then started; the candidate is
not selected from the smoke subset.

The full development run completed at 0.1791 answer EM, 0.2288 answer F1,
0.1594 support precision, 0.6405 support recall, and 0.2522 support F1. The
single-step control was 0.1841, 0.2385, 0.1602, 0.6443, and 0.2534. All target
aggregates decreased and every paired interval included zero. Fixed-point
expansion was rejected without a second-dataset adjustment, and its
experimental implementation was removed. The run and paired artifacts remain
under `data/results/musique-dev-fixedpoint-*`.

The next candidate keeps the selected single expansion and the same candidate
and evidence budgets. The established implementation retained a graph path on
a directly retrieved chunk only as provenance; it did not let an independent
HOP or NEXT path affect that chunk's representation order. The `graph_rrf`
ablation forms one ranked target list per graph edge type and contributes each
list once to the existing unweighted representation fusion. Multiple paths of
the same type cannot accumulate extra mass, and no raw similarity, fitted
weight, or cutoff is introduced. The isolated development run is
`musique-dev-graphrrf-raw-owner-20260830`.

Graph-path RRF increased support precision from 0.1602 to 0.1664 and support
F1 from 0.2534 to 0.2601, both with positive paired intervals. It reduced
support recall from 0.6443 to 0.6339 and answer F1 from 0.2385 to 0.2120; both
negative differences were significant. Because the target requires joint
improvement rather than a precision tradeoff, the candidate was rejected and
its experimental implementation removed. The result shifts subsequent work
from reordering known graph candidates to generating missing answer evidence.
Artifacts remain under `data/results/musique-dev-graphrrf-*`.

The next candidate changes candidate generation rather than graph ranking. A
generated Q- is defined as an atomic question answerable by one passage, but
the established routing searched it only against generated Q- nodes; the body
index still received only the original compound question. The `answer_views`
ablation also searches Q- views against chunk bodies. Original and generated
body queries are fused inside one body list before representation fusion, so
the body role gains no extra weight. The rule is shared across datasets and
adds no cutoff or fitted value. Its isolated run is
`musique-dev-bodyviews-raw-owner-20260830`.

Answer-view body routing reduced answer EM/F1 from 0.1841/0.2385 to
0.1542/0.2084 and support precision/recall/F1 from
0.1602/0.6443/0.2534 to 0.1543/0.6190/0.2439. Every paired difference was
significantly negative. The candidate was rejected without testing a
dataset-specific adjustment, and its experimental implementation was removed.
The artifacts remain under `data/results/musique-dev-bodyviews-*`.

A candidate-coverage audit then separated retrieval failure from final
ranking failure on the same 201 development IDs. The audit used the selected
role rewrite, raw HOP, owner activation, and top-k 12. Gold paragraph recall
was 0.6613 in the direct representation pool and 0.6973 after graph expansion,
but fell to 0.6339 in the final top 12. The expanded-pool ceiling is below the
0.7681 support-recall target implied by a 10% improvement over the comparison
result. No reranker over the existing candidates can therefore satisfy that
target. The problem is most severe on four-hop questions: direct-pool,
expanded-pool, and final recall were 0.5037, 0.5224, and 0.4590. The diagnostic
artifact is `data/results/musique-dev-candidate-coverage-20260830/results.json`.

The next isolated candidate adds one deterministic retrieval child per
sentence while retaining the six-sentence chunk as the only ranked and
synthesized evidence unit. Sentence vector and full-text results collapse to
their owner chunks before the existing unweighted representation fusion. Only
the original benchmark query searches this representation; generated Q-/Q+
views retain their established roles. The sentence count follows the fixed
chunking boundary, so the candidate introduces no learned score, fitted
threshold, dataset branch, or new final evidence budget. It requires a full
corpus reindex, and its additional indexing time and capacity will be measured
if it passes both fixed development gates. The implementation is guarded by
`RAG_SENTENCE_CHANNEL_ENABLED`; it is not a primary default before those
comparisons complete.

The independent sentence-role run completed at 0.1891 answer EM, 0.2229
answer F1, 0.1555 support precision, 0.6240 support recall, and 0.2459
support F1. The aligned control was 0.1841, 0.2385, 0.1602, 0.6443, and
0.2534. Paired intervals showed significant decreases in all three support
metrics; the EM increase and answer-F1 decrease both included zero. Trace
comparison found that 1,603 of 2,412 selected chunks carried a sentence path,
but only 60 were introduced by the sentence path alone and only four of those
were gold. Relative to the control, 20 gold chunks were displaced and eight
were gained. The independent fourth role was rejected because it mainly
double-counted candidates already found at chunk resolution. Results and
paired analysis remain under `data/results/musique-dev-sentence-*`.

The follow-up keeps the same sentence index but removes the extra role vote.
Chunk and sentence rankings are first fused into one body list; that list then
contributes once to the established Q-/body/Q+ union. This is an isolated
test of multi-resolution candidate generation without representation-weight
inflation. It retains the same original query, top-k, graph, prompt, and fixed
development IDs and introduces no fitted value. It remains unselected pending
the MuSiQue development result.

The body-fused run completed at 0.1592 answer EM, 0.1999 answer F1,
0.1550 support precision, 0.6190 support recall, and 0.2448 support F1.
Against the original control, answer F1 and every support metric decreased;
the paired intervals for answer F1 and all three support metrics were
significantly negative. Multi-resolution sentence retrieval was therefore
rejected in both fusion placements.

A control rerun with sentence retrieval disabled exposed a separate index
reproducibility issue before the next query experiment was started. The old
development control used 67,859 Q- nodes, 65,646 Q+ nodes, and 61,185 HOP
edges. Reindexing from the current cache produced 67,867 Q- nodes, 65,632 Q+
nodes, and 61,317 HOP edges; the same query policy consequently changed from
0.1841/0.2385 answer EM/F1 and 0.6443 support recall to
0.1592/0.1997 and 0.6235. The old and current controls must not be mixed in a
paired selection. Subsequent query-only candidates use
`musique-dev-reindexed-control-raw-owner-20260830` on the current active index.
The count drift also means a final paper index must be frozen and its generated
Q-/Q+ artifacts retained; a cache-assisted development index is not a cold
indexing-cost result.

Trace logs showed that the role rewrite sometimes generated four atomic
questions and then discarded the fourth because the indexing schema's
three-question bound had also been applied at query time. That bound is valid
for controlling stored Q-/Q+ nodes per chunk, but it is not a property of a
user question and can delete an explicit dependency. The next isolated
candidate removes only the query-time count cutoff. The prompt requests the
smallest set covering every explicit dependency, while validation still
deduplicates and rejects malformed values. Per-role view fusion remains a
single ranked list, so additional dependencies do not add another evidence
role or alter top-k. This candidate uses the current reindexed control and has
no dataset-specific hop-count branch or fitted maximum.

The complete-role-query run produced 0.1642 answer EM, 0.2008 answer F1,
0.1552 support precision, 0.6219 support recall, and 0.2454 support F1.
The current-index control was 0.1592, 0.1997, 0.1552, 0.6235, and 0.2454.
All paired intervals included zero; support recall and support F1 moved
slightly downward. Four-hop support recall improved, but four-hop answer F1
decreased. The query-time cutoff removal was rejected as a global policy and
the established bound was restored.

The next candidate keeps that bound and changes when the second rewrite is
formed. It first runs the established retrieval path, then asks for new
role-aligned questions that bind intermediate entities explicitly present in
the retrieved evidence to still-unresolved relations in the original query.
The initial and evidence-conditioned questions are fused inside their existing
Q- or Q+ role, after which one final retrieval and graph traversal produces
the unchanged top 12. This is a fixed two-stage retrieval architecture, not an
iterative stopping rule, fitted depth, or dataset-specific hop branch. The
preview retrieval cost and second rewrite latency are included in the query
latency record.

On the current MuSiQue index, the fixed development run
`musique-dev-evidence-role-raw-owner-20260830` improved answer EM/F1 from
0.1592/0.1997 to 0.2338/0.3019 and support precision/recall/F1 from
0.1552/0.6235/0.2454 to 0.1695/0.6766/0.2677. Every listed paired difference
was positive with a 95% bootstrap interval excluding zero. Mean query latency
increased from 4.14 s to 7.35 s because the preview retrieval and second
rewrite are part of the measured request. This is the first current-index
candidate to improve answer quality and all support aggregates together, so it
advances to cross-dataset validation; it is not yet a frozen primary method.

The first MultiHop-RAG guard attempt failed before retrieval because a prior
MuSiQue clear-graph rebuild had removed the corpus-prefixed MultiHop-RAG nodes
and indices. Its all-error zero-valued artifact
`mhr-dev-current-control-raw-owner-20260830` is invalid and must not be used in
any comparison. MultiHop-RAG was then rebuilt from all 609 documents without
clearing MuSiQue. The cache-assisted development rebuild completed 609/609
with zero errors and all integrity checks passing, producing 8,529 chunks and
21,874 HOP edges in 582.49 s. That time is not a cold indexing-cost result.
Both corpora and all 14 Prehop search indices were verified present and ONLINE
before the guard was rerun.

On that shared active database, the fixed MultiHop-RAG control and the
evidence-conditioned candidate had identical official retrieval results:
Hits@4 0.7133, Hits@10 0.8600, MRR@10 0.5632, and MAP@10 0.2743. Every
per-query retrieval difference was exactly zero. Candidate answer EM/F1 moved
from 0.2067/0.2120 to 0.2000/0.2053, but both paired intervals included zero;
mean latency moved from 3.42 s to 3.55 s. The candidate therefore passes the
cross-dataset retrieval no-regression gate and remains the development
checkpoint. It still falls short of the predeclared MuSiQue target, so no
default or paper-primary claim is changed at this stage.

Per-query trace comparison explained the remaining gap. The two-stage policy
raised support recall on 36 queries and lowered it on six. It introduced 38
additional gold paragraphs, 36 of which were directly attributable to a new
evidence-conditioned query path. Four-hop questions still had 0.5000 support
recall because one intermediate binding often left another unresolved
relation. A follow-up therefore repeats evidence-conditioned rewriting only
while a new, non-duplicate question selects at least one previously unseen
chunk. It stops on an empty question set or an unchanged selected-chunk set;
there is no fitted iteration count, score threshold, hop label, or dataset
branch. The five-query termination smoke run was functional only and was not
used for model selection.

The fixed development run
`musique-dev-evidence-iterative-raw-owner-20260830` completed without errors.
Relative to the two-stage checkpoint, answer EM/F1 changed from
0.2338/0.3019 to 0.2388/0.3074 and support precision/recall/F1 from
0.1695/0.6766/0.2677 to 0.1728/0.6866/0.2726. The three support improvements
had positive paired 95% intervals; the answer and evidence-document intervals
included zero. Mean latency rose from 7.35 s to 9.92 s. Refinement stopped
after zero, one, two, three, or four rewrite calls for 11, 35, 124, 27, and
four queries respectively. Four-hop answer F1 and support recall improved from
0.1289/0.5000 to 0.1587/0.5299, while two-hop answer F1 decreased from 0.4456
to 0.4158 with unchanged support recall. This candidate advances because all
aggregate target metrics increased, but its latency and mixed hop-level answer
effect remain explicit costs.

The MultiHop-RAG guard again produced exactly the control retrieval values:
Hits@4 0.7133, Hits@10 0.8600, MRR@10 0.5632, and MAP@10 0.2743, with zero
per-query differences on every official retrieval metric. Answer F1 changed
from 0.2120 to 0.2051 with a paired interval including zero, and mean latency
was 3.50 s. The iterative policy is therefore the current development
checkpoint, not a frozen default. It remains below the predeclared MuSiQue
answer and support-recall targets, so further work must improve dependency
coverage rather than add unconditional refinement rounds.

Trace termination was then separated by cause. The 11 four-hop questions
above the rewrite word gate had 0.6818 support recall, while the 42 rewritten
four-hop questions that ended because no further grounded question could be
formed had 0.4821 recall. Removing the long-query gate would therefore target
the stronger subgroup rather than the observed failure. The next diagnostic
instead changed within-role view fusion from reciprocal-rank accumulation to
deterministic round-robin, so each dependency view could contribute a unique
owner before any view contributed its next owner. Cross-role votes and top-k
were unchanged.

The view-round-robin run improved answer EM/F1 from 0.2388/0.3074 to
0.2587/0.3264 and changed support precision/recall/F1 from
0.1728/0.6866/0.2726 to 0.1733/0.6862/0.2732. Every paired interval against
the iterative checkpoint included zero, support recall moved slightly down,
and latency remained 9.73 s. The candidate neither dominated nor produced a
reliable improvement, so it was rejected and its experimental policy code was
removed. The artifact remains under
`data/results/musique-dev-evidence-iterative-viewrr-*`.

The next diagnosis tested whether the remaining failure originated in
query-time rewriting or in the index. On the fixed 201 MuSiQue development
queries, the stored Q+→Q− HOP graph contained only 33 of 402 ordered adjacent
gold paragraph pairs and a complete gold chain for 11 of 201 queries. A
read-only ANN ceiling that retained ten foreign Q− owners per existing Q+
still covered only 109 pairs and 28 complete chains; it covered no complete
four-hop chain. On this development set, the result did not support
candidate-width expansion or pair filtering as a sufficient correction to the
existing Q+ representation.

The same diagnostic examined the grounded answer that connects each official
decomposition step, without exposing those annotations to indexing. The
previous step's answer occurred verbatim in the next gold paragraph for 318
of 402 pairs. Exact answer mentions formed a complete chain for 127 of 201
queries, including 23 of 67 four-hop queries. This is an analysis ceiling, not
an index result, but it supports changing how corpus-only edges are built.

An opt-in `linked_v2` question schema and `HOP_CONTINUE` branch were therefore
implemented. Q− stores a source-verifiable complete answer and marks it as a
continuation anchor only when it is a specific named entity. After the full
corpus is indexed, one deterministic scan links each anchor owner to exact
mentions in foreign documents. Query retrieval preserves matched Q− IDs and
activates only their continuation edges. Candidate bodies are ordered with
the query embedding and bounded by the unchanged final top-k; no benchmark
question, gold paragraph, hop label, score threshold, dataset branch, or new
candidate-width parameter participates. The legacy and grounded-v1 paths are
unchanged. The generation cache identity was also corrected to include model,
declared revision, and seed so a final index cannot silently reuse questions
generated under a different model state. Unit and targeted regression tests
passed before the full-corpus development build was started under the separate
`musique_linked_v2` tag. No primary/default claim changes before its results
and the MultiHop-RAG guard are available.

The first build attempt was stopped after 453 documents when logs showed that
a partial `continuation_anchor` caused the otherwise valid Q− record to be
discarded. Validation now retains the grounded Q− and clears only the invalid
optional anchor. The question-cache version was advanced so those filtered
development records cannot be reused, relevant tests passed, and the complete
corpus build was restarted from generation under a new run ID.

That restart was also stopped during its initial documents because an invalid
auxiliary `anchor_entities` value still discarded a Q− or Q+ whose quote,
answer, and question were valid. The auxiliary list is not used in retrieval
or edge creation. `linked_v2` now keeps only source-verifiable entries in that
list while preserving the grounded question; `grounded_v1` remains strict.
The cache version was advanced again, the focused suite passed, and a clean
full-corpus development build was started under a third run ID.

Before evaluation, a query-time `RAG_CONTINUATION_EDGES_ENABLED` switch was
added so the same completed `linked_v2` graph can be benchmarked with its
continuation edges hidden or exposed. The switch does not change ranking
scores, evidence width, or stored edges, and its value is recorded in each
benchmark artifact. This separates the effect of exact Q− continuation from
the question-schema and index-construction changes.

A read-only check during the third build found a storage problem before the
continuation pass ran. At 505 completed source documents, the partial graph
contained 1,935 distinct answer anchors; `United States` alone occurred in 116
foreign documents. Direct source-owner-to-target materialization would create
the source-question × target-mention product for such answers even though the
query-time candidate paths are identical. No score or benchmark annotation was
involved in this check.

Continuation storage was changed to one shared `AnswerAnchor` per normalized
answer, with `ANSWER_ANCHOR` from Q− and `MENTIONED_IN` to exact corpus
mentions. Query-time traversal still begins at the exact matched Q− ID,
excludes the source document, orders the same target chunks by the query
embedding, and keeps the unchanged final evidence budget. This is a lossless
graph factorization, not a frequency cutoff or retrieval heuristic. A unit
case verifies that two questions and three mentions require five stored links
instead of six materialized source-target links. The full suite passed with
276 tests and 4 skips. A temporary-label Neo4j smoke test also built the shared
anchor and returned only its foreign target through the exact matched Q− ID;
the temporary nodes were removed afterward. The third build was stopped
before edge construction, and a fourth build was started with its valid v3
generation and embedding caches retained; the cache identity and generated
question content are unchanged.

The paired-analysis contract was tightened before the linked-index result was
available. `--expected-ablation-difference` now rejects a query-only pair when
any unlisted ablation metadata changes, so the continuation off/on comparison
must differ only in `continuation_edges_enabled`. Index-changing comparisons
under different corpus tags require the explicit `--allow-index-variant`
override while retaining identical dataset, corpus fingerprint, evaluation
scope, and stable query IDs. This prevents a retrieval setting or evidence
budget change from being hidden inside the continuation comparison. The
query-only contract also requires the active-index snapshot, model identities
and seed, code provenance, benchmark concurrency, and judge state to match.

A second read-only construction audit considered removing the generation
model's optional named-entity marker and treating every grounded Q− answer as
an anchor. On the then-visible 3,026 chunks and 8,145 Q− records, the current
marker produced 857 shared anchors, 1,176 useful question links, and 5,803
mention links. The all-answer construction would produce 1,241 anchors, 1,714
question links, and 12,959 mention links. Shared storage makes that broader
policy feasible, but it changes the retrieval candidate set rather than merely
compressing it. It was therefore not folded into the running candidate before
its result; if needed, it will be evaluated as a separately recorded structural
index policy instead of an unreported adjustment.

The final 10% objective is now executable through
`scripts/performance_gate.py`. MultiHop-RAG is fixed to official Hits@4,
Hits@10, MRR@10, and MAP@10; MuSiQue is fixed to official answer EM/F1 and
support precision/recall/F1. The strongest supplied non-Prehop model is chosen
independently for each metric, and every metric must meet the relative margin.
The command rejects incomplete, corpus-mismatched, query-mismatched, and
sample artifacts for paper use. This gate was added before the linked-index
result, so its success criteria cannot be changed to fit that result.

A dry run against the currently stored MultiHop-RAG full artifacts rejected
the older HopRAG and MS GraphRAG rows because both report
`corpus_index_fingerprint_status=manifest_absent`; the controlled Naive row is
matched. MuSiQue has no completed full-split artifact for the restored
six-sentence controlled Naive configuration. These historical scores remain
development references, but the final gate requires fresh fingerprint-bound
baseline artifacts after the Prehop method is frozen. They cannot be promoted
by copying their aggregate values into the final table. After the comparison
contracts and their tests were added, the full suite passed with 281 tests and
4 skips, and repository-wide Ruff and diff checks passed.

The completed `musique_linked_v2` development index covered all 21,099 corpus
sources with zero failures and passed every integrity check. It contained
10,228 shared answer anchors, 18,974 question-to-anchor links, and 156,499
anchor-to-mention links. The cache-assisted development build took
11,729.95 seconds and its estimated logical property payload was 3.720 GiB;
these are development diagnostics, not the final cold-index cost claim.

On the fixed 201 queries, the linked schema with continuation hidden produced
0.2338 answer EM, 0.2943 answer F1, 0.1731 support precision, 0.6915 support
recall, and 0.2734 support F1. Exposing exact matched-Q− continuation changed
those values to 0.2239, 0.2903, 0.1677, 0.6737, and 0.2652 respectively. The
three support decreases were statistically separated from zero: precision
Δ -0.0054, 95% CI [-0.0103, -0.0009]; recall Δ -0.0178,
[-0.0352, -0.0004]; F1 Δ -0.0082, [-0.0157, -0.0010]. Answer EM and F1
also moved downward, with intervals including zero. Against the prior
iterative checkpoint, continuation-on likewise improved none of the five
official MuSiQue aggregates. The named-entity continuation policy is therefore
rejected rather than adjusted after observing the result. The OFF artifact is
also not selected: it traded lower answer EM/F1 for small support gains and did
not dominate the prior checkpoint.

Only after that rejection, the predeclared broader construction policy was
opened as the next structural candidate. It uses the complete source-grounded
answer of every valid Q− as the exact cross-document anchor instead of relying
on the generation model's optional named-entity marker. The same shared-anchor
factorization, foreign-document exclusion, exact matched-Q− activation,
query-to-body ordering, evidence budget, fixed development IDs, and global
retrieval policy remain unchanged. The policy is index-time metadata and will
use a new corpus tag; it is not a score, frequency cutoff, candidate-width
setting, dataset branch, or change to the operational default.

The implementation records `RAG_CONTINUATION_ANCHOR_POLICY` in index and
benchmark provenance. Its unchanged default is `named_only`; the isolated
candidate is `all_grounded`, which selects `q.answer` only after the linked
schema's existing source-grounding validation. The named marker and grounded
answer continue to coexist on Q− nodes, while only the policy-selected value
builds shared anchors. Focused retrieval, consistency, and benchmark-integrity
tests passed (139 tests), Ruff passed, and `git diff --check` passed before the
new `musique_linked_all_grounded_v1` corpus build was started under run ID
`prehop-musique-linked-all-grounded-dev-20260830`.

The new tag initially caused identical linked-v2 questions to be regenerated
because the cache namespace includes the corpus tag. After 9,657 documents,
the completed r4 cache and the new cache were audited before reuse: all 9,657
filenames intersected, the old cache contained all 21,099 sources, and both
had the exact signature
`schema=linked_v2-prompt=ef46703c-generation=5bf2fddd2895`, which binds the
question prompt, model, declared revision, and seed. The 11,442 missing files
were hard-linked without overwriting the new run's files. This preserved the
same generated questions for the structural comparison and raised throughput
from roughly 110 to 1,200 documents/minute. No source, graph, code, result, or
existing index was changed.

The all-grounded index completed all 21,099 sources with zero failures and all
integrity checks passing. It stored 13,827 shared anchors, 25,609 question
links, and 303,537 mention links. The cache-assisted development run took
6,062.02 seconds and occupied an estimated 3.722 GiB. With continuation hidden,
the fixed development result was 0.2239 answer EM, 0.2901 answer F1, 0.1709
support precision, 0.6812 support recall, and 0.2698 support F1. Enabling the
all-grounded edges produced 0.2239, 0.2779, 0.1672, 0.6758, and 0.2648. EM was
unchanged and the other four official aggregates decreased; every paired
interval included zero. The candidate also remained below the iterative
checkpoint on all five metrics and is rejected without an anchor-frequency,
answer-type, or dataset-specific filter.

A final-selection trace audit found that continuation changed 177 of 201
queries, added 422 paragraphs and removed 407. Eleven added paragraphs were
gold while 17 removed paragraphs were gold, across 11 gain and 12 loss queries.
The edges therefore expose real missing evidence but rank too many irrelevant
mentions into the unchanged final budget. Gold identities were used only for
this post-hoc diagnosis, never as retrieval input.

The next query-only structural candidate follows the existing role contract:
continuation targets are ordered and semantically checked against the Q+
dependency views, which state the unresolved relation the foreign mention must
satisfy, instead of the original compound question. Exact matched-Q− activation,
anchors, graph snapshot, rank fusion, final evidence budget, and every global
policy remain fixed. If no Q+ view exists, the original query remains the
deterministic fallback. The policy is evaluated first as a one-way fixed-sample
gate; a same-code paired control is run only if all five aggregates improve.

The one-way run `musique-dev-continuation-dependency-views-20260830`
completed at 0.2388 answer EM, 0.2879 answer F1, 0.1652 support precision,
0.6687 support recall, and 0.2617 support F1, with 10.97 s mean latency. The
iterative checkpoint was 0.2388, 0.3074, 0.1728, 0.6866, and 0.2726. EM was
unchanged and all four remaining official aggregates decreased, so the
candidate failed its one-way gate. No same-code control or second-dataset run
was started, and the rejected dependency-to-body implementation was removed.

The next bounded diagnostic keeps the exact same index and activation path but
tests a narrower relation contract before any benchmark is allowed. A foreign
anchor mention is eligible only when one of that target chunk's stored Q-
questions matches an unresolved Q+ dependency view. Expansion is ordered by
the maximum Q+-to-target-Q- similarity, while final semantic selection retains
the established conjunction of the original query and the traversed target
Q- representations. This uses only precomputed linked-schema representations,
adds no score weight, threshold, candidate budget, dataset branch, or reindex,
and deterministically falls back to the original policy when no Q+ view exists.

Focused tests passed 141/141 and Ruff passed. A one-query live smoke run then
completed in 6.67 s and exercised one continuation path, confirming that the
target-Q- Cypher branch is valid before the fixed development evaluation.

The fixed run `musique-dev-continuation-target-qminus-20260830` completed at
0.2289 answer EM, 0.2832 answer F1, 0.1667 support precision, 0.6733 support
recall, and 0.2639 support F1, with 11.85 s mean latency. All five official
aggregates were below the iterative checkpoint. The target-Q- candidate was
therefore rejected without a paired control, second-dataset run, or any
similarity cutoff. Its query-time experimental implementation was removed;
the run artifact preserves the negative result.

A read-only reachability ceiling then started from the 12 paragraphs selected
by the all-grounded OFF run. Those paragraphs contained 394 of 603 gold support
units (0.6534 unit-weighted coverage and 0.6812 official query-macro recall).
Following every stored Q- answer anchor from those same paragraphs by one exact
foreign mention would contain 502 gold units (0.8325 unit-weighted coverage and
0.8516 query-macro recall), adding 108 units across 91 queries. Gold source
identities were used only to measure this post-hoc ceiling; they did not choose
an anchor or target. The ceiling exceeds the 0.7681 support-recall target and
localizes the remaining failure to query-conditioned source-fact activation
rather than anchor storage or deeper graph reachability.

The next diagnostic therefore presents only the indexed, source-grounded
Q-/answer facts attached to retrieved paragraphs and asks the generation model
to select the smallest set whose answers bind an unresolved dependency in the
original question. Selection is by opaque fact ID, and any returned ID outside
the supplied set is rejected. Gold answers, paragraphs, hop labels, and target
mentions are absent from the planner input. This is evaluated first as a
candidate-coverage diagnostic; no synthesis benchmark is started unless the
selected facts recover enough missing evidence to justify an implementation.

Across all 201 queries, the planner received 3,214 indexed candidate facts and
selected 200 facts on 147 queries, with zero unknown-ID outputs. Unioning every
foreign mention of those selected anchors raised unit-weighted support coverage
from 0.6534 to 0.7645, adding 67 gold units on 59 queries. It nevertheless
exposed 5,963 target documents. A correction audit found that this early
diagnostic had been compared to the query-macro target using the wrong micro
aggregation; its official-form macro ceiling is 0.7877, not 0.7645. The broad
expansion is still rejected because thousands of unranked targets cannot enter
the unchanged evidence budget, not because its corrected ceiling is too low.

One narrower diagnostic reuses the same grounded-fact selection contract but
requires each selected fact to produce a self-contained follow-up Q- question
that binds its answer to the still-unresolved relation. Those questions search
the existing Q- index with the established hybrid retrieval and top-12 owner
budget. This tests explicit fact-to-question planning rather than exposing all
documents that happen to mention the intermediate answer.

The planner produced 178 valid grounded steps on 146 queries and 162 unique
follow-up questions; 21 malformed or non-answer-binding steps were rejected.
Searching the global Q- index returned 2,010 document candidates, but the
direct-plus-candidate unit-weighted coverage reached only 0.7280, adding 45
gold units on 44 queries. The corrected macro audit produced 176 valid steps
and 20 rejected steps, reflecting the already documented non-bitwise-stable
generation path; its macro ceiling was 0.7537. Both aggregations are below the
target, so global follow-up Q- search is rejected.

The final planner diagnostic preserves both constraints at once: the selected
source fact fixes one exact shared anchor, and its grounded follow-up question
ranks only that anchor's foreign mention bodies. Each step retains the existing
top-12 owner budget. If this constrained candidate ceiling does not exceed the
support-recall target, no planner synthesis benchmark will be run.

The constrained body diagnostic produced the same 178 valid steps and exposed
1,331 mention targets. Direct-plus-target unit-weighted coverage reached
0.7479, adding 57 gold units on 54 queries. Under the corrected query-macro
aggregation, the 176-step audit reached 0.7716 and therefore narrowly passed
the 0.7681 coverage gate. The earlier closure of the planner family is
withdrawn; no synthesis benchmark had been run from that erroneous decision.

The next non-generative diagnostic mirrors the already selected owner-wide HOP
activation rule. Every source-grounded Q- anchor attached to a retrieved owner
is structurally eligible, but an outgoing path is ordered by the conservative
minimum of original-query-to-source-Q- similarity and original-query-to-target-
body similarity. This removes the failed exact matched-Q- bottleneck while
retaining a two-sided semantic conjunction, existing top-12 budget, and exact
foreign mention relation. Gold is used only after ranking to measure coverage.

Original-query owner activation returned 2,363 target documents but added only
36 gold units on 35 queries. Its 0.7131 unit-weighted coverage corresponded to
0.7392 query-macro recall and is rejected without implementation. The final
non-generative diagnostic keeps roles separate: accumulated Q- views score
source Q- facts, accumulated Q+ views score foreign target bodies, and the path
score is their conservative minimum. This is the role-aligned form of owner
activation; it is tested once because the original compound query is not itself
a source-fact or unresolved-relation representation.

Role-aligned owner activation reached 0.7463 unit-weighted coverage and 0.7711
query-macro recall, just above the same predeclared gate. It is preferred for
the next implementation over the 0.7716 planner ceiling because it uses only
the role views already produced by the selected query policy, adds no planner
generation call, and has no generated fact-selection output to validate. This
is a candidate-coverage result, not a benchmark result or default change.

The isolated implementation is guarded by
`RAG_CONTINUATION_SOURCE_ACTIVATION=role_owner`; the unchanged default is
`exact`. Evidence-conditioned preview and refinement passes keep continuation
hidden. After refinement stops, one final retrieval pass first selects the
established direct/NEXT/HOP owners, then ranks at most 12 exact foreign
continuation targets globally with the Q-/Q+ path conjunction. The stronger of
a candidate's established semantic route and this fully specified path enters
the existing unweighted semantic/representation rank fusion. The inherited
representation score uses the same reciprocal one-edge discount as every
other graph-only target. The activation policy is recorded in benchmark
provenance and does not change the index, final evidence count, or defaults.

Focused retrieval, workflow, consistency, and benchmark-integrity tests passed
170/170; Ruff and `git diff --check` passed. A one-query live smoke completed
in 6.29 s, recorded the `role_owner` policy, and selected eight
`continuation_owner` paths without a query or provenance error. The fixed 201
query run is therefore allowed as a one-way gate. A paired control and
MultiHop-RAG run remain prohibited unless all five MuSiQue aggregates improve
over the iterative checkpoint.

The fixed run `musique-dev-continuation-role-owner-20260830` completed at
0.2388 answer EM, 0.3031 answer F1, 0.1694 support precision, 0.6915 support
recall, and 0.2688 support F1, with 11.50 s mean latency. Against the iterative
checkpoint, EM was unchanged and support recall increased by 0.0050, but
answer F1, support precision, and support F1 all decreased. The candidate
failed the one-way gate; no paired control or MultiHop-RAG run was started.
The gap between its 0.7711 candidate ceiling and 0.6915 final recall shows that
the existing rank fusion did not preserve the newly reachable gold evidence.

Before removing the experimental path, one final ranking diagnostic takes the
union of the established 12 selected paragraphs and the 12 globally ranked
role-owner continuation targets. A listwise generation call sees only the
original question and opaque candidate IDs with paragraph text, and selects
the complete set of paragraphs needed to answer. Unknown IDs are rejected;
gold, hop labels, path scores, and retrieval ranks are absent from the prompt.
This is a read-only selector diagnostic, not a benchmark or implementation.

The selector saw 20.17 candidate documents per query on average (maximum 24)
and returned 460 documents total with zero unknown-ID outputs. It achieved
0.7000 support precision and 0.6319 support F1, but only 0.5991 support recall;
20 queries received an empty selection. The high precision confirms that the
model could identify strong evidence, but its conservative omissions make it
unsuitable for the recall objective. Prompt or selection-count tuning was not
attempted. The selector was not implemented, and the rejected role-owner
activation code was removed. After cleanup, focused tests passed 166/166,
Ruff passed, and `git diff --check` passed.

The global role-path pool had only a 0.0030 macro-recall margin above the target,
so any final reranker would have to preserve nearly its entire ceiling. The next
read-only construction instead allocates one foreign mention target to every
grounded Q- fact on the 12 established owners. Within each exact shared anchor,
accumulated Q+ views rank the target bodies and the single best target is kept.
The allocation is defined by the atomic indexed fact rather than a fitted pool
width or score cutoff. Gold is used only to measure the resulting union ceiling.

The per-fact body construction produced 12.81 unique targets per query on
average (maximum 21) but reached only 0.7396 query-macro recall when unioned
with the established evidence. It is rejected without implementation. One
role-aligned representation check keeps the same one-target-per-fact contract
but ranks each anchored target by the maximum Q+-view similarity of its stored
Q- questions rather than its body. No candidate width or policy changes.

Target-Q- ranking produced 12.89 unique targets per query on average and a
0.7400 macro-recall ceiling, effectively identical to body ranking. Per-fact
semantic allocation is closed. The next structural diagnostic distinguishes
an arbitrary mention from an entity's canonical document: a foreign target is
eligible only when its title has the exact normalized token identity of the
shared answer anchor. This uses corpus metadata already present on every chunk
and introduces no score, candidate width, or dataset-specific condition.

Canonical-title targets occurred on 159 queries (7.62 per query on average,
maximum 50) but raised macro recall only from 0.6812 to 0.6932. Exact title
identity is rejected without implementation. A separate coverage diagnostic
now preserves one exact-mention target for each `(source Q- fact, accumulated
Q+ question)` pair. Each Q+ question represents a distinct unresolved relation,
so this allocation follows generated structure rather than a fitted width or
dataset hop label.

The fact-and-question allocation added 16.32 unique targets per query on
average (maximum 52), producing 24.51 sources in the union on average (maximum
59). It raised query-macro recall from 0.6812 to only 0.7483: 42 of 603
additional gold support units were reached across 40 queries. The corresponding
unit-weighted coverage rose from 0.6534 to 0.7231. This remains below the
0.7681 next-stage threshold while adding more candidates than the per-fact
constructions, so it is rejected without implementation or a 201-query
benchmark. Gold was used only for this post-hoc coverage measurement. The
read-only graph query was regrouped by individual relation after its first
equivalent bulk form exceeded Neo4j's transaction-memory limit; neither form
wrote data and the selection rule was unchanged.

The next read-only diagnostic tests whether independently ranking paragraphs
is what discards reachable support under the fixed 12-paragraph evidence
budget. Every exact source-to-target relation receives the lower of two
role-aligned similarities: accumulated Q- views to the stored source fact and
accumulated Q+ views to the target body. Relations are sorted once; both ends
of each relation are admitted together while space remains, then unused slots
retain the established direct order. The only width is the benchmark's fixed
12-output contract. There is no fitted weight, cutoff, hop-count branch, or
gold input.

The paired selection filled all 12 chunk slots but reduced query-macro support
from 0.1709 precision / 0.6812 recall / 0.2698 F1 to 0.1564 / 0.6310 /
0.2478. It changed 190 queries: recall improved on 35 and regressed on 50.
An average of 10.85 relations (maximum 12) were admitted because endpoints
were frequently shared, but preserving those complete relations displaced
more established gold evidence than it recovered. The construction is
rejected without implementation or a benchmark run.

The next diagnostic changes the unit of coverage instead of another score
formula: every distinct generated Q- and Q+ view should retain its best direct
paragraph before remaining slots follow the established global order. This
tests whether fusion is suppressing an unresolved relation. One paragraph per
generated view is a structural ownership rule, not a fitted quota; duplicate
paragraphs share a slot, and the final evidence budget remains 12.

On the all-grounded linked index, 842 generated views collapsed to 2.92 unique
owners per query on average (maximum 8); no query exceeded the 12-paragraph
budget. The rule changed 40 queries and moved query-macro support from 0.1709
precision / 0.6812 recall / 0.2698 F1 to 0.1716 / 0.6841 / 0.2710. Recall
improved on four queries and regressed on two. This is a small gain, but that
index remains below the iterative checkpoint, so it does not permit a
benchmark by itself.

The same read-only diagnostic was therefore repeated against the actual
iterative checkpoint and its untouched `musique` index. Its 829 generated
views collapsed to 2.87 unique owners per query on average (maximum 8), again
with no budget overflow. It changed 44 queries and improved all three support
aggregates from 0.1728 precision / 0.6866 recall / 0.2726 F1 to 0.1737 /
0.6895 / 0.2740. Recall improved on four queries and regressed on two. This
passes the retrieval-only screen, but answer metrics still require a real
run.

The isolated implementation is selected with
`RAG_SOURCE_SELECTION_VARIANT=view_owners`; `global` remains the default.
Each per-view first result is carried through within-role fusion and removed
from public result rows after final selection. Focused retrieval, workflow,
consistency, and benchmark-integrity tests passed 168/168; Ruff and
`git diff --check` passed. The fixed first development query then completed a
live smoke in 4.20 s with 12 evidence paragraphs, 1.0 support recall, no
runtime error, and the selected policy present in benchmark provenance. The
201-query run is now allowed as a one-way gate; no MultiHop-RAG run is allowed
unless all five MuSiQue aggregates improve over the iterative checkpoint.

The fixed run `musique-dev-view-owners-20260830` completed all 201 queries at
0.2438 answer EM, 0.3070 answer F1, 0.1733 support precision, 0.6857 support
recall, and 0.2732 support F1, with 9.90 s mean latency. Relative to the
iterative checkpoint, EM increased by 0.0050, support precision by 0.0005, and
support F1 by 0.0006, while answer F1 decreased by 0.0004 and support recall by
0.0008. It therefore failed the required all-five one-way gate. No paired
control or MultiHop-RAG run was started, and the experimental selection path
was removed.

The next read-only diagnostic addresses a different role boundary. A generated
Q+ view states the information still needed from another passage, but the
selected path searches it only through stored question representations and
graph edges. The diagnostic also searches each accumulated Q+ view directly
against chunk bodies. Its first body result owns one final slot, duplicate
owners share a slot, and remaining slots retain the checkpoint order. It also
reports the union ceiling of the standard 12 body results per view. The final
budget remains 12, and no score weight, cutoff, dataset branch, or gold input
participates in selection.

The 363 accumulated Q+ views produced 1.48 unique first-body owners per query
on average (maximum 5), with no output-budget overflow. Preserving those owners
changed 38 queries and improved query-macro support from 0.1728 precision /
0.6866 recall / 0.2726 F1 to 0.1757 / 0.6973 / 0.2772; recall improved on
seven queries and regressed on one. However, the union of every standard
top-12 body result exposed 16.80 unique candidates per query on average and
still reached only 0.7521 macro recall, below the 0.7681 target. Direct Q+-body
routing therefore cannot close the required coverage gap and is rejected
without implementation.

One final direct-body diagnostic applies the same first-result ownership rule
to both generated roles: Q- views search bodies for their answer-bearing
passage and Q+ views search bodies for the still-needed passage. All owned
paragraphs precede the checkpoint order, duplicates share a slot, and the
unchanged output budget is 12. The broader union ceiling is measured before
any implementation is considered.

Across 829 accumulated role views, the combined rule preserved 2.67 unique
body owners per query on average (maximum 7), so no query exceeded the output
budget. It changed 77 queries and improved query-macro support from 0.1728
precision / 0.6866 recall / 0.2726 F1 to 0.1786 / 0.7094 / 0.2817. Recall
improved on 15 queries and regressed on two. The union of all standard body
results reached 0.8076 macro recall, above the 0.7681 target, although it was
broad at 31.76 unique candidates per query. The first-result construction is
therefore allowed for an isolated implementation and one-way benchmark; the
broad union is diagnostic only.

The implementation will expose the rule only through
`RAG_SOURCE_SELECTION_VARIANT=role_body_owners`; `global` remains unchanged.
Auxiliary body owners do not receive a representation vote or seed graph
expansion unless the established Q-/body/Q+ path independently retrieved the
same paragraph. Their internal ownership order is removed from public rows,
and only the final selector admits them before filling from the unchanged
global order.

Focused retrieval, traversal, workflow, consistency, and benchmark-integrity
tests passed 169/169; Ruff and `git diff --check` passed. The fixed first
development query completed a live smoke in 4.17 s with 12 evidence
paragraphs, 1.0 support recall, and four distinct generated role views present
on body retrieval paths. The selected policy was recorded in benchmark
provenance and no runtime error occurred. A 201-query MuSiQue run is therefore
allowed as a one-way gate; no paired control or MultiHop-RAG run is allowed
unless all five target aggregates improve over the iterative checkpoint.

The fixed run `musique-dev-role-body-owners-20260830` completed all 201
queries with zero errors at 0.2587 answer EM, 0.3273 answer F1, 0.1774 support
precision, 0.7052 support recall, and 0.2800 support F1. Relative to the
iterative checkpoint, the changes were +0.0199, +0.0199, +0.0046, +0.0187,
and +0.0074 respectively; mean latency increased from 9.92 s to 10.88 s. The
candidate passes the all-five one-way gate. A same-code `global` control is
now required to separate the structural change from the documented
non-bitwise-stable generation path before the MultiHop-RAG guard.

The first candidate and its planned control had identical functional code,
but the changelog update above occurred between them and therefore changed the
recorded source-tree hash. The strict paired-analysis contract correctly
rejected that provenance mismatch. No artifact was relabelled or relaxed; the
candidate was rerun once without any intervening source edit. The resulting
candidate and control both record source-tree hash
`dfca9a6a2ea357e06bae12887913ab428b4b031acad0714616c6ec0cc8b2f0c8`.

In that exact-source pair, the `global` control scored 0.2488 answer EM,
0.3149 answer F1, 0.1716 support precision, 0.6820 support recall, and 0.2708
support F1. The body-owner candidate scored 0.2687, 0.3372, 0.1772, 0.7048,
and 0.2796 respectively, with deltas +0.0199, +0.0224, +0.0056, +0.0228,
and +0.0089. Query-level paired bootstrap intervals were [-0.0100, 0.0498]
for EM, [-0.0074, 0.0549] for answer F1, [0.0010, 0.0104] for support
precision, [0.0058, 0.0406] for support recall, and [0.0018, 0.0162] for
support F1. Thus all five aggregates improved, and all three support gains
were statistically separated from zero.

The same exact-source candidate/control pair was then run on the fixed 200
MultiHop-RAG development queries with the same iterative rewrite and raw HOP
policy. Candidate versus control official retrieval was Hits@4
0.7333 versus 0.7133, Hits@10 0.8533 versus 0.8533, MRR@10 0.5714 versus
0.5626, and MAP@10 0.2802 versus 0.2739. MAP@10 had a positive paired
interval [0.0007, 0.0139]; Hits@4, MRR@10, and every other retrieval interval
included zero, while Hits@10 was identical per query. Fact recall@4 also
increased by 0.0189 with interval [0.0067, 0.0344]. Answer EM/F1 decreased by
0.0067 and official QA accuracy by 0.0050, but each interval ended at zero and
did not show a statistically separated regression. The candidate therefore
passes the established cross-dataset retrieval no-regression gate and becomes
the new development checkpoint. It is not a final method or default: its
MuSiQue support recall of 0.7048 remains below the predeclared 0.7681 target.

The next read-only selector continues the body-ownership rule in complete
rank rounds rather than stopping after each view's first result. Every
generated role view contributes its result at rank one before any view can
contribute rank two, and so on until the fixed 12-paragraph budget is full.
Duplicates share a slot; candidates within the same round retain their
existing vector/full-text fused score, and the checkpoint order fills only an
unfilled remainder. There is no per-view depth, fitted quota, score cutoff,
dataset branch, or gold input.

The round selector changed 190 of 201 queries. It used 4.65 result-rank rounds
per query on average (maximum 12) and improved query-macro support from 0.1772
precision / 0.7048 recall / 0.2796 F1 to 0.1809 / 0.7289 / 0.2864. Recall
improved on 34 queries and regressed on 20. The gain is materially larger than
the earlier first-result screen and improves all support aggregates, so an
isolated implementation and one-way answer benchmark are allowed even though
the result remains below the final 0.7681 recall target.

The isolated implementation keeps every standard auxiliary body result and
completes one result rank across all generated views before considering the
next rank. Auxiliary-only paragraphs still receive no representation vote and
cannot start graph traversal. Focused retrieval, traversal, workflow,
consistency, and benchmark-integrity tests passed 171/171; Ruff and
`git diff --check` passed. The fixed first development query then completed a
live smoke in 4.94 s with 12 evidence paragraphs, 1.0 support recall, the
intended `role_body_rounds` policy in provenance, no transient selection
metadata in the public result, and no runtime error. A fixed 201-query run and
an unchanged-source first-result control are therefore allowed.

The fixed 201-query candidate and its immediately following first-result
control both completed with zero errors and the same query/evaluation source
hash, `2da805d66bb12ea4637b6476b25bbaa4e2c8816d55ef0c60fa49e16c163c47eb`.
The candidate scored 0.2985 answer EM, 0.3715 answer F1, 0.1807 support
precision, 0.7326 support recall, and 0.2865 support F1. The control scored
0.2637, 0.3323, 0.1791, 0.7102, and 0.2825 respectively, giving candidate
deltas of +0.0348, +0.0392, +0.0016, +0.0224, and +0.0040. Thus all five
predeclared aggregates improved.

Their paired 95% intervals were [-0.0050, 0.0796] for answer EM, [-0.0029,
0.0821] for answer F1, [-0.0056, 0.0084] for support precision, [-0.0041,
0.0493] for support recall, and [-0.0072, 0.0149] for support F1. None was
separated from zero, so this is a development checkpoint rather than an
isolated effectiveness claim. Evidence-document precision/F1 changed by
-0.0058/-0.0036 with intervals spanning zero; these are additional diagnostic
metrics rather than MuSiQue's paragraph-support target metrics. The
predeclared all-five aggregate gate permits the fixed MultiHop-RAG
no-regression check. The candidate is still not final because 0.7326 support
recall remains below the 0.7681 target.

The fixed 200-query MultiHop-RAG candidate/control runs also completed with
zero errors and the same query/evaluation source hash,
`6ae35b047cc6f5b0e6ea3b8f39ee274c775a1ce3d87540bdc7cfe596a0e7f758`.
Candidate versus first-result control official retrieval was Hits@4 0.7133
versus 0.7133, Hits@10 0.8533 versus 0.8533, MRR@10 0.5647 versus 0.5659,
and MAP@10 0.2777 versus 0.2777. Their paired deltas and intervals were 0.0000
[0.0000, 0.0000], 0.0000 [-0.0200, 0.0200], -0.0012 [-0.0134, 0.0096],
and effectively 0.0000 [-0.0041, 0.0036].

Fact recall@4 changed by -0.0022 with interval [-0.0067, 0.0000], while fact
recall@10 changed by +0.0033 with interval [0.0000, 0.0100]. Evidence-document
precision/recall/F1 changed by +0.0020/+0.0033/+0.0016, all with intervals
spanning zero. Answer EM/F1 and official QA accuracy were exactly unchanged.
No official retrieval or answer metric showed a statistically separated
regression, so the cross-dataset no-regression condition passes. The complete
rank-round implementation becomes the next development checkpoint, not a
final method or default.

Several read-only selections then tested how much of the reachable body pool
could be retained without fitting a score weight, cutoff, or dataset branch.
Adding the established global order as one more rank-synchronised list raised
support precision/recall/F1 from 0.1766/0.7334/0.2814 to
0.1795/0.7475/0.2862, but remained below the 0.7681 recall target. Collapsing
body results into separate Q-/Q+ role lists reached 0.1762/0.7355/0.2810;
collapsing all generated-body results into one list reached
0.1754/0.7338/0.2798. Both are rejected because at least one aggregate
regressed.

Using the established global order only to break ties inside a body-rank wave
reached 0.1783/0.7417/0.2842, improving four queries without a recall loss but
still missing the target. Admitting only complete waves below the fixed budget
and filling the remainder globally reached 0.1774/0.7396/0.2830. After the
user explicitly allowed more than 12 outputs, completing the first wave that
crossed 12 produced 13.03 paragraphs per query on average (maximum 19). Recall
rose to 0.7446, but precision/F1 fell to 0.1656/0.2678, so merely expanding
the output is rejected. Treating all generated-body searches as a fourth
representation channel also regressed to 0.1721/0.7177/0.2744. No output-width
or score sweep was performed.

The remaining diagnostic changes the operation rather than another ranking
formula. It forms a candidate pool from every standard generated-view body
result plus the established global top 12. A single list-ranking call sees the
original question and opaque candidate IDs with title and paragraph text, and
must return every ID from most to least useful. Unknown IDs are ignored;
duplicates are collapsed and any omitted known IDs retain their input order at
the end. The first 12 form the evidence set. Gold labels, retrieval paths,
scores, ranks, and dataset identity are absent from the prompt. This differs
from the earlier rejected variable-size selector: it produces a complete
ranking and preserves the unchanged 12-paragraph evidence count.

A ten-query format smoke returned nine complete rankings; the remaining case
omitted one of 36 known IDs and was completed by the fixed input-order rule.
The full read-only 201-query diagnostic then used 35.71 candidates per query
on average (maximum 80), with zero unknown IDs and zero JSON failures; 181
rankings were complete before deterministic completion. Support
precision/recall/F1 improved from 0.1770/0.7351/0.2821 to
0.1924/0.7968/0.3064. It changed 45 queries, improving recall on 40 and
regressing on five. The result clears the predeclared 0.7681 recall threshold
while improving all three support aggregates, so one isolated implementation,
focused tests, a live smoke, and the all-five 201-query benchmark gate are
allowed. No prompt variant or selection-count tuning was attempted.

The isolated implementation is available only through
`RAG_SOURCE_SELECTION_VARIANT=role_body_list_ranking`; `global` remains the
default. Evidence-refinement previews continue to use complete body-rank
rounds, and the complete list ranking runs exactly once after refinement has
stopped. This preserves the query path used by the read-only diagnostic and
avoids multiplying generation calls. Focused retrieval, workflow,
consistency, and benchmark-integrity tests passed 174/174; Ruff and
`git diff --check` passed. The fixed first development query completed a live
smoke in 12.42 s with exactly 12 public evidence paragraphs, 1.0 support
recall, the intended policy in provenance, no transient selection fields, and
no runtime error. The fixed 201-query all-five gate is therefore allowed.

The first gate attempt was stopped after row 112 because one ranking returned
`C0019`, which was not in that query's candidate pool. The parser incorrectly
made that invalid ID fatal instead of ignoring it; no aggregate from the
incomplete run is used. The parser now discards unknown IDs, then applies the
same duplicate collapse and deterministic completion over known IDs. The
unknown-ID case is covered by the focused test, the two affected test modules
passed 83/83, the full suite passed 292 tests with four skips, and Ruff plus
`git diff --check` passed. The fixed gate is restarted from the first query
under a new run ID so the incomplete artifact cannot be mistaken for a result.

The restarted candidate (`musique-dev-role-body-list-ranking-retry-20260830`)
completed all 201 queries without an error at 0.3284 answer EM, 0.3968 answer
F1, 0.2018 support precision, 0.7944 support recall, and 0.3178 support F1.
The exact-source complete-round control
(`musique-dev-role-body-rounds-list-control-20260830`) reached
0.2985/0.3694/0.1805/0.7301/0.2860. Thus every predeclared aggregate improved
and support recall cleared the 0.7681 target. The paired differences were
+0.0299 answer EM (95% CI -0.0050 to +0.0647), +0.0274 answer F1
(-0.0068 to +0.0625), +0.0213 support precision (+0.0156 to +0.0273),
+0.0643 support recall (+0.0423 to +0.0879), and +0.0318 support F1
(+0.0229 to +0.0413). The three support improvements exclude zero; the two
answer improvements do not. Both runs used source-tree fingerprint
`57f7c5384ecff8b36d3698380f48174613d841206bc0a26d2f35678774efb5a1`,
the same full-index fingerprint, fixed query manifest, and seed; the declared
selection policy is their only expected difference. This passes the first
dataset gate and permits the exact-source MultiHop-RAG no-regression pair.

The exact-source MultiHop-RAG pair then compared the same two policies on the
fixed 200-query manifest. Complete list ranking reached 0.8333 Hits@4, 0.8600
Hits@10, 0.7078 MRR@10, and 0.3716 MAP@10; complete body-rank rounds reached
0.7133/0.8533/0.5647/0.2777. The paired differences were +0.1200 Hits@4
(95% CI +0.0667 to +0.1733), +0.0067 Hits@10 (-0.0133 to +0.0267), +0.1431
MRR@10 (+0.0862 to +0.1991), and +0.0939 MAP@10 (+0.0668 to +0.1221).
No official metric regressed; Hits@4, MRR, and MAP improved with intervals
excluding zero, while the smaller Hits@10 improvement did not. Both runs used
source-tree fingerprint
`7f5d512724790219d5c0c48bdd39418e27346d269c9349a81bbc880cf9108171`
and full-index fingerprint
`87781bfec56d944e9e57c3f0e96dc28ba473d837bf4a48d17fdc5b8690a4a0b8`.
The candidate therefore passes both development gates without a
dataset-specific branch. Its higher generation cost is retained for the final
latency report rather than used to alter the policy after seeing results.

It is not yet a final candidate because the strongest historical non-Prehop
Hits@10 result, 0.8457, implies a 0.9302 ten-percent target, while the
development candidate reached only 0.8600. Starting the long cold runs at this
point would therefore have been unlikely to satisfy the declared gate. A
read-only candidate-coverage audit retrieved the already established complete
representation-and-graph union, without generation and without exposing gold
data to retrieval or ranking. On the 150 judged MultiHop-RAG development
queries, any-hit coverage in that union was 0.9733 (146/150), compared with
0.8533 in its existing first ten; the union averaged 64.88 candidates and had
a maximum of 111. On MuSiQue, reconstructing the final generated query views
from the saved traces increased the gold-support recall ceiling from 0.8130 in
the current ranking pool to 0.8346 in the complete union; the complete union
averaged 67.85 candidates and had a maximum of 116.

The next isolated candidate therefore ranks the complete candidate union that
the retrieval and one-edge traversal stages already produced, rather than
only generated-body results plus the established first 12. It still emits the
first 12 paragraphs, uses the same single complete-list call and opaque-ID
contract, and introduces no new search width, threshold, weight, fitted value,
or dataset branch. This is the user-approved wider-candidate interpretation:
the public evidence count does not increase. No output-width or pool-width
sweep is performed.

The implementation now passes the complete existing `ordered` union to the
same ranking contract. The affected retrieval and workflow modules passed
83/83 tests, the full suite passed 292 tests with four skips, and Ruff plus
`git diff --check` passed. A one-query live smoke completed without an error,
returned exactly 12 public paragraphs with no transient ranking fields, and
recovered both supporting paragraphs. The fixed 201-query all-five gate is
therefore allowed; the smoke row is not used for selection.

The fixed complete-pool run (`musique-dev-complete-pool-list-20260830`)
completed all 201 rows without an error at 0.3333 answer EM, 0.4048 answer
F1, 0.2057 support precision, 0.7968 support recall, and 0.3229 support F1.
This strictly improved the preceding restricted-pool candidate's
0.3284/0.3968/0.2018/0.7944/0.3178. Its exact-source complete-round control
(`musique-dev-rounds-complete-pool-control-20260830`) reached
0.2985/0.3724/0.1805/0.7297/0.2859. Paired differences against that control
were +0.0348 official answer EM (95% CI -0.0050 to +0.0746), +0.0324 official
answer F1 (-0.0028 to +0.0685), +0.0253 support precision (+0.0187 to
+0.0320), +0.0672 support recall (+0.0410 to +0.0954), and +0.0369 support F1
(+0.0264 to +0.0476). All five aggregates improved; all three support
intervals exclude zero. Both runs used source-tree fingerprint
`461b59b12d2dda3558fb777be87b72d41de8a6742efd79525293cbdad69c57da`
and the same full-index fingerprint. Average candidate latency was 35.37 s
versus 11.10 s for the control; that cost is recorded, not tuned away. The
candidate passes the MuSiQue gate and permits its exact-source MultiHop-RAG
pair.

The corresponding fixed MultiHop-RAG run
(`mhr-dev-complete-pool-list-20260830`) completed all 200 rows without a fatal
error at 0.8600 Hits@4, 0.9200 Hits@10, 0.7658 MRR@10, and 0.4184 MAP@10.
All four aggregates improved over the restricted-pool candidate's
0.8333/0.8600/0.7078/0.3716, but Hits@10 remained below the declared 0.9302
projection target. An exact-source control was not repeated because this
candidate had not yet cleared that target.

Failure review found repeated paragraphs from one logical document occupying
multiple top-ten positions after complete-list ranking. A conservative
read-only replay over only the already saved 12 outputs, taking one paragraph
per logical document before a second paragraph from any document, changed
Hits@10 from 0.9200 to 0.9267 and MAP@10 from 0.4184 to 0.4216; it did not have
access to the remaining ranked pool, where the earlier audit measured a
0.9733 any-hit ceiling. Final list selection now preserves the generated
paragraph order within each logical document but consumes one paragraph per
document per round. This is the same dataset-neutral document identity already
used by source balancing, introduces no new count, weight, threshold, or
dataset branch, and still returns exactly the requested top-k paragraphs.
The first smoke command referenced a nonexistent development filename and
failed before loading a query or running retrieval; it produced no benchmark
result. The smoke was restarted with the fixed manifest filename.
The corrected one-query smoke returned 12 paragraphs from 12 logical
documents, exposed no transient ranking fields, and exactly matched the saved
pre-change row on official answer EM/F1 and paragraph-support P/R/F1. The two
affected test modules passed 84/84, the full suite passed 293 tests with four
skips, and Ruff plus `git diff --check` passed. The fixed 201-query all-five
gate is therefore restarted under a new run ID; the smoke row is not used for
selection.

The fixed run (`musique-dev-document-rounds-list-20260830`) rejected this
document-round policy. It reached 0.2687 answer EM, 0.3418 answer F1, 0.1816
support precision, 0.7504 support recall, and 0.2891 support F1, below the
complete-pool candidate's 0.3333/0.4048/0.2057/0.7968/0.3229 on every target
aggregate. MuSiQue can require separate paragraph evidence under one shared
title, so title-level rounds discarded useful evidence despite the positive
MultiHop-RAG replay. The policy failed the first-dataset gate; its code and
test were removed and no MultiHop-RAG run was started. The failed-policy
benchmark artifact is retained for provenance.

The next failure audit identified missing non-evidence provenance rather than
another ranking parameter. MultiHop-RAG questions frequently distinguish
outlet and publication date, and the immutable raw corpus already supplies
`source`, `published_at`, `author`, `category`, and URL for every article.
Preparation previously wrote only the title and body, so all chunks from
similar FTX, Meta, sports, or entertainment articles reached final list
ranking without the outlet/date fields explicitly named by the question.

Preparation now writes an optional, filename-keyed `source_metadata.json`
sidecar. The generic indexer validates and fingerprints that sidecar, attaches
available fields to every chunk after cached evidence generation, and stores
them as non-evidence node properties. Hybrid retrieval and graph traversal
carry the fields without indexing or scoring them. Complete-list ranking adds
the available publisher, publication time, author, and category beside each
candidate title; corpora without the sidecar render the exact prior prompt.
There is no dataset branch, learned value, width, threshold, gold field, or
change to retrieval scores. Existing index tags are not modified; the first
live check uses a new tag. The affected modules passed 124 tests, the full
suite passed 294 tests with four skips, and Ruff plus `git diff --check`
passed.

The first new-tag index attempt was stopped after 96/609 completed files when
three long source names exceeded the filesystem's per-component limit in the
chunk-cache temporary filename. The failures occurred before those documents
were written, so the incomplete snapshot is not a benchmarkable index. Cache
source components now retain short established names but replace names longer
than 96 characters with a bounded prefix plus a full-source digest. This
preserves collision resistance and the existing cache identity inputs without
changing indexed content. The same new tag is restarted; completed short-name
caches remain reusable.

The restarted index (`mhr-metadata-dev-index-retry-20260830`) completed all
609 source files with zero failures and passed every index-quality check. It
contains 609 documents, 8,529 chunks, 21,805 HOP edges, and 2,821 materialized
reciprocal provenance pairs. Its optional source-metadata digest is
`e9ba632f60dac0614360b29afde6f6d6d2d80c42062b6cb2d36cb01eaf991eb8`;
the original corpus fingerprint remains unchanged. A live readback found a
publisher and publication time on all 8,529 chunks and an author on the 7,794
chunks whose raw article supplied one. Total elapsed indexing time was
1,843.62 seconds. The expected warnings for disabled sentence and continuation
representations did not fail the build or its quality checks. The user also
explicitly permitted increasing top-k if the fixed result remains short of the
gate. No width sweep is started before evaluating this metadata-only candidate;
any later increase must use one shared, dataset-independent rule and a new run
ID. A one-query live smoke on the completed index then returned exactly 12
public paragraphs without an error and found the judged document at both
Hits@4 and Hits@10. Its 21.70-second latency was dominated by complete-list
ranking; the smoke row is a plumbing check and is not used for candidate
selection. The fixed 200-query run is therefore allowed.

The first fixed-run attempt was stopped after row 106 because one query with a
large candidate union exhausted all three JSON parse attempts. The ranking
prompt had unnecessarily required the model to repeat every candidate ID even
though only the first 12 can be returned as evidence; the response was cut off
at the generation limit. Continuing would have produced an invalid 200-row
aggregate, so the partial run is rejected. The ranking contract now asks for
exactly the useful `min(top_k, candidate_count)` prefix and retains the same
deterministic known-order completion for missing or invalid IDs. Candidate
generation, ranking input, top-k, retrieval scores, and public evidence are
unchanged. The two affected test modules pass 84/84, Ruff passes, and the diff
has no whitespace errors. The previously failing query is checked directly
before restarting the fixed run under a new run ID.

That direct check completed in 12.61 seconds without a parse retry or runtime
error, returned 12 public sources, and produced a non-empty answer. The full
suite then passed 295 tests with four skips; repository-wide Ruff and
`git diff --check` also passed. The corrected fixed run is started from query
one as `mhr-dev-metadata-prefix-list-20260830`; no partial result from the
stopped attempt is reused.

The corrected fixed run completed all 200 rows without an error or JSON retry.
On the 150 judged retrieval queries it reached 0.9200 Hits@4, 0.9600 Hits@10,
0.8509 MRR@10, and 0.4719 MAP@10, with 28.70 seconds average end-to-end
latency. These exceed the declared ten-percent targets of 0.7527, 0.9302,
0.6226, and 0.3081 respectively. They also improve the prior metadata-blind
complete-pool result of 0.8600/0.9200/0.7658/0.4184 on all four aggregates.
The artifact records all 200 fixed query IDs, a matched 609-source live index
snapshot, completed status, seed 42, source-selection policy
`role_body_list_ranking`, query/evaluation source-tree fingerprint
`c81a2eea8770e94b724600065008238a7bd0712a8f9764775f3702593c109fe6`,
and unchanged corpus fingerprint
`87781bfec56d944e9e57c3f0e96dc28ba473d837bf4a48d17fdc5b8690a4a0b8`.

The first round-control run reproduced 0.7133 Hits@4, 0.8533 Hits@10, 0.5639
MRR@10, and 0.2806 MAP@10, but the paired checker correctly rejected it
because this changelog had been edited between the candidate and control,
changing query/evaluation provenance. No relaxed comparison was accepted. The
new result paragraph was temporarily removed, restoring the candidate's exact
source-tree fingerprint, and only the control was rerun as
`mhr-dev-rounds-metadata-control-exact-20260830`. That exact control again
reached 0.7133/0.8533/0.5639/0.2806 with no errors. The accepted 10,000-sample
paired analysis reports candidate-minus-control differences of +0.2067 Hits@4
(95% CI +0.1400 to +0.2733), +0.1067 Hits@10 (+0.0600 to +0.1600), +0.2870
MRR@10 (+0.2256 to +0.3496), and +0.1913 MAP@10 (+0.1611 to +0.2219).
Every official retrieval interval excludes zero. The metadata-aware candidate
therefore passes the MultiHop-RAG development gate. Because the shared ranking
contract now requests only the useful output prefix, MuSiQue is rerun on its
same fixed 201 IDs before any final cold execution; the prior full-order result
is not treated as proof for the changed contract.

The shared-prefix MuSiQue candidate (`musique-dev-prefix-list-20260830`)
completed all 201 fixed rows without an error at 0.3284 official answer EM,
0.4098 official answer F1, 0.2042 paragraph-support precision, 0.7952 recall,
and 0.3208 F1. Average end-to-end latency fell from the full-order candidate's
35.37 seconds to 24.58 seconds because the model no longer repeats unused
candidate IDs. The exact-source round control
(`musique-dev-rounds-prefix-control-20260830`) reached
0.2935/0.3669/0.1817/0.7355/0.2880. Both artifacts use source-tree fingerprint
`ed6457bf5e7580faa6957b0f673400ccc148a5844d052975089993d183f271bd`.
Candidate-minus-control paired differences were +0.0348 answer EM (95% CI
-0.0050 to +0.0746), +0.0429 answer F1 (+0.0066 to +0.0815), +0.0225 support
precision (+0.0157 to +0.0296), +0.0597 support recall (+0.0336 to +0.0871),
and +0.0329 support F1 (+0.0223 to +0.0437). All five aggregates improved;
answer F1 and all three support intervals exclude zero. Five bounded rewrite
warnings retained the first three unique role questions as specified; there
were no runtime or result-row errors. The candidate therefore retains the
MuSiQue development gate and permits final cold full-split preparation.

The final full-split matrix is now frozen before execution. The public top-k
remains 12: both development gates passed, so the user's permission to increase
it is not used and no width comparison is introduced. Prehop is run first on
new tags `musique_final_20260830` and `multihoprag_final_20260830`; the existing
`musique` and `multihoprag` indexes are not cleared or modified. The repository
cold-run wrapper cannot be used directly because it both requires a clean
tracked tree before the required final-result documentation exists and passes
`--clear-graph`. Manual equivalent commands therefore disable both project
caches, pin the selected global policy and seed, use the full prepared query
files, and omit graph clearing. If both full Prehop artifacts remain above the
historical ten-percent projections, Naive, HopRAG, and MS GraphRAG are rebuilt
and measured on the same new tags, model endpoints, full queries, manifests,
and frozen source tree. No code or documentation edit is allowed between these
final runs. `.env` remains unchanged until the complete performance gate passes.

## 2026-08-29 — Compact-query rewrite selected and indexing path tightened

Query rewriting was corrected so multiple generated views are fused within
Q− or Q+ before the three representation lists are combined. This prevents a
role with three views from receiving three times the rank mass of the unchanged
body role. The rewrite contract now permits an empty role, retains at most the
first three unique questions, and provides replacement and additive-original
view variants.

The unrestricted variants improved MuSiQue sample-201 support retrieval, but
both lowered MultiHop-RAG sample-200 Hits@4. A dataset-neutral input gate was
therefore evaluated on the fixed development IDs. Rewriting is now applied
only to questions of at most 32 words; longer questions preserve their
original explicit source and relation constraints. The body channel always
uses the original question. This setting rewrote 190/201 MuSiQue rows and
8/200 MultiHop-RAG rows.

| Development setting | Mu Answer EM | Mu Answer F1 | Mu Support F1 | MHR Hits@4 | MHR Hits@10 | MHR MRR@10 | MHR MAP@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No rewrite | 0.1393 | 0.1739 | 0.2394 | 0.7267 | 0.8533 | 0.5654 | 0.2750 |
| Rewrite every query, replacement | 0.1642 | 0.2027 | 0.2474 | 0.7133 | 0.8533 | 0.5764 | 0.2891 |
| Rewrite every query, additive | 0.1642 | 0.1968 | 0.2490 | 0.7200 | 0.8533 | 0.5689 | 0.2809 |
| **Replacement, at most 32 words** | **0.1493** | **0.1955** | **0.2463** | **0.7400** | **0.8600** | **0.5726** | **0.2766** |

The unrestricted MuSiQue support-F1 gains were positive under paired
bootstrap, while the gated gain was +0.0062 with a 95% interval spanning zero.
These are development results, not paper claims. The gate was retained because
it improved all four MultiHop-RAG official ranking aggregates instead of
trading Hits@4 for later-rank recall. End-to-end development latency changed
from 2.168 s to 4.038 s on MuSiQue and from 3.358 s to 3.419 s on
MultiHop-RAG.

Exact matched-Q+ activation was rerun with the gated rewrite. With benchmark
seed 42, MuSiQue owner and exact activation had identical answer EM/F1;
support F1 differed by +0.00025 for exact with a paired interval spanning zero.
Owner activation retained the stronger MultiHop-RAG Hits@10 result and remains
the default. Seeded runs still differed on four generated rewrites, so answer
generation is not presented as bitwise deterministic; deterministic retrieval
comparisons and seed metadata remain separate.

MuSiQue graph diagnosis found that most graph effects came from rank changes
among the representation-union candidates rather than large numbers of new
graph-only chunks. No-graph versus exact reciprocal traversal differed by only
+0.0005 support F1 with an interval spanning zero. Depth two, direct-anchored
ranking, source balancing, graph-pair selection, round-robin selection,
candidate-pool doubling, bridge-only scoring, route hints, and adjacency
reordering were rejected on the development IDs. These diagnostics motivate
the pre-declared no-graph, NEXT-only, raw-HOP, and reciprocal-HOP confirmatory
ablations rather than a graph-effect claim from the development sample.

Index-cache loading now backfills the per-chunk title stored only at the top
level by early v1 cache files. Previously a cache hit could fail later graph
writing with a missing `title` key. HOP ANN waves and reciprocal reverse-ANN
pages now run in bounded concurrent groups. Reciprocal results are collected
read-only, grouped per HOP edge, and written once so concurrent updates cannot
lose question IDs. On the current MultiHop-RAG graph the optimized pass
reproduced all 2,753 reciprocal triples exactly with zero additions or
omissions and took 68.8 s. Development rebuilds completed 609/609 and
21,099/21,099 sources with zero failures and all integrity checks passing.
Their cache-assisted wall times are not eligible as paper indexing costs; the
final cost table requires the cold runner with both project caches disabled.

## 2026-08-29 — Controlled Naive configuration restored

The document-level Naive configuration produced a useful architecture-level
sensitivity result, but it changed both the retrieval unit and top-k relative
to Prehop. Naive has therefore returned to the earlier controlled setting:
the shared page-scoped six-sentence chunks, top-k 12, and the shared final
answer prompt. Its vector-only retrieval remains the changed component. The
document-level artifacts are superseded and are not reused as results for this
configuration; both datasets require fresh indexing and benchmarking.

The shared answer prompt now explicitly asks the model to connect intermediate
entities and relations silently and to abstain only when a required evidence
link is missing. A paired development check used frozen retrieved contexts, so
it changed synthesis only. On MuSiQue sample-201, answer EM changed from
0.1095 to 0.1443 and F1 from 0.1506 to 0.1832. On the 150 non-null
MultiHop-RAG sample queries, answer EM changed from 0.1400 to 0.2133 and F1
from 0.1461 to 0.2182; all 50 null queries remained correctly refused. These
are exploratory sample results, not paper results.

The same frozen-context prompt check was applied to HopRAG rather than treating
the shared wording as a Prehop-only advantage. On the MuSiQue development IDs,
HopRAG answer EM changed from 0.1393 to 0.1891 and F1 from 0.1790 to 0.2260.
Accordingly, the final comparison must rerun every shared-prompt system; the
old answer scores are not a valid target under the new prompt. Reordering
Prehop's selected passages so graph neighbours were adjacent was rejected:
MuSiQue F1 changed from 0.18318 to 0.18254 and EM lost one query.

Exposing selected HOP routes as passage-to-passage hints in the synthesis
context was also rejected. With identical frozen passages and the revised
shared answer prompt, MuSiQue sample-201 answer EM/F1 changed from
0.1443/0.1832 to 0.1244/0.1621; the F1 difference was -0.0211 with a paired
95% bootstrap interval of [-0.0437, -0.0013]. On the 150 eligible
MultiHop-RAG development queries, EM/F1 changed from 0.2133/0.2182 to
0.2067/0.2117, while all 50 null queries remained correctly refused. Graph
routes therefore remain retrieval provenance and are not injected into the
answer context.

The fixed MuSiQue development file predated paragraph-identity annotations and
therefore failed the current support-metric input contract. Its 201 IDs and
order were preserved exactly while records were refreshed from the current
full preparation. `refresh_sample_records.py` now performs this annotation
sync without drawing a new sample; the ordered ID digest remained
`f7c5782a...` before and after the update.

## 2026-08-28 — Naive changed to document-level retrieval

Naive RAG now embeds and retrieves exactly one complete prepared source per
unit: one news article for MultiHop-RAG and one paragraph source for MuSiQue.
It uses top-k 10. Embedding input truncation is forbidden, so an input rejected
by the configured 32k endpoint fails indexing. Answer synthesis packs complete
documents in retrieval order within the configured 256k generation budget and
records retrieved and used document counts separately.

This architecture-level baseline replaces the earlier six-sentence Naive
configuration. Its artifacts are retired and must be removed before the
replacement benchmark is run. Prehop's A0–A5 internal ablations provide its
component controls. The generation and embedding input limits in `.env.example`
are now 262,144 and 32,768 tokens respectively.

## 2026-08-28 — Reproducible corpus identity and cold-run entrypoint

MultiHop-RAG preparation now writes a content-bound corpus manifest with
source-content, query-ID, and canonical query-record digests. Full benchmarks
validate available query-record digests as well as the corpus/index
fingerprint before retrieval. The existing MuSiQue manifest contract remains
supported.

`scripts/run_paper_target.sh` now provides one clean-worktree entrypoint for a
single cold index and full benchmark. It disables shared Prehop caches, assigns
run-specific HopRAG and MS GraphRAG output roots, clears Neo4j, fixes query
concurrency at four, and refuses reused run IDs. Model identifiers are
recorded in index and benchmark artifacts.

The HopRAG adapter no longer turns an upstream empty document representation
into an indexing failure. Represented and omitted source counts and digests are
recorded separately while the complete input manifest remains the evaluation
identity. No HopRAG generation, retry, or retrieval setting was changed.

## 2026-08-27 — Strict benchmark recovery and bounded checkpoints

The deterministic benchmark runner can now continue an interrupted
`in_progress` artifact when `RAG_BENCHMARK_RESUME=true`. It validates the full
query identity, strategy, model and retrieval configuration, corpus/index
identity, prior status, unique query IDs, query text, and trace alignment
before retaining any row. Rows with runtime errors are run again. Resume is
rejected when supplemental judging is enabled. Retained and newly executed
query sets keep separate code-provenance records, so recovery does not present
two source states as one.

Incremental artifacts are now written every ten completed queries rather than
after every query, with an unconditional final write. The interval is
configurable and recorded. This bounds repeat work after an interruption while
avoiding repeated rewrites of the complete result, detail, and trace files.
The benchmark log is appended rather than truncated during an explicit resume.

## 2026-08-27 — Index capacity captured before strategy isolation cleanup

Each index run now stores its capacity measurement in the strategy-scoped index
statistics before a later isolated run can clear or replace the artifacts.
The common reporting concept is the size of the persisted index each strategy
uses during retrieval, rather than total workspace or process disk usage.
Prehop, Naive RAG, and HopRAG use the same versioned logical-payload estimate as
the existing MultiHop-RAG table: vector and list elements and graph records are
counted at eight bytes, with selected text-property characters added directly.
MS GraphRAG does not use Neo4j in this repository and records the physical bytes
of its local retrieval artifacts while excluding `_cache`, `_logs`, and
`_input`. Capacity reporting runs after `total_elapsed_seconds` is frozen, so
reporting overhead is not included in indexing time. A failed capacity
measurement marks the corresponding index run incomplete rather than silently
leaving the final cost table blank.

## 2026-08-27 — Complete indexing wall-clock time in run artifacts

All four indexers now write `timing_seconds.total_elapsed_seconds` to their
strategy-scoped index statistics. The value uses one wall-clock definition: it
starts when the indexing command enters the strategy run and ends after index
finalization and integrity checks, immediately before statistics serialization.
This removes the need to recover total indexing time from console logs. Storage
capacity remains a separately reported measurement because the Neo4j systems
use estimated logical property payload while MS GraphRAG uses physical retrieval
artifact size.

## 2026-08-26 — Paper and architecture synchronized to the fixed configuration

The manuscript now describes the selected configuration without promoting its
sample-200 development result to a paper claim: legacy Q−/Q+, owner-wide
activation, materialized reciprocal provenance, full depth-one NEXT/HOP
traversal, and no rewrite. The method is fixed pending the
pre-declared confirmatory evaluation. The primary hypothesis explicitly
requires improvement over representation-matched NEXT-only/no-HOP controls,
not only over external systems.

The approved primary ablation is cumulative from body only through questions
without graph and NEXT only. Raw HOP and reciprocal HOP are two branches from
the same NEXT-only control; A4−A3 measures reciprocal filtering, while A4−A2
measures the complete reciprocal-HOP contribution beyond NEXT.
A5 exact matched-Q+ activation is reported separately as a stricter activation
scope sensitivity analysis, not as an additional component of the main method.
Online/offline reciprocal is classified separately as an implementation
equivalence and efficiency study. `ARCHITECTURE.md` now expresses the same
current defaults in implementation terms and removes P1/P2 experimental
shorthand. No benchmark, index, or code behavior changed in this
documentation synchronization.

The query-time channel selector now includes `body_only`, allowing A0 to reuse
the same complete index as the other performance ablations. The default `full`
path is unchanged. Retrieval tests verify that this variant searches only the
body channel.

A repository-wide Markdown audit also removed duplicated operating details,
corrected stale Exact-Q+ default descriptions, distinguished hypotheses from
established claims, and reduced private submission history. The vendored
HopRAG guide was otherwise preserved; only its local supported-corpus count was
corrected.

Repository hygiene now ignores root-level PDF, PPTX, ZIP, and CSV handoff
exports as well as `_tmp.py` scripts. These are local generated artifacts;
publishable release copies remain deliberate additions rather than accidental
working-tree files. Local PPT generation code, source files, outputs, and
`_workspace/` drafts follow the same rule. `CLAUDE.md` records this policy.

`CLAUDE.md` now requires each document to retain its audience-specific tone and
prohibits unnecessary proper nouns, invented management labels, repeated
slogans, and unexplained experimental shorthand. Actual dataset, system, code,
configuration, and established research names remain available where needed
for precision and reproducibility.

Direct `run_servers.sh` invocation now loads `.env` before applying the empty
API-key fallback. Previously the fallback was exported first and then preserved
as a caller override, causing authenticated external endpoints to fail their
preflight with HTTP 401 even when `.env` contained a valid key.

`scripts/paired_bootstrap.py` now creates its requested output directory before
writing comparison artifacts. A new nested `--out-dir` therefore works without
a separate manual directory-creation step.

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

The Prehop artifact has the 200-query digest `d5facf87...`, with 150 eligible
non-null queries. The obsolete six-sentence Naive row was removed when Naive
changed to document-level retrieval. HopRAG retains official top-k 20 and MS
GraphRAG retains its official LocalSearch context budget, so latency and
resource-cost comparisons are descriptive rather than equal-budget.

| System | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Doc P | Doc R | Doc F1 | Fact R@10 | Answer F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Prehop P2 offline** | **0.7267** | **0.8600** | **0.5679** | **0.2770** | **0.4331** | 0.7000 | **0.5122** | **0.5822** | 0.1472 | 3.408 s |
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
