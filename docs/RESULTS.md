# Final Results and Evidence Register

This document is the canonical register for submission-facing numbers. It
separates the two datasets, identifies the stage tested by each internal
control, and links every conclusion to a complete-split artifact.

## 1. Eligible complete-system runs

| Dataset | Evaluated queries | Metric denominator | Query-ID digest | Corpus fingerprint | Gate |
|---|---:|---:|---|---|---|
| MultiHop-RAG | 2,556 | 2,255 answerable for retrieval; 301 null queries separate | `e683d65c…77b0` | `87781bfe…a0b8` | eligible |
| MuSiQue answerable dev | 2,417 | 2,417 | `a66bb869…a8bc` | `7560a211…3735` | eligible |

The full eligibility records are
`data/results/final-multihoprag-performance-gate-20260831.json` and
`data/results/final-musique-performance-gate-20260831.json`. Both record
`paper_eligible=true`. The Prehop summaries record completed status, seed 42,
`gemma-4-31b-it` generation, and 2,560-dimensional
`qwen3-embedding-4b` embeddings.

## 2. Complete-system results

### 2.1. MultiHop-RAG

| System | Hits@4 | Hits@10 | MRR@10 | MAP@10 | Mean latency (s) |
|---|---:|---:|---:|---:|---:|
| Prehop | **0.9268** | **0.9494** | **0.8232** | **0.4550** | 27.63 |
| Naive RAG | 0.6820 | 0.8404 | 0.5442 | 0.2586 | 2.82 |
| HopRAG | 0.6843 | 0.8457 | 0.5660 | 0.2800 | 159.48 |
| MS GraphRAG | 0.4705 | 0.5348 | 0.3033 | 0.1482 | 18.32 |

These retrieval metrics use the 2,255 answerable questions. The 301 null
questions use the separate refusal/hallucination protocol and are not zeros in
the retrieval table. HopRAG and MS GraphRAG are official-setting references,
not equal-budget paired controls. Their point estimates therefore do not carry
same-query paired confidence intervals against Prehop.

### 2.2. MuSiQue

| System | Answer EM | Answer F1 | Support P | Support R | Support F1 | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| Prehop | **0.4150** | **0.5115** | **0.2034** | **0.8840** | **0.3267** | 23.86 |
| Naive RAG | 0.2106 | 0.2726 | 0.1364 | 0.6241 | 0.2215 | 1.88 |
| HopRAG | 0.2619 | 0.3394 | 0.0889 | 0.6982 | 0.1566 | 25.62† |

All values use the 2,417-question answerable development split. HopRAG retains
its official retrieval and context settings. † HopRAG's
reported result uses two complete artifacts over the same 2,417 query IDs.
The displayed 25.62 seconds is their recorded phase sum, 12.53 + 13.09
seconds, rather than a single-pass end-to-end measurement.

## 3. Which stage each MuSiQue test evaluates

| Stage under test | Intervention | Index rebuilt? | Candidate/answer generation | Primary conclusion |
|---|---|---|---|---|
| Stored connection structure | Compare stored Q+→foreign-Q− paragraph links with gold paragraph identities | Same index, read only | Structural coverage | Full connectivity: 2-hop 23.56%, 3-hop 1.32%, 4-hop 0.00% |
| Query-time graph expansion | Compare one-step neighbor expansion on and off | Same index | Separate complete run | ΔSupport F1 +0.00435; ΔAnswer EM +0.00538 |
| Query refinement | Remove evidence-conditioned follow-up views; retain the initial role rewrite | No; same index | Separate complete run | Refinement is a major complete-pipeline contributor |
| Candidate selection policy | Compare question-role selection with the top 12 from one integrated ranking | Same index | Separate complete run | ΔAnswer EM −0.08854; ΔSupport F1 −0.04952 |
| Candidate presentation order | Replay the same candidate texts in the reference and fixed-shuffled orders | Same candidates | Selection and support | ΔSupport F1 −0.00368 |
| Ranking rule | Recompute ranking variants on the same candidate pools | Same candidates | Support | Ranking signals and distance weights change Support F1 |
| Query-time latency | Measure non-overlapping timers in one complete fixed-concurrency run | Same run | Complete run | Generation-model stages 84.9%; graph expansion 1.7% |

Ablation 1–3 reuse the same completed index and compare query-stage behavior.
Ablation 4 and the order robustness test keep the candidate texts fixed. This
separates query-pipeline effects from score- and presentation-order effects.

## 4. Stored connection coverage

### 4.1. Structural coverage

| MuSiQue depth | Questions | At least one stored gold-paragraph edge | Gold-only graph fully connected |
|---|---:|---:|---:|
| 2-hop | 1,252 | 23.56% | 23.56% |
| 3-hop | 760 | 21.97% | 1.32% |
| 4-hop | 405 | 25.68% | 0.00% |

The stored links connect at least one pair of gold paragraphs in about 22–26%
of questions. Full connectivity falls from 23.56% at 2-hop to 1.32% at 3-hop
and 0.00% at 4-hop. The structural source is
`data/results/presentation-p0-analysis/gold_hop_coverage.json`.

### 4.2. Graph-on minus graph-off effects

| Group | N | Δ Answer EM (95% CI) | Δ Support F1 (95% CI) | Δ retrieval passes (95% CI) |
|---|---:|---:|---:|---:|
| All | 2,417 | +0.00538 [−0.00455, +0.01489] | +0.00435 [+0.00269, +0.00603] | +0.00372 [−0.00910, +0.01614] |
| 2-hop | 1,252 | +0.00240 [−0.01038, +0.01518] | +0.00596 [+0.00428, +0.00767] | +0.00559 [−0.00799, +0.01917] |
| 3-hop | 760 | +0.00921 [−0.00921, +0.02763] | +0.00543 [+0.00235, +0.00849] | +0.00789 [−0.01711, +0.03289] |
| 4-hop | 405 | +0.00741 [−0.01481, +0.02963] | −0.00268 [−0.00866, +0.00330] | −0.00988 [−0.05185, +0.03210] |

The Answer EM and retrieval-pass intervals include zero in each hop group.
Support F1 is positive for 2-hop and 3-hop and includes zero for 4-hop. The
paired artifact is
`data/results/presentation-p0-analysis/graph_shortcut_effect_2417.json`.

## 5. Query-stage component controls

All effects below are control minus the complete Prehop result on the same
2,417 query IDs. Separate generation calls make these descriptive paired
complete-split diagnostics rather than decoding-invariant counterfactuals.

| Removed query-stage component | Δ Answer EM (95% CI) | Δ Answer F1 (95% CI) | Δ Support F1 (95% CI) | Δ Support R (95% CI) |
|---|---:|---:|---:|---:|
| Evidence-conditioned refinement | −0.12784 [−0.14398, −0.11212] | −0.14000 [−0.15574, −0.12447] | −0.04811 [−0.05162, −0.04466] | −0.12657 [−0.13508, −0.11802] |
| Role/body candidate-selection policy | −0.08854 [−0.10343, −0.07365] | −0.09898 | −0.04952 [−0.05301, −0.04618] | −0.11481 |

Source: `data/results/presentation-full-analysis/full_component_controls_2417.json`.

## 6. Fixed-candidate controls

These analyses reuse the same candidate texts and report selection and support
metrics.

### 6.1. Candidate input order

| Measure | Complete-split result |
|---|---:|
| Same fused order: recorded vs replay selected-set Jaccard | 0.96760 [0.96363, 0.97129] |
| Fixed shuffle vs same-order replay Jaccard | 0.63909 [0.63090, 0.64717] |
| Calibrated Jaccard difference | −0.32851 [−0.33705, −0.31997] |
| Fixed-shuffle Δ Support F1 | −0.00368 [−0.00535, −0.00201] |
| First-input selection excess over random position | +2.85 percentage points [+1.25, +4.56] |

The ordering call is position-sensitive but does not merely copy the first
candidate. Source:
`data/results/presentation-full-analysis/frozen_candidate_order_replay_2417.json`.

### 6.2. Ranking rule

| Rank variant | Δ Support F1 vs fused, decay 0.5 (95% CI) |
|---|---:|
| Semantic only | −0.03269 [−0.03595, −0.02943] |
| Representation only | −0.00359 [−0.00582, −0.00142] |
| Body semantics only | −0.00028 [−0.00072, +0.00015] |
| Bridge semantics only | −0.00464 [−0.00567, −0.00364] |
| Graph decay 0 | +0.00457 [+0.00327, +0.00592] |
| Graph decay 1 | −0.00870 [−0.01027, −0.00721] |

The signal choice and propagation weight materially change Support F1. On this
split, decay 0 exceeds the default decay 0.5 by 0.00457. Source:
`data/results/presentation-full-analysis/frozen_rank_variants_2417.json`.

## 7. Cost and stage timing

| Dataset | System | Cold indexing wall time (s) | Logical index estimate (GiB) |
|---|---|---:|---:|
| MultiHop-RAG | Prehop | 2,145.63 | 1.545 |
| MultiHop-RAG | Naive RAG | 100.54 | 0.170 |
| MuSiQue | Prehop | 5,676.81 | 4.290 |
| MuSiQue | Naive RAG | 227.82 | 0.456 |

The complete MuSiQue timing profile at concurrency 32 records 131.69 seconds
of accounted mean time per query: rewrite/refinement 57.79 s (43.9%),
retrieval 17.61 s (13.4%), graph expansion 2.25 s (1.7%), deterministic scoring
0.09 s (0.1%), candidate ordering 41.83 s (31.8%), and synthesis 12.12 s
(9.2%). Generation-model stages account for 84.9%. Only these within-run
shares are interpreted; the absolute value is not compared with the official
23.86-second run because concurrency and service load differ. Source:
`data/results/presentation-full-analysis/full_stage_profile_2417.json`.

## 8. Reporting scope

- MultiHop-RAG and MuSiQue use separate metrics and denominators.
- External systems are reference results under their official settings.
- MuSiQue component controls use all 2,417 questions.
- Ablation 1 compares the complete one-step neighbor expansion policy.
- Ablation 3 compares the complete candidate-selection policy.
- Ablation 4 and the input-order test report support and selected-set metrics.
- Stage timing is interpreted as the share measured within one complete run.
