# Component Evaluation Protocol

Component results are admitted only when they use the current
`qwen3-embedding-8b`, 4,096-dimensional configuration and cover all 2,417
MuSiQue questions. MultiHop-RAG and MuSiQue are not pooled.

## Shared controls

- Reuse one completed Prehop index across paired query-stage conditions.
- Keep query IDs, model revisions, seed, top-k, prompts, and judge state fixed.
- Join results by immutable query ID.
- Report paired effects with 10,000 bootstrap resamples and seed 42.
- Compare latency only within a synchronized, fixed-concurrency run.
- Keep fixed-candidate analyses separate from complete query-pipeline runs.

## Stage map

| ID | Pipeline stage | Intervention | Required output |
|---|---|---|---|
| Ablation 1 | Query: graph expansion | One-step `NEXT` and `HOP_ANSWER` expansion on versus off | Answer, support, retrieval passes |
| Ablation 2 | Query: refinement | Evidence-conditioned follow-up views on versus initial rewrite only | Answer and support |
| Ablation 3 | Query: candidate selection | Question-role selection versus integrated top 12 | Answer and support |
| Ablation 4 | Fixed candidates: ranking | Recompute rank signals and graph-distance weights | Support |
| Robustness | Fixed candidates: presentation order | Reference order versus deterministic shuffle | Selected-set overlap and support |
| Timing | Complete query path | Non-overlapping stage timers | Within-run stage shares |

## Ablation 1. Query-time graph expansion

- **Intent.** Measure the contribution of one-step stored-neighbor expansion.
- **Changed condition.** Set graph depth to 1 or 0 on the same completed index.
- **Fixed conditions.** Query text, query views, candidate-selection policy,
  top-k, models, seed, prompts, and synthesis.
- **Primary metrics.** Answer EM, Support F1, and retrieval passes.

## Ablation 2. Evidence-conditioned query refinement

- **Intent.** Measure the contribution of follow-up views generated from
  retrieved evidence.
- **Changed condition.** Compare iterative evidence-conditioned refinement
  with the initial role-aligned rewrite only.
- **Fixed conditions.** Index, initial rewrite, candidate selection, top-k,
  models, seed, prompts, and synthesis.
- **Primary metrics.** Answer EM and Support F1.

## Ablation 3. Candidate-selection policy

- **Intent.** Measure the contribution of selecting candidates by question
  role rather than one integrated rank.
- **Changed condition.** Compare question-role selection with the integrated
  top 12.
- **Fixed conditions.** Index, query views, scoring signals, top-k, models,
  seed, prompts, and synthesis.
- **Primary metrics.** Answer EM and Support F1.

## Ablation 4. Ranking rule

- **Intent.** Measure how rank signals and graph-distance weights change
  evidence selection.
- **Changed condition.** Recompute variants over identical candidate texts and
  paragraph IDs.
- **Fixed conditions.** Candidate pool, query, paragraph text, and evaluation
  annotations.
- **Primary metric.** Paragraph Support F1.

## Candidate-order robustness

- **Intent.** Measure whether candidate presentation order changes the selected
  evidence set.
- **Changed condition.** Replay identical candidates in the reference order and
  a query-specific deterministic shuffle.
- **Fixed conditions.** Candidate IDs, titles, texts, metadata, model, prompt,
  top-k, and shuffle seed.
- **Primary metrics.** Selected-set Jaccard and Support F1.

## Query-stage timing

- **Intent.** Attribute online processing time to query refinement, retrieval,
  graph expansion, deterministic scoring, candidate selection, and synthesis.
- **Measurement.** Record non-overlapping timers in one complete run at fixed
  concurrency.
- **Primary metric.** Each stage's share of accounted query time.
