# Submission Consistency Audit

This audit covers the current clean 8B evaluation matrix, manuscript,
presentation, and result register.

## Configuration contract

| Field | Required value |
|---|---|
| Generation model and revision | `gemma-4-31b-it` |
| Embedding model and revision | `qwen3-embedding-8b` |
| Embedding dimension | 4,096 |
| Query concurrency | 4 |
| Seed | 42 |
| LLM judge | disabled |
| Evaluation scope | complete prepared split |

## Dataset contract

| Dataset | Full query count | Metric denominator |
|---|---:|---|
| MultiHop-RAG | 2,556 | Retrieval: 2,255 answerable; 301 null queries reported separately |
| MuSiQue answerable development split | 2,417 | Answer and support: 2,417 |

The two datasets use separate tables. MultiHop-RAG reports Hits@k, MRR@10,
and MAP@10. MuSiQue reports Answer EM/F1 and paragraph Support
precision/recall/F1.

## Matrix checks

For each dataset, the four strategy artifacts must agree on:

1. evaluated query count and sorted query-ID digest;
2. corpus manifest fingerprint;
3. generation and embedding model revisions;
4. embedding dimensions;
5. seed, concurrency, and judge state;
6. full-benchmark scope and completed status.

Each strategy must also record a complete index snapshot and its own cold
indexing measurement. A partial checkpoint cannot enter a result table.

## Document checks

- `README.md`, `docs/RESULTS.md`, and `docs/prehop_paper.md` must print values
  from the admitted current artifacts.
- Both presentation sources must use the same values, model identities,
  denominators, and comparison scope.
- Derived values such as relative improvements, chart widths, and latency
  summaries must be recomputed from those artifacts.
- PDF exports must be regenerated after the HTML sources change.
- `docs/CHANGELOG.md` records only changes made under the current 8B
  configuration.
- `scripts/verify_submission_consistency.py` is a manually invoked check; no CI
  workflow is required.
