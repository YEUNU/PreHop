# Result Evidence Register

This register defines the conditions for admitting complete full-run results.
Every reported value must come from a
completed, integrity-checked full-split artifact and retain its recorded
dataset, strategy, model revisions, index identity, and query identity.

## Evaluation configuration

| Setting | Value |
|---|---|
| Generation model and revision | `gemma-4-31b-it` |
| Embedding model and revision | `qwen3-embedding-8b` |
| Embedding dimension | 4,096 |
| Query concurrency | 4 |
| Seed | 42 |
| LLM judge | disabled |

## Admission checks

Every result row must satisfy all of the following conditions:

1. `status` is `completed`, with the expected full query count and no failed
   rows.
2. The evaluated query digest and corpus fingerprint match across all listed
   strategies for the dataset.
3. The index and query traces record `gemma-4-31b-it` generation and
   `qwen3-embedding-8b` retrieval embeddings at 4,096 dimensions.
4. The index snapshot is complete and matches the active corpus manifest.
5. Metrics are recomputed from the saved query rows rather than copied from a
   progress log.
6. MultiHop-RAG and MuSiQue retain separate metrics and denominators.
7. Indexing cost and query latency are reported separately.

## Publication synchronization

- `README.md`, this register, and `docs/prehop_paper.md` must use values read
  from admitted artifacts.
- Presentation sources must use the same values, model identities,
  denominators, and comparison scope.
- Relative changes, chart dimensions, and latency summaries must be recomputed
  from the artifacts; exported presentation files must be regenerated after
  their sources change.
- `scripts/verify_submission_consistency.py` is the manual matrix and document
  check. It is not a CI job.

## Verified evaluation runs

### MultiHop-RAG (2,556 questions)

| Strategy | Run ID | Hits@4 | Hits@10 | MRR@10 | MAP@10 | QA Acc | Null Refusal | Latency | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Prehop** | `20260903_064500_336304548_765319` | 0.9215 | 0.9499 | 0.8229 | 0.4584 | 0.3513 | 0.9967 | 24.50s | Completed |
| **Naive RAG** | `20260903_061601_249614384_747808` | 0.6962 | 0.8439 | 0.5505 | 0.2615 | 0.2868 | 0.9934 | 2.32s | Completed |
| **MS GraphRAG** | `20260903_143020_647669664_1030700` | 0.4541 | 0.5069 | 0.2965 | 0.1479 | 0.3584 | 0.9402 | 12.11s | Completed |
