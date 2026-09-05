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

No run is currently admitted. The previous artifacts were invalidated after
the provenance and adapter audit and removed before the isolated
`naacl27-clean-20260905` campaign. Rows will be added only after all admission
checks above pass.
