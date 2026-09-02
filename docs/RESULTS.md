# Result Evidence Register

This register defines the current clean 8B run matrix and the conditions for
admitting its complete results. Every reported value must come from a
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

## Run matrix

| Dataset | Strategy | Run ID |
|---|---|---|
| MultiHop-RAG | Prehop | `final-clean-20260902-multihoprag-prehop` |
| MultiHop-RAG | Naive RAG | `final-clean-20260902-multihoprag-naive` |
| MultiHop-RAG | HopRAG | `final-clean-20260902-multihoprag-hoprag` |
| MultiHop-RAG | MS GraphRAG | `final-clean-20260902-multihoprag-ms_graphrag` |
| MuSiQue | Prehop | `final-clean-20260902-musique-prehop` |
| MuSiQue | Naive RAG | `final-clean-20260902-musique-naive` |
| MuSiQue | HopRAG | `final-clean-20260902-musique-hoprag` |
| MuSiQue | MS GraphRAG | `final-clean-20260902-musique-ms_graphrag` |

## Admission checks

Every result row must satisfy all of the following conditions:

1. `status` is `completed`, with the expected full query count and no failed
   rows.
2. The evaluated query digest and corpus fingerprint match across all four
   strategies for the dataset.
3. The index and query traces record `gemma-4-31b-it`,
   `qwen3-embedding-8b`, and 4,096 dimensions.
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
