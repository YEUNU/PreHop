# Result Evidence Register

This register admits results from the current clean 8B matrix only. Each row
must come from a completed, integrity-checked full-split artifact and retain
its recorded dataset, strategy, model revisions, index identity, and query
identity.

## Evaluation configuration

| Setting | Value |
|---|---|
| Generation model and revision | `gemma-4-31b-it` |
| Embedding model and revision | `qwen3-embedding-8b` |
| Embedding dimension | 4,096 |
| Query concurrency | 4 |
| Seed | 42 |
| LLM judge | disabled |

## Full-system matrix

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
