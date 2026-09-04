# Changelog

This file records changes made after the current 8B configuration was adopted.
Entries are listed in reverse chronological order. `ARCHITECTURE.md` defines
current behavior, and result values belong in `RESULTS.md`.

## 2026-09-03 — MultiHop-RAG Prehop, Naive, and MS GraphRAG evaluations completed

Completed full prepared-split (2,556 queries) evaluation of Prehop, controlled
Naive RAG, and official MS GraphRAG under the clean 8B setting (`gemma-4-31b-it`
generation, `qwen3-embedding-8b` 4096-dim embeddings):
- Prehop achieved 0.9215 Hits@4, 0.9499 Hits@10, 0.8229 MRR@10, 0.4584 MAP@10,
  and 0.3513 QA accuracy with 0.9967 null refusal (latency: 24.50s/query).
- Controlled Naive RAG achieved 0.6962 Hits@4, 0.8439 Hits@10, 0.5505 MRR@10,
  0.2615 MAP@10, and 0.2868 QA accuracy (latency: 2.32s/query).
- Official MS GraphRAG completed with 0.4541 Hits@4, 0.5069 Hits@10, 0.2965
  MRR@10, 0.1479 MAP@10, 0.3584 QA accuracy, and 0.9402 null refusal (latency:
  12.11s/query).
- Prehop outperforms MS GraphRAG by +46.74%p in Hits@4 and +52.64%p in MRR@10.
- Following MS GraphRAG completion, MultiHop-RAG evaluation proceeded to
  official HopRAG / BrowseNet.

## 2026-09-02 — Clean 8B full-system evaluation
 
The embedding endpoint and recorded revision use `qwen3-embedding-8b` with
4,096-dimensional vectors. The generation model remains `gemma-4-31b-it`.

The evaluation runs Prehop, Naive RAG, HopRAG, MS GraphRAG, BrowseNet, and
PropRAG independently on the complete MultiHop-RAG and MuSiQue prepared splits.
Every target clears the graph, constructs a cold index, validates index
integrity, and evaluates the complete query set with concurrency 4 and the LLM
judge disabled. Official BrowseNet and PropRAG execute in isolated runtime
environments while routing semantic retrieval embeddings through the shared
LiteLLM endpoint.

Previous-model results, caches, baseline outputs, logs, and document records
were removed before the new result register was produced.

Documentation responsibilities were reduced to one source per concern.
Component-control definitions now belong to `ARCHITECTURE.md`, while artifact
admission and publication synchronization belong to `RESULTS.md`. The former
standalone ablation protocol and consistency-audit files were removed after
their unique requirements were transferred.

Submission documentation now consistently uses complete prepared splits,
query concurrency 4, and `qwen3-embedding-8b` with 4,096-dimensional vectors.
Former sample-exclusion rules were removed from the maintained workflow.
