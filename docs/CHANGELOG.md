# Changelog

This file records changes made after the current 8B configuration was adopted.
Entries are listed in reverse chronological order. `ARCHITECTURE.md` defines
current behavior, and result values belong in `RESULTS.md`.

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
