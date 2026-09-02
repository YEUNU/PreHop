# Changelog

This file records changes made after the current 8B configuration was adopted.
Entries are listed in reverse chronological order. `ARCHITECTURE.md` defines
current behavior, and result values belong in `RESULTS.md`.

## 2026-09-02 — Clean 8B full-system evaluation

The embedding endpoint and recorded revision use `qwen3-embedding-8b` with
4,096-dimensional vectors. The generation model remains `gemma-4-31b-it`.

The evaluation runs Prehop, Naive RAG, HopRAG, and MS GraphRAG independently
on the complete MultiHop-RAG and MuSiQue prepared splits. Every target clears
the graph, constructs a cold index, validates index integrity, and evaluates
the complete query set with concurrency 4 and the LLM judge disabled.

Previous-model results, caches, baseline outputs, logs, and document records
were removed before the new result register was produced.
