# Prehop repository guide

The current system evaluates Prehop, Naive RAG, official HopRAG, and official
MS GraphRAG on MultiHop-RAG, HotpotQA, and MuSiQue. The detailed module and
branch map is in `docs/ARCHITECTURE.md`.

## Supported state

- Python 3.12, dependencies managed by `uv` from `pyproject.toml`.
- Neo4j is the only local service. Generation and embeddings are mandatory
  external OpenAI-compatible endpoints configured by `VLLM_URL` and
  `VLLM_EMBED_URL`; there is no localhost/model-launch fallback.
- The embedding dimension in `NEO4J_VECTOR_DIMENSIONS` must equal the endpoint's
  actual vector length. Startup probes validate the configured model ids and
  dimensions.
- The server capacity is `VLLM_MAX_NUM_SEQS=128`. Generation uses one global
  per-target/event-loop semaphore (`MAX_CONCURRENT_LLM_CALLS`, default 30), not
  one limit per document. Embeddings default to batches of 32 with two requests
  per target. The full-matrix runner detects whether generation and embedding
  share a URL; for the current shared endpoint and width 2 it lowers embedding
  concurrency to one, keeping the combined aggregate bound at 124. Separate
  URLs are budgeted and pressure-sampled independently.
- A dedicated reranker is not used. The external embedding endpoint supplies
  vectors for threshold-free cosine top-k ordering. Final selection caps each
  source document at `RAG_MAX_CHUNKS_PER_SOURCE_FRACTION` (default 0.34) of
  top_k so one strongly-matching document cannot fill the whole evidence set.
- Prehop has no query rewrite, rerank simplification prompt, continuation
  prompt, runtime HOP, company/domain gate, metadata boost, boilerplate
  penalty, table-to-text generation, or Q+ heuristic post-filter.
- Prehop retrieval has no generation call. Its only query-time generation is
  final answer synthesis after deterministic retrieval/traversal.
- Official baselines keep their upstream behavior. In particular, official
  HopRAG `bfs_node` uses its published LLM node judgement during retrieval;
  this is documented baseline behavior and is routed to the external endpoint.
- Benchmark LLM judging uses OpenAI Batch by default and never silently falls
  back to synchronous paid calls. Disable it only for an explicit debug run.

## Data and tags

| Tag | Corpus directory | Full query file |
|---|---|---|
| `multihoprag` | `data/multihoprag_corpus` | `data/multihoprag_queries.json` |
| `hotpotqa` | `data/hotpotqa_corpus` | `data/hotpotqa_queries.json` |
| `musique` | `data/musique_corpus` | `data/musique_queries.json` |

Corpus files use `Title: ...`, optional `--- Page N ---` markers, then raw
text. Prehop and Naive share the parser and page-scoped fixed sentence-window
chunker. Raw pipe text remains raw.

## Common commands

```bash
# Validate/start Neo4j and validate both external endpoints
./run_servers.sh all

# One target
./run_index.sh --model prehop --dataset data/multihoprag_corpus --corpus-tag multihoprag

# One benchmark
./run_benchmark.sh --model prehop \
  --queries data/multihoprag_sample200_queries.json \
  --corpus-tag multihoprag

# Remove all graph data and application schema
.venv/bin/python main.py --mode clear_graph

# Rebuild only Prehop HOP/provenance edges after changing rank-fusion settings
.venv/bin/python main.py --mode hop_rebuild --strategy prehop --corpus-tag multihoprag

# Full cold measured matrix
.venv/bin/python scripts/run_index_matrix.py --clear-graph --max-parallel 2

# Resume an already submitted OpenAI Batch judge after interruption
.venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<run-id>

# Verification
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

`run_servers.sh` never starts a model. If either inference endpoint or served
model id is missing/unreachable, indexing and benchmarking fail before work.

## Full matrix behavior

`scripts/run_index_matrix.py` runs 12 targets: three datasets by four
strategies. It starts with at most two targets concurrently, samples host and
inference pressure, and reduces the remaining width when pressure is sustained.
It does not increase width again within the same run. Each child is a separate
process group, so interruption terminates descendants and prevents overlapping
debug/index jobs.
At most one of Prehop/HopRAG/MS GraphRAG runs at once by default; these are all
generation-heavy. The scheduler scans ahead for Naive targets to fill the
second slot, preserving parallel embedding work without generation contention.
Naive batches 32 source documents per embedding/write transaction. Prehop's
outer in-flight file cap defaults to 16; its generation semaphore remains 30,
so short one-chunk corpora can use the endpoint without making long-document
fan-out unbounded.
Official HopRAG defaults to 10 document workers × 4 chunk threads, and MS
GraphRAG to 32 concurrent requests. Since no other generation-heavy target is
allowed beside either one, their upper bounds remain within max-seq 128.

A measured cold run must:

1. stop any prior matrix process;
2. clear Neo4j nodes, constraints, and non-lookup indexes;
3. remove prior `data/index_cache`, HopRAG outputs/caches, MS GraphRAG outputs,
   debug, failure, and stats artifacts for the selected targets;
4. set `RAG_CHUNK_CACHE=off` and disable baseline resume reuse;
5. use a new `RAG_RUN_ID` and artifact directory;
6. run endpoint/model/dimension preflight before launching targets.

Each target writes an isolated stdout/stderr log and resource record. The final
run directory includes `summary.csv`, a paper-ready Markdown table, pressure
samples, manifest/settings, and integrity counts. A target with any document,
workflow, graph-finalization, or measurement failure is failed, never silently
classified as complete.

## Prehop inspection during indexing

For Prehop targets, inspect real intermediate files and the live graph rather
than relying only on progress logs:

- sample `data/debug/<run-id>/prehop/<corpus>/<source>/final_chunks.json` when
  `--save-intermediate` is enabled;
- verify raw chunk text, title/page/sent_id ordering, grounded Q-, outward Q+,
  summary, and absence of fabricated/table-converted text;
- query total Documents/Chunks and Q-/Q+ coverage;
- after the final pass, inspect HOP direct-channel provenance, per-source
  out-degree, cross-source property, and representative Q+→Q-/body/SAME_NEED
  source/target text;
- compare source file count to indexed Document count and fail on any mismatch.

Debug output is namespaced by run, strategy, corpus, and source. Index logs and
paper measurement artifacts use separate directories, so simultaneous targets
cannot overwrite one another.

## Comparison policy

Prehop and Naive use the same chunks, shared synthesis prompt, and top-k 12.
HopRAG retains the upstream official end-to-end top-k 20; MS GraphRAG retains
its official context budget. These unequal official settings must be stated in
the paper. A controlled equal-budget retrieval experiment, if added, must be
reported separately from official-baseline results.

## Generated files and repository hygiene

Generated logs, caches, debug output, graphs, results, index outputs, and matrix
artifacts are not source files and must remain ignored. Do not commit virtual
environments, `__pycache__`, model weights, server logs, or partial indexes.
Obsolete scripts should be removed instead of kept as compatibility wrappers.
