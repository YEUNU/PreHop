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
- The server capacity is `VLLM_MAX_NUM_SEQS=128`. The client deliberately uses
  lower concurrency (`MAX_CONCURRENT_LLM_CALLS`, currently 30) so concurrent
  strategies have queue headroom.
- A dedicated reranker is not used. The external embedding endpoint supplies
  vectors for threshold-free cosine top-k ordering.
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

# Rebuild only Prehop HOP edges after changing the HOP threshold
.venv/bin/python main.py --mode hop_rebuild --strategy prehop --corpus-tag multihoprag

# Full cold measured matrix
.venv/bin/python scripts/run_index_matrix.py --clear-graph --max-parallel 2

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
- after the final pass, inspect HOP edge score ranges, per-source out-degree,
  cross-source property, and representative source/target Q+ text;
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
