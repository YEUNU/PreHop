# Prehop repository guide

The current system evaluates Prehop, Naive RAG, official HopRAG, and official
MS GraphRAG on MultiHop-RAG, 2WikiMultiHopQA, and MuSiQue. The detailed module and
branch map is in `docs/ARCHITECTURE.md`.

## Supported state

- Python 3.12, dependencies managed by `uv` from `pyproject.toml`.
- Neo4j is the only local service. Generation and embeddings are mandatory
  external OpenAI-compatible endpoints configured by `VLLM_URL` and
  `VLLM_EMBED_URL`; there is no localhost/model-launch fallback.
- The embedding dimension in `NEO4J_VECTOR_DIMENSIONS` must equal the endpoint's
  actual vector length. Startup probes validate the configured model ids and
  dimensions.
- The paper matrix server capacity is `VLLM_MAX_NUM_SEQS=120`. The matrix runner
  clamps `MAX_CONCURRENT_LLM_CALLS`, `RAG_MS_CONCURRENT_REQUESTS`, and
  `RAG_HOP_DOC_WORKERS × RAG_HOP_MAX_THREADS` under one per-target budget, then
  multiplies that budget by the three concurrent datasets in the active
  strategy phase. Embeddings default to batches of 32 and are budgeted
  separately when generation and embedding use separate accelerator servers.
- A dedicated reranker is not used. The external embedding endpoint supplies
  vectors for threshold-free cosine top-k ordering. Final selection caps each
  source document at `RAG_MAX_CHUNKS_PER_SOURCE_FRACTION` (default 0.34) of
  top_k so one strongly-matching document cannot fill the whole evidence set.
- Query-time candidate-pool widths are config-driven:
  `RAG_CANDIDATE_LIMIT_MULTIPLIER` (8), `RAG_SUPPORT_POOL_MULTIPLIER` (4),
  `RAG_STAGE1_POOL_MULTIPLIER` (6), `RAG_WIDE_POOL_MULTIPLIER` (6). A single
  query-time query string is embedded once per client instance and reused
  across every channel/scoring call that needs it in the same retrieval.
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
- Benchmark answer quality now emits deterministic normalized EM/F1 (including
  MuSiQue aliases) as the primary signal and keeps LLM-judge semantic
  correctness and context groundedness as separate diagnostic fields. Evidence
  metrics are dataset-aware: null
  MultiHop-RAG queries are excluded from retrieval averages, sentence/fact
  ranking metrics are skipped for paragraph-level MuSiQue gold evidence, and
  title-level evidence precision/recall/F1 is reported separately.

## Data and tags

| Tag | Corpus directory | Full query file |
|---|---|---|
| `multihoprag` | `data/multihoprag_corpus` | `data/multihoprag_queries.json` |
| `2wikimultihopqa` | `data/2wikimultihopqa_corpus` | `data/2wikimultihopqa_queries.json` |
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

# Full cold measured matrix (aggregate sequence bound 120)
VLLM_MAX_NUM_SEQS=120 VLLM_GENERATION_MAX_NUM_SEQS=120 VLLM_EMBED_MAX_NUM_SEQS=120 \
.venv/bin/python scripts/run_index_matrix.py \
  --datasets multihoprag 2wikimultihopqa musique \
  --strategies ms_graphrag hoprag naive prehop \
  --clear-graph --max-parallel 3 --max-generation-parallel 3

# Combine stopped/resumed matrix fragments into cumulative phase timings
.venv/bin/python scripts/merge_index_matrix_runs.py \
  artifacts/indexing/<run-a> artifacts/indexing/<run-b> \
  --out-dir artifacts/indexing/merged-paper-run

# Continue an interrupted matrix in the same run folder
.venv/bin/python scripts/run_index_matrix.py \
  --run-id <run-id> --resume --target-attempts 3

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

`scripts/run_index_matrix.py` runs 12 targets: MultiHop-RAG, 2WikiMultiHopQA,
and MuSiQue by four strategies in the order
`ms_graphrag → hoprag → naive → prehop`. It starts with at most three targets concurrently, samples host and
inference pressure, and reduces the remaining width when pressure is sustained.
It does not increase width again within the same run. Each child is a separate
process group, so interruption terminates descendants and prevents overlapping
debug/index jobs.
The scheduler uses a strict strategy barrier: all selected datasets for
`ms_graphrag` finish before `hoprag` starts, then `naive`, then `prehop`.
Within a phase, selected datasets run concurrently when the parallel limits
are at least the dataset count.
Naive batches 32 source documents per embedding/write transaction. Prehop's
outer in-flight file cap defaults to 16; its generation semaphore remains 30,
so short one-chunk corpora can use the endpoint without making long-document
fan-out unbounded.
Official HopRAG and MS GraphRAG receive adapter-specific limits derived from
the same phase budget. With three generation targets and a 120 sequence
server, each target is capped at 40 generation calls; HopRAG's worker/thread
product and MS GraphRAG's request semaphore are both clamped to that value.
The runner writes pending-phase ETA components to `progress.json` and emits a
watch line hourly by default (`--watch-interval` changes this).

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
the paper. A controlled equal-budget retrieval experiment is reported
separately from official-baseline results.

## Paper specification rules

`docs/prehop_paper.md` is a local, gitignored finalized evaluation
specification. It records the current method, dataset-specific metrics, and
reporting decisions; it is not a source of benchmark results. When editing it:

- Separate confirmed implementation facts, measured results, and hypotheses.
  Never invent a result, citation, author detail, dataset count, or statistical
  conclusion. A value without a verified artifact is omitted from the paper.
- Preserve established AI-research terminology when it improves precision,
  including RAG, GraphRAG, LLM, approximate-nearest-neighbor search, RRF, and
  LLM-as-a-judge. Define an acronym or specialized term at first use, then
  avoid unnecessary restatement; simplify only implementation-specific names
  that do not help a reader reproduce or understand the method.
- Record the exact run ID, git revision, environment/model settings, dataset
  split, seed, judge status, and artifact path for every reported number. A
  result is paper-eligible only when the target completed without integrity or
  measurement failures and the judge output is complete.
- Keep official-baseline behavior and Prehop ablations clearly distinct. Do
  not silently equalize top-k/context budgets or alter upstream HopRAG/MS
  GraphRAG behavior; label any controlled comparison separately.
- Report indexing and query-time costs separately. State that Prehop uses
  offline Q-/Q+/HOP construction, has no query-time retrieval LLM call, and
  makes one final synthesis call only when context is non-empty.
- For every table or figure, retain its metric definition, denominator,
  aggregation rule, uncertainty estimate, and source artifact. Do not publish
  self-judged or partial Batch-judge numbers as final results.
- Keep the local specification ignored. A submission copy is a deliberate
  tracked export only after its venue/version is fixed and secrets, local
  endpoints, generated logs, and private submission notes are removed.

## Generated files and repository hygiene

Generated logs, caches, debug output, graphs, results, index outputs, and matrix
artifacts are not source files and must remain ignored. Do not commit virtual
environments, `__pycache__`, model weights, server logs, or partial indexes.
Obsolete scripts should be removed instead of kept as compatibility wrappers.
