# Prehop repository guide

The current system evaluates Prehop, Naive RAG, official HopRAG, and official
MS GraphRAG on MultiHop-RAG and MuSiQue. The detailed module and
branch map is in `docs/ARCHITECTURE.md`.

## Supported state

- Python 3.12, dependencies managed by `uv` from `pyproject.toml`.
- Neo4j is the only local service. Generation and embeddings are mandatory
  external OpenAI-compatible endpoints configured by `VLLM_URL` and
  `VLLM_EMBED_URL`; there is no localhost/model-launch fallback.
- The embedding dimension in `NEO4J_VECTOR_DIMENSIONS` must equal the endpoint's
  actual vector length. Startup probes validate the configured model ids and
  dimensions.
- `EMBEDDING_MAX_NUM_SEQS=512` records the embedding endpoint's batch capacity.
  The client uses one batch of up to 512 texts, independently from the
  generation endpoint's `VLLM_MAX_NUM_SEQS` limit.
- Paper runs execute one dataset/strategy target at a time with an explicit
  run ID. `VLLM_MAX_NUM_SEQS` records endpoint capacity; each adapter's worker
  controls must remain within that capacity. Embeddings default to batches of
  512 and use the separately configured embedding endpoint.
- A dedicated reranker is not used. Query-time scoring reuses body and Q+
  document embeddings stored during indexing; only the user query is embedded.
  Final selection uses global cosine order. Source round-robin is an explicit
  query-only ablation rather than part of the primary method.
- Query-time search is role-based and has no candidate-width multipliers. Q−
  and body provide direct-evidence candidates; Q+ provides dependency seeds
  for outgoing offline HOP traversal. Each enabled representation is searched
  once and retains at most `top_k` owner chunks, so the fused base-pool bound
  is `top_k × active_representation_count`. A single query embedding is reused
  throughout. Q−/Q+ raw search uses the indexing schema's maximum three
  questions per owner rather than a tuned modality limit.
- Vector/full-text fusion uses equal reciprocal ranks `1 / (rank + 1)`.
  Representation results form a set union without cross-representation scores.
- Graph expansion batches the complete frontier once per depth, retains the
  structurally bounded result set without a reservoir multiplier, and scores
  a HOP target as `min(body similarity, best individual source-Q+ similarity)`.
- Prehop has no query rewrite, rerank simplification prompt, continuation
  prompt, runtime HOP, company/domain gate, metadata boost, boilerplate
  penalty, table-to-text generation, or Q+ heuristic post-filter.
- Prehop retrieval has no generation call. Its only query-time generation is
  final answer synthesis after deterministic retrieval/traversal.
- Official baselines keep their upstream behavior. In particular, official
  HopRAG `bfs_node` uses its published LLM node judgement during retrieval;
  this is documented baseline behavior and is routed to the external endpoint.
- Benchmark LLM judging is disabled by default. When explicitly enabled for
  exploratory error analysis, it uses OpenAI Batch by default and never
  silently falls back to synchronous paid calls. Without qualified-human
  validation, judge output is excluded from paper rankings and quantitative
  result tables.
- Benchmark answer quality emits deterministic normalized EM/F1 (including
  MuSiQue aliases) as a downstream signal and keeps LLM-judge semantic
  correctness and context groundedness as separate diagnostic fields.
  MultiHop-RAG emits official-compatible `official_hits@k`, `official_mrr@10`,
  and `official_map@10` separately from custom `evidence_fact_recall@k`.
  Gold-evidence retrieval is primary for the method claim. MuSiQue support
  uses the official formula over stable global paragraph
  identities and is named `paragraph_support_*`, not official query-local
  `idx` support; title-level evidence precision/recall/F1 is diagnostic only.
  Paragraph-ID headers are stripped before indexing. Negative sentinels and
  runtime-error rows are excluded from aggregates, whose eligible counts are
  recorded. Sample manifests are exploratory rather than full benchmark runs.

## Data and tags

| Tag | Corpus directory | Full query file |
|---|---|---|
| `multihoprag` | `data/multihoprag_corpus` | `data/multihoprag_queries.json` |
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

# Rebuild only Prehop HOP/provenance edges after changing HOP construction
.venv/bin/python main.py --mode hop_rebuild --strategy prehop --corpus-tag multihoprag

# Run each dataset and strategy independently. Use a new run id for each cold run.
RAG_RUN_ID=<run-id> ./run_index.sh \
  --model prehop --dataset data/multihoprag_corpus --corpus-tag multihoprag

# Resume an already submitted OpenAI Batch judge after interruption
.venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<run-id>

# Optional supplemental LLM-judge analysis (off by default)
RAG_JUDGE_ENABLED=true EVAL_MODEL=<independent-judge-model> \
  .venv/bin/python main.py benchmark ...

# Non-paper debugging only; recorded as judge_independent=false
RAG_JUDGE_ALLOW_SELF=true RAG_JUDGE_ENABLED=true EVAL_MODEL=<generation-model> \
  .venv/bin/python main.py benchmark ...

# Verification
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

`run_servers.sh` never starts a model. If either inference endpoint or served
model id is missing/unreachable, indexing and benchmarking fail before work.

## Independent indexing runs

`run_index.sh`, `run_multihoprag.sh`, and `run_dataset.sh` accept exactly one
strategy per invocation. Dataset/strategy targets are never scheduled as an
implicit matrix. This makes the run ID, log, resource use, failure state, and
graph mutation attributable to one target.

Naive batches 32 source documents per embedding/write transaction. The paper
environment allows 64 active Prehop files; each file generates one chunk at a
time, and clients on the same endpoint/event loop share the 120-request
generation limit. Official HopRAG and MS GraphRAG retain their
adapter-specific worker limits.

A measured cold run must:

1. stop any prior indexing process;
2. clear Neo4j nodes, constraints, and non-lookup indexes;
3. remove prior `data/index_cache`, HopRAG outputs/caches, MS GraphRAG outputs,
   debug, failure, and stats artifacts for the selected targets;
4. set `RAG_CHUNK_CACHE=off` and disable baseline cache reuse;
5. use a new `RAG_RUN_ID`;
6. run endpoint/model/dimension preflight before launching the target.

Each target writes an isolated stdout/stderr log. A target with any document,
workflow, graph-finalization, or integrity failure is failed, never silently
classified as complete.

For Prehop, completion also requires the live index-quality gate in
`cli/index.py`. The gate checks representation ownership and embeddings,
question sanitation and role separation, exact NEXT topology, HOP direction,
Q+→Q−→owner provenance, bounded out-degree, and online
indexes. Coverage and density are diagnostics; benchmark retrieval metrics are
the evidence of effectiveness.

## Prehop inspection during indexing

For Prehop targets, inspect real intermediate files and the live graph rather
than relying only on progress logs:

- sample `data/debug/<run-id>/prehop/<corpus>/<source>/final_chunks.json` when
  `--save-intermediate` is enabled;
- verify raw chunk text, title/page/sent_id ordering, grounded Q-, outward Q+,
  generated questions and absence of fabricated or converted text;
- query total Documents/Chunks and Q-/Q+ coverage;
- after the final pass, inspect HOP question/owner provenance, per-source
  out-degree, cross-source property, and representative Q+→Q−→owner
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
  measurement failures. Optional judge output must be complete only when it is
  reported as a separately labelled supplemental analysis.
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

Generated logs, caches, debug output, graphs, results, and index outputs
artifacts are not source files and must remain ignored. Do not commit virtual
environments, `__pycache__`, model weights, server logs, or partial indexes.
Obsolete scripts should be removed instead of kept as compatibility wrappers.
