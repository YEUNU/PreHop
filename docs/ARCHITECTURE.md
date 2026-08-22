# Architecture

Module-by-module reference for prehop's own pipeline
(`models/prehop/`) — what each file owns, its key functions, and where it
sits in the two pipelines below. For *why* things are shaped this way, see
`docs/CHANGELOG.md`; for *how to run* any of this, see `CLAUDE.md`.

`GraphRAG` (`models/prehop/graphrag.py`) is the facade: it composes
`IndexingPipeline` (offline) and `RetrievalPipeline` (query-time) as mixins
onto one class, and all Neo4j labels/index names are derived from
`(strategy, corpus_tag)` on `__init__` so multiple corpora/strategies coexist
in one database.

## Indexing pipeline (offline, one pass per corpus)

```
 per document                                          after all documents
┌─────────────┐   ┌──────────────────┐   ┌───────────┐   ┌─────────────┐
│  chunking   │──▶│ knowledge_mapping │──▶│graph_writer│──▶│  hop_edges  │
│  (parse +   │   │   (Q-/Q+ per      │   │ (embed +   │   │ (one-shot   │
│ fixed-size  │   │  chunk, LLM)      │   │  Neo4j     │   │  ANN over   │
│  windows)   │   │                   │   │  write)    │   │  whole      │
└─────────────┘   └──────────────────┘   └───────────┘   │  corpus)    │
                                                            └─────────────┘
```

Driven by `cli/index.py::run_indexing` (not part of `models/prehop/`, but
the entry point): reads files, offloads page-splitting to a
`ProcessPoolExecutor` (pure-CPU, keeps the asyncio loop free for LLM/embed
calls), fans out `process_file` per document under a semaphore
(`RAG_MAX_PARALLEL_FILES`), then — once every document has flushed to
Neo4j — makes the single `build_all_hop_edges()` call so every chunk sees
the complete corpus as HOP-edge candidates, not just documents indexed so
far.

### `indexing/chunking.py` — `ChunkingMixin`
Parses `--- Page N ---` markers (or falls back to treating the whole
document as page 1 when none exist — MultiHop-RAG/HotpotQA/MuSiQue corpora
don't carry page markers), sentence-splits each page, and groups sentences
into fixed-size windows of `RAGConfig.CHUNK_SENTENCES` (default 6); a
trailing window shorter than `RAGConfig.MIN_CHUNK_SENTENCES` merges into the
previous chunk rather than standing alone. Also owns the markdown-table
plain-text fallback (`_table_to_text`, gated on `RAG_ABLATION_TABLE`) and the
on-disk chunk cache (`_chunk_cache_load`/`_chunk_cache_save`,
`data/index_cache/v0/<corpus_tag>/`) — keyed on content hash + an ablation
signature that folds in a hash of every prompt template that ends up baked
into a cached chunk, so an edited prompt invalidates old entries
automatically. `extract_knowledge()` is the entry point; per-chunk Q-/Q+ is
generated inside it by calling into `KnowledgeMappingMixin`.

### `indexing/knowledge_mapping.py` — `KnowledgeMappingMixin`
One method, `extract_hoprag_queries(chunk, title)`: a single LLM call
(`HOPRAG_PROMPT`) that returns a chunk's Q⁻ (self-contained questions
answerable from the chunk alone) and Q⁺ (questions the chunk only partially
answers, pointing at its dependencies — the seed material for HOP edges)
plus a `chunk_summary`. No rolling context across chunks. Uses
`llm_json.py`'s `generate_json_or_raise` — a chunk whose LLM output never
becomes valid JSON after `core/vllm_client.py`'s own retries raises, which
`cli/index.py` catches per-file (`stage="task"` in
`data/index_failures/*.json`) rather than silently indexing that chunk with
empty Q-/Q+.

### `indexing/graph_writer.py` — `GraphWriterMixin`
Owns the Neo4j side end to end:
- `setup_index()`: creates the 3 vector indexes (body/Q⁻/Q⁺, on
  `embedding`/`q_minus_embedding`/`q_plus_embedding`) and 3 matching
  fulltext indexes, plus range indexes on `Chunk.id`/`Document.filename`.
- `build_graph()`: embeds body/Q⁻/Q⁺ text in parallel (`_embed_sparse_texts`
  from `QualityGatesMixin`), gates each chunk's Q⁺ candidates through
  `_is_high_quality_q_plus`, and stages one batch-write dict per chunk. The
  primary `embedding` field is always the body embedding — never
  substituted with Q⁻, since `body_vector_index` is built on that property
  and a substitution would silently point body search at Q⁻ content (see
  `docs/CHANGELOG.md` "Fail-loud sweep" for the bug this was).
- `_flush_graph_batch_unlocked()`: batched `MERGE` of `Chunk`/`Document`
  nodes and `NEXT` edges (chunk-sequence order within one document) once
  `RAGConfig.NEO4J_BATCH_SIZE` documents have queued up.
- `retry_query()` / `_is_retryable_neo4j_error()`: shared transient-error
  retry wrapper (deadlock/service-unavailable/session-expired) used by every
  Neo4j write in both `GraphWriterMixin` and `HopEdgeMixin`.

### `indexing/hop_edges.py` — `HopEdgeMixin`
`build_all_hop_edges()`: the one-shot, whole-corpus HOP-edge construction
pass (paper §3.1.4) — streams Q⁺-bearing chunks from Neo4j in pages
(`RAG_HOP_PAGE_SIZE`), and for each one ANN-queries the Q⁺ vector index with
that chunk's own Q⁺ embedding (`_find_hop_candidates`, top-15, same-source
excluded, same-company filtered when `RAGConfig.COMPANY_ANCHORING` is on).
Candidates scoring `>= RAGConfig.HOP_THRESHOLD` become `HOP` edges, capped at
`RAGConfig.HOP_LINK_LIMIT` per source chunk, using the cosine-similarity
score Neo4j's own ANN query already returned — no extra model call, no LLM.
Edges flush to Neo4j after every gather-wave (`RAG_HOP_GATHER_WAVE`) rather
than accumulating in memory, and `MERGE` on `(src)-[:HOP]->(tgt)` makes a
restart after a crash safe to just re-run.

## Query pipeline (per query)

```
run_workflow()                                              (graphrag.py)
     │
     ├─ RAG_GRAPH_HOP_DEPTH > 0 ──▶ graph_search()  ─────────────────┐
     │                                   │                            │
     │                          retrieve() for seeds                 │
     │                                   │                            │
     └─ RAG_GRAPH_HOP_DEPTH == 0 ─▶ retrieve()                       │
                                         │                            │
                          ┌──────────────┴───────────────┐            │
                          │  retrieve.py: 2-stage RRF     │            │
                          │  Stage1: Q-(0.7)+body(0.3)    │            │
                          │       → rerank.py gate        │            │
                          │  Stage2 (if needed):          │            │
                          │    Q+(0.6)+Q- support(0.4)    │            │
                          │       → rerank.py gate         │            │
                          └────────────────────────────────┘            │
                                                                        │
                          traversal.py: NEXT/HOP frontier expansion ◀──┘
                          (continuation-decision LLM call per hop,
                           skipped when force_expand=True)
                                         │
                                         ▼
                          single LLM synthesis call (graphrag.py)
```

### `retrieval/hybrid.py` — `HybridSearchMixin`
`_hybrid_rrf_candidates(query, limit, channel)`: the lowest-level retrieval
primitive. One channel (`"body"`, `"q_minus"`, or `"q_plus"`) at a time —
embeds the query, runs a Neo4j vector-index ANN query and a fulltext query
against that channel's pair of indexes, then fuses the two rank-ordered
lists via reciprocal rank fusion (`TextUtilsMixin._rrf_accumulate`,
`RRF_VECTOR_WEIGHT`/`RRF_TEXT_WEIGHT`, `k=RRF_K_CONSTANT`). Raises if the
query embedding comes back empty rather than silently treating that channel
as having no candidates.

### `retrieval/retrieve.py` — `RetrieveMixin`
`retrieve(query, top_k, user_query)`: the two-stage orchestrator (paper
§3.2.3). Expands `query` into variants via `QueryRewriteMixin` (original +
up to `QUERY_REWRITE_COUNT` rewrites, each weighted `QUERY_REWRITE_WEIGHT`
relative to the original) if `ENABLE_QUERY_REWRITE`. Stage 1 accumulates RRF
scores over Q⁻(0.7)+body(0.3) per variant (or a single channel, under the
`HYPO_CHANNEL_VARIANT` ablation switch — `qminus_only`/`qplus_only`/
`single_combined`), reranks via `RerankMixin`, and returns immediately if
the result is already good enough (`>= top_k` results and best score clear
of `RERANKER_THRESHOLD + 0.08`). Otherwise Stage 2 adds Q⁺(0.6)+Q⁻-support
(0.4) RRF contributions on top of the Stage 1 candidate pool and reranks
again. Both stages share `TextUtilsMixin._rrf_accumulate` for the actual RRF
math.

### `retrieval/rerank.py` — `RerankMixin`
`_rerank_and_select(query, candidates, top_k, query_meta)`: the single
shared reranking path used by both `retrieve.py` (Stage 1 and Stage 2) and
`traversal.py` (seed gating and every hop's frontier gating). Runs the query
through `_simplified_rerank_query` first (strips verbose/role-framing
preludes that otherwise collapse embedding-similarity scores — cached per
GraphRAG instance), scores every candidate via `_embedding_rerank_scores`
(cosine similarity between query and candidate-text embeddings), adds
`META_BOOST_WEIGHT * meta_boost - BOILERPLATE_PENALTY_WEIGHT *
boilerplate_penalty` (both from `TextUtilsMixin._apply_retrieval_calibration`)
to get `final_score`, strict-filters to same-company candidates when the
query is company-anchored, then gates on `rerank_score >= RERANKER_THRESHOLD`
and slices to `top_k`. Returns `(gated_nodes, all_reranked_nodes)` — callers
that don't need the ungated list (traversal.py) just discard the second
element.

### `retrieval/traversal.py` — `TraversalMixin`
`graph_search(entities, depth, top_k, ...)`: paper §3.2.3's graph
traversal. Gets seed nodes from `retrieve()`, gates them through
`_rerank_and_select`, then walks `depth` hops along `[:NEXT|HOP]` edges
(offline mode — the paper-canonical path) or `[:NEXT]` only plus a live Q⁺
ANN supplement (`_runtime_hop_candidates`, `RAGConfig.HOP_MODE=="runtime"`,
a HopRAG-style fallback mode) — re-gating each hop's frontier through the
same `_rerank_and_select`. Between hops, `_need_more_for_next_depth` makes
one LLM call asking whether the accumulated context already answers the
query; if so, traversal stops early. `force_expand=True` (used by the
agentic-OFF path) skips that continuation check entirely so every depth hop
runs deterministically — the only LLM call left in the whole traversal is
the query-simplification step inside reranking, since HOP-edge expansion
itself needs no LLM reasoning (the headline latency claim).

### `retrieval/rewrite.py` — `QueryRewriteMixin`
`_rewrite_query(query)`: one LLM call producing up to
`RAGConfig.QUERY_REWRITE_COUNT` paraphrase variants (deduplicated against
each other and the original query by normalized text). Also holds the
`SEARCH_CONTINUATION_PROMPT` accessor used by `traversal.py`'s continuation
check.

### `retrieval/text_utils.py` — `TextUtilsMixin`
Shared, stateless-ish helpers used throughout retrieval: text/entity
normalization, fulltext-query sanitization, company-key extraction and
matching (`_extract_company_keys`, neutralized entirely when
`RAGConfig.COMPANY_ANCHORING` is off), query metadata extraction
(`_extract_query_metadata` — years/doc-types/company-keys/financial-intent
signals from a raw query string), meta-boost and boilerplate-penalty
scoring (`_meta_boost_for_node`, `_boilerplate_penalty`,
`_company_mismatch_penalty`), `_build_context_from_nodes` (the
`[[title, Page N, Chunk N]]` synthesis-context formatter), `_node_identity`
(dedup key for merging candidates across channels), and `_rrf_accumulate`
(the shared reciprocal-rank-fusion accumulator both `hybrid.py` and
`retrieve.py` call into).

### `retrieval/quality_gates.py` — `QualityGatesMixin`
Two unrelated concerns that happen to share a file: `_is_high_quality_q_plus`
(the entity/period/metric/source-anchor signal-counting gate that decides
which Q⁺ questions survive to become HOP-edge material — used at *indexing*
time from `graph_writer.py`, not query time), and `_embed_sparse_texts`
(batches a list of possibly-empty strings through `self.llm.get_embeddings`,
skipping blanks and reassembling results at their original positions — the
embedding primitive both `graph_writer.py` and, indirectly, `hybrid.py` build
on).

## Shared module

### `models/prehop/llm_json.py` — `generate_json_or_raise`
A plain function (not a mixin — `IndexingPipeline` and `RetrievalPipeline`
are separate mixin compositions only joined together in `GraphRAG`, so a
free function avoids inheritance games). Wraps
`core.vllm_client.VLLMClient.generate_json` — which itself swallows a
persistent JSON-parse failure after exhausting its own retries and returns
`{}` — and raises instead, so every prehop-specific call site
(`knowledge_mapping.py`, `traversal.py`, `rewrite.py`, `rerank.py`) that
depends on `generate_json` actually succeeding gets a real exception instead
of silently continuing on empty/default data.

## Where synthesis happens

Not a separate module — `graphrag.py::run_workflow` builds the final
prompt directly (`_build_answer_prompt`, structurally identical across
prehop and the baselines so any score gap traces to retrieval, not
synthesis-prompt asymmetry) and makes the single LLM call. No agent loop, no
reflection, no refinement — this is a deliberate scope boundary, not a
missing feature (see `docs/CHANGELOG.md`).
