# Architecture

This is the module-level source of truth for the current indexing and query
paths. All model inference is sent to configured external OpenAI-compatible
generation and embedding endpoints. The repository never starts a local model.

## Shared input contract

`models/prehop/indexing/chunking.py` owns the only in-repo parser and fixed
window splitter:

- `parse_pages_offline(filename, content)` reads an optional first-line
  `Title: ...` header and `--- Page N ---` markers.
- If page markers are absent, the remaining body is one logical page. This is
  a supported corpus format, not an LLM fallback.
- `split_fixed_sentence_windows(...)` sentence-splits each page, emits fixed
  six-sentence windows, and retains the final partial window.
- Page boundaries are never crossed. Pipe-delimited text is preserved exactly;
  there is no table-to-text branch.
- Prehop and Naive both call these functions, so their chunk synthesis
  condition is identical. Official HopRAG and MS GraphRAG retain their own
  upstream chunkers because changing those would no longer be an official
  baseline comparison.

## Strategy dispatch and every indexing branch

`cli/index.py::run_indexing` obtains a per-`(strategy, corpus_tag)` file lock,
then `run_indexing_unlocked` dispatches as follows:

```text
strategy == prehop
  shared parser in spawn ProcessPool
  -> shared fixed page windows
  -> external generation: Q-/Q+ per chunk
  -> external body/Q-/Q+ document embeddings + Q+ query embeddings
  -> atomic Neo4j Document/Chunk/question replacement + NEXT writes
  -> after every document succeeds: whole-corpus HOP edge pass

strategy == naive
  shared parser + shared fixed page windows
  -> external body embeddings
  -> Neo4j Chunk writes

strategy == hoprag
  -> official HopRAG stage 1 node/question generation
  -> official per-problem edge groups
  -> Neo4j node/edge/index writes

strategy == ms_graphrag
  -> official GraphRAG Standard pipeline
  -> text units/entities/relationships/communities/reports/embeddings
  -> corpus-scoped parquet + LanceDB output

anything else
  -> ValueError
```

All `.txt` and `.md` files are selected. There is no company/sample filter and
no unsupported-dataset fallback grouping. HopRAG accepts only the active known
corpus tags (`multihoprag`, `musique`) because each needs its
official problem-context grouping; another tag fails explicitly.

The in-repo path limits simultaneously active files with
`RAG_MAX_PARALLEL_FILES`. CPU parsing uses a spawn-based `ProcessPoolExecutor`
(`RAG_PARSE_WORKERS`) so forked HTTP clients cannot corrupt the async inference
clients. Files are read and scheduled in bounded batches. A document failure is
isolated and persisted in `data/index_failures`; any failure still makes the
target fail after cleanup/finalization, so partial success cannot be reported as
a complete index. Graph indexing uses a bounded rolling task window: when any
document finishes, its slot is reused immediately. A slow document therefore
does not impose a barrier on the rest of its original scheduling batch, while
the number of resident tasks remains bounded by `RAG_FILE_SCHEDULE_BATCH`.

### Prehop modules

`indexing/chunking.py`

- Owns the parser, shared splitter, optional content-addressed Q-/Q+
  cache, and run-namespaced debug output.
- `RAG_CHUNK_CACHE=off` disables reuse. A measured cold run sets it explicitly
  and clears prior artifacts.
- `--save-intermediate` writes only to
  `data/debug/<run-id>/<strategy>/<corpus>/<source>/`; normal logs and paper
  artifacts live elsewhere, so parallel debugging cannot overwrite them.

`indexing/knowledge_mapping.py`

- Makes one schema-validated external generation call per chunk using
  `HOPRAG_PROMPT`.
- Uses greedy decoding (`temperature=0`) so indexing does not inherit a
  deployment-specific sampling default.
- Returns `q_minus` and `q_plus`; missing keys, invalid JSON,
  non-string list values, or more than three questions raise
  after client retries. Empty Q-/Q+ lists are intentional valid outputs.
- Empty strings, within-channel duplicates, source-relative wording, and exact
  Q−/Q+ duplicates are removed deterministically before storage. The filter
  does not score semantic quality or introduce a dataset-tuned threshold.

`indexing/embedding.py`

- Reuses cached vectors only when model, revision, endpoint, dimensions,
  encoding role, instruction, and normalized text all match.
- Batches cache misses through the external embedding endpoint.
- Cold timing runs disable both generation and embedding caches.
- Restores sparse results to original positions and verifies response count,
  non-empty vectors, and consistent dimensions.

`indexing/graph_writer.py`

- Creates corpus-tagged body/Q-/Q+ vector and full-text indexes plus id indexes.
- Stores each generated question as an individual `QMinus` or `QPlus` node;
  multiple directions from one chunk are never concatenated into one vector.
- Body, Q-, and document-side Q+ embeddings include `Document title: ...` so
  manuals or reports from different years retain version scope. Q+ also stores
  a separately instructed query embedding used only as the outgoing search.
- Re-indexing one document atomically deletes its old contained chunks and
  question nodes and writes its replacement subgraph. A second write creates
  forward-only ordered `NEXT` edges; stale NEXT/HOP edges disappear with the
  deleted old chunks.
- There is no company property and no conditional index-recreation path. A cold
  run clears the graph/schema once before indexing.
- Neo4j retries are restricted to transient/session/service errors. Failed
  batches are restored for safe replay; non-transient errors fail immediately.

`indexing/hop_edges.py`

- Runs once after the complete Prehop corpus is flushed and indexes are online.
- Every individual source Q+ retrieves the best cross-document Q-. The matched
  Q-'s owner chunk is the HOP target; no second body ANN search is needed.
- Multiple Q+ questions from one source that resolve to the same target are merged
  into one `HOP_ANSWER` edge while retaining every question.
  `ANSWERED_BY` preserves the Q+→Q- link and Q- ownership preserves the
  evidence target. In the no-Q- indexing ablation, Q+ retrieves body directly
  and `SUPPORTED_BY` records that alternative path.
- There are deliberately no Q-↔Q- edges. Documents with the same answer but
  different year/version remain alternative candidates rather than being
  asserted as semantic continuations. Cross-document scope is mandatory.
- Neo4j filters source documents after ANN. Each source therefore requests its
  own channel count plus one foreign slot, without a fixed ANN floor.
- There is no cosine threshold, same-company filter, runtime-HOP mode,
  cross-encoder, domain rule, or semantic verification call. If Q+ is disabled,
  the pass skips.

### Official baseline modules

`models/hoprag/official_indexer.py`

- Stages every corpus file, routes both model types externally, and preserves
  upstream node/question generation.
- Edges are constructed inside official problem contexts: MuSiQue paragraph
  groups or MultiHop-RAG evidence lists.
- Per-document caches and stage markers support safe resume for ordinary runs;
  the measured cold runner removes them first.
- Stage 2 inserts each document once and then streams each problem group to
  bound memory. No company metadata is stored.

`models/ms_graphrag/official_indexer.py`

- Stages every corpus file and calls the official `Standard` build pipeline.
- LiteLLM routes generation and embedding to external endpoints. Output is
  isolated under `data/ms_graphrag_output/<corpus_tag>`.
- Expected output tables are verified; workflow errors fail the target.

## Prehop query path and every branch

`models/prehop/graphrag.py::run_workflow` strips only the benchmark output
format suffix and then:

```text
RAG_GRAPH_HOP_DEPTH == 0
  -> retrieve(query, top_k=12)

RAG_GRAPH_HOP_DEPTH == 1 (default)
  -> retrieve(query) for seeds
  -> deterministic NEXT/HOP expansion for the configured depth
  -> RAG_GRAPH_EDGE_VARIANT selects full, hop_only, or next_only for query-only ablation

empty context
  -> fixed "Insufficient evidence" result, no synthesis call

non-empty context
  -> one shared external synthesis call
```

`retrieval/hybrid.py` embeds the original query and runs vector plus Neo4j
full-text search for one channel (`body`, `q_minus`, or `q_plus`). Vector and
full-text branches share one Cypher request per representation; enabled
representations run concurrently. Their results fuse into one ordered list
using equal reciprocal ranks `1 / (rank + 1)`. There is no
modality weight or query-time fusion constant.

`retrieval/retrieve.py` searches each enabled representation exactly once with
the same query embedding. Q- and body hits have the direct-evidence role; Q+
hits have the dependency-seed role and can expose outgoing offline HOP edges.
Enabled representation results form a set union, so direction is expressed
by graph role rather than a cross-representation score.

- `HYPO_CHANNEL_VARIANT=qminus_only`: Q- direct evidence only.
- `qplus_only`: Q+ dependency seeds only.
- `single_combined`: Q-/Q+ once each, set union, no body.
- `full`: Q-/body direct evidence plus Q+ dependency seeds.

The searches run concurrently. There is no second Q- support search: a Q-
hit already identifies its owner evidence chunk, while a Q+ hit reaches target
Q-/body evidence through the pre-built `HOP_ANSWER` relation. The query
embedding is created once before parallel channel search and passed unchanged
to every vector channel and final scoring call.

`retrieval/scoring.py` reuses the body and source-Q+ document embeddings stored
during indexing and embeds only the user query. Cosine similarity orders
candidates; there is no dedicated reranker model, rerank
prompt, query rewrite, metadata boost, boilerplate penalty, company filter,
or domain gate. Final top-k selection uses global cosine order by default.
`RAG_SOURCE_SELECTION_VARIANT=round_robin` is an explicit ablation that takes
one ranked chunk per source per round; it is not part of the primary method.

Each representation retains at most `top_k` owner chunks, so the fused base
pool is bounded by `top_k × active_representation_count` without a candidate
multiplier. Vector and full-text search do not have separate tunable width
knobs. Body nodes use the owner budget as-is. Q−/Q+
indexes contain at most three questions per owner chunk, so their raw
question-node searches use exactly three times the owner budget before
deduplication. This factor is an indexing-schema bound, not a tuned retrieval
parameter. The query embedding is created once before parallel channel search
and passed unchanged to every vector channel.

`retrieval/traversal.py` treats the complete representation-union pool (not just the
final top-k) as one frontier and expands it in one Neo4j request per depth. `NEXT` is
walked in both directions to recover preceding/following document context;
`HOP_ANSWER` is exposed only by owner chunks actually matched through the Q+
dependency channel and is walked only in the Q+→answer-evidence direction.
Q−/body-only seeds and graph-discovered nodes expose NEXT only, preventing an
unrelated Q+ attached to a direct-evidence chunk from triggering a HOP. NEXT and
HOP paths are ranked separately per expansion step, then fused per target
chunk. The structurally bounded results are retained without a candidate
reservoir or graph-search floor. HOP candidates compare the query against
each indexed source Q+ separately, take the best bridge similarity, and use
`min(body, bridge)` as the final score. This requires agreement on both sides
without a mixing weight. There is no query-time generation, continuation
prompt, heuristic stop gate, or runtime ANN supplement in Prehop retrieval.

`retrieval/text_utils.py` contains only normalization, Lucene sanitization,
context formatting, node identity/dedup, and RRF helpers.

## Prompt inventory

Project-owned prompt templates are deliberately limited to:

- `utils/prompts/indexing.py`: indexing-time Q-/Q+ generation.
- `utils/prompts/shared.py`: one dataset-neutral final-answer prompt shared by
  Prehop, Naive, and the HopRAG adapter.
- `utils/prompts/evaluation.py`: the offline benchmark judge.

LLM-as-a-judge is disabled by default (`RAG_JUDGE_ENABLED=false`). When it is
explicitly enabled, the benchmark uses the OpenAI Batch API by default
(`RAG_JUDGE_BATCH=true`). Each strategy/seed writes an atomic pending manifest
immediately after submission; `main.py` resolves all submitted batches in
parallel and only publishes aggregates when every expected custom ID is
present. Submission, terminal-state, parsing, or partial-output failures are
reported as incomplete runs and never fall back silently to more expensive
synchronous judge calls. `RAG_JUDGE_BATCH=false` is reserved for explicit
debug runs.

Prehop retrieval contains no prompt. Official HopRAG is the documented
baseline exception: its upstream `bfs_node` traversal includes its published
LLM helpful/helpless node judgement. The adapter routes that call externally
and does not add a local reranker or new prompt. MS GraphRAG also retains the
official package's extraction/community/report/search prompts. Those upstream
prompts are baseline algorithms, not hidden Prehop gates.

## Comparison settings

- Prehop and Naive use the same fixed chunks and shared final synthesis prompt.
- Prehop and Naive use top-k 12.
- HopRAG keeps the official repository's end-to-end top-k 20.
- MS GraphRAG keeps its official context-budget search configuration.

Because these retrieval budgets are intentionally method-official rather than
identical, paper tables and captions must state them. A separate controlled
retrieval-only study is a clearly named ablation and never silently changes an
official baseline.

## Evaluation output contract

The benchmark emits deterministic normalized answer EM/F1 as a downstream
answer signal; benchmark-annotated gold-evidence retrieval is primary for the
method claim. MuSiQue uses answer aliases. The LLM-as-a-judge `score` is an
optional exploratory semantic-correctness field for aliases and equivalent wording;
`groundedness` and `hallucination` are separate context-directed diagnostics.
None replaces deterministic answer scoring, and without qualified-human
validation none enters quantitative submission results or system rankings.

Evidence metrics follow the prepared gold unit for each dataset:

- MultiHop-RAG reports fact recall at `k` on non-null queries; null queries
  report refusal and attempted-answer hallucination separately and do not
  enter retrieval denominators.
- MuSiQue reports supporting-paragraph/title precision, recall, and F1. Its
  paragraph-level gold evidence is not compared with the six-sentence fact
  matcher.

Missing gold units are emitted as `-1`, while an evaluated query with no match
is zero. Paper aggregates exclude failed, incomplete, and unreconciled rows.
The exact metric definitions, official evaluator references, paper-eligibility
rules, and reporting decisions are maintained in the local, intentionally
untracked `docs/prehop_paper.md`. This architecture document summarizes the
implemented evaluation contract but does not replace that paper specification.

## Measured independent runs

Every paper run invokes one dataset/strategy target with a unique `RAG_RUN_ID`.
The run records wall time, phase timings exposed by the adapter, structural
integrity, and failures without an orchestration-level retry or resume policy.
`VLLM_MAX_NUM_SEQS` records endpoint capacity. Clients that share a generation
endpoint and event loop share one request semaphore, and Prehop processes at
most one generation request per active file. Embeddings use bounded batches. Naive flattens 32
source documents into each embedding/write batch. Official adapters report
only timing boundaries their upstream implementations expose.

The measurement set directly addresses the indexing-time tradeoff: overall
and Prehop phase latency, logical storage, document/chunk/question/edge counts,
Q-/Q+ and Q+-direction coverage, provenance completeness, exact NEXT topology,
cross-document HOP invariants, and observed endpoint pressure. Retrieval and
answer-quality attribution remains a separate benchmark/ablation concern; an
index with valid topology is not reported as evidence that HOP improves QA.
Before publishing the corpus snapshot as complete, `cli/index.py` reads the
live graph and enforces the index-quality contract: embeddings and ownership
are complete; question representations are non-empty, source-independent,
deduplicated, and role-distinct; NEXT is exactly consecutive within each
document; every HOP is cross-document, channel-consistent, provenance-complete,
within the schema out-degree bound, and has the expected Q+→Q-→owner
provenance (or the explicit body-only ablation path); and all search indexes
are online. Coverage, linkage rate, and graph density remain descriptive and
do not become dataset-tuned pass thresholds. Held-out retrieval metrics test
effectiveness separately.
For official MS GraphRAG and HopRAG adapters, the stored timing includes the
explicit aggregate `official_pipeline_seconds` plus adapter-observed workflow
or stage timings; the runner does not fabricate boundaries that the upstream
package does not expose. Prehop retains its finer adapter phase timings;
Naive reports its aggregate pipeline and measurement timing only.
MS GraphRAG relationship drops caused by missing extracted entities are
recorded as integrity warnings in the target result rather than silently
treated as a clean graph.
For the semantic middle layer, the runner resolves every full-query evidence
title against indexed documents, then reports the fraction of fully resolved
gold queries and gold document pairs connected by at least one `HOP_ANSWER`.
