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
- `split_fixed_sentence_windows(...)` sentence-splits each page, emits windows
  of `RAG_CHUNK_SENTENCES` (default 6), and merges a final window shorter than
  `RAG_MIN_CHUNK_SENTENCES` (default 2) into its predecessor.
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
  -> external generation: Q-/Q+/summary per chunk
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
no unsupported-dataset fallback grouping. HopRAG accepts only the three known
corpus tags (`multihoprag`, `2wikimultihopqa`, `musique`) because each needs its
official problem-context grouping; another tag fails explicitly.

The in-repo path limits simultaneously active files with
`RAG_MAX_PARALLEL_FILES`. CPU parsing uses a spawn-based `ProcessPoolExecutor`
(`RAG_PARSE_WORKERS`) so forked HTTP clients cannot corrupt the async inference
clients. Files are read and scheduled in bounded batches. A document failure is
isolated and persisted in `data/index_failures`; any failure still makes the
target fail after cleanup/finalization, so partial success cannot be reported as
a complete index.

### Prehop modules

`indexing/chunking.py`

- Owns the parser, shared splitter, optional content-addressed Q-/Q+/summary
  cache, and run-namespaced debug output.
- `RAG_CHUNK_CACHE=off` disables reuse. A measured cold matrix run forces it
  off and clears prior artifacts.
- `--save-intermediate` writes only to
  `data/debug/<run-id>/<strategy>/<corpus>/<source>/`; normal logs and paper
  artifacts live elsewhere, so parallel debugging cannot overwrite them.

`indexing/knowledge_mapping.py`

- Makes one schema-validated external generation call per chunk using
  `HOPRAG_PROMPT`.
- Returns `summary`, `q_minus`, and `q_plus`; missing keys, invalid JSON,
  non-string list values, an empty summary, or more than three questions raise
  after client retries. Empty Q-/Q+ lists are intentional valid outputs.
- Q+ output has no post-generation keyword, domain, or heuristic quality gate.
  Only empty strings and exact duplicates are removed before storage.

`indexing/embedding.py`

- Batches non-empty strings through the external embedding endpoint.
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
- Every individual source Q+ retrieves up to 15 cross-document candidates from
  target Q-, target body, and target Q+ channels.
- Q+→Q- means the target advertises an answerable formulation; Q+→body means
  direct passage evidence; Q+→Q+ means two documents express the same
  unresolved need. Only Q-/body are direct evidence. Q+→Q+ receives weight
  0.5 and can boost a direct match but can never create an edge alone.
- Reciprocal-rank scores are fused per target chunk across every source Q+.
  At most five targets become outgoing `HOP_ANSWER` edges. Provenance remains
  inspectable as `ANSWERED_BY`, `SUPPORTED_BY`, and `SAME_NEED` relations.
- There are deliberately no Q-↔Q- edges. Documents with the same answer but
  different year/version remain alternative candidates rather than being
  asserted as semantic continuations. Cross-document scope is mandatory.
- Neo4j filters source documents after ANN. Each source therefore uses
  `max(50, own-channel-count + 15)` as its pool: this keeps 15 foreign slots
  without letting one giant document inflate every request in the corpus.
- There is no cosine threshold, same-company filter, runtime-HOP mode,
  cross-encoder, domain rule, or LLM call. If Q+ is disabled, the pass skips.

### Official baseline modules

`models/hoprag/official_indexer.py`

- Stages every corpus file, routes both model types externally, and preserves
  upstream node/question generation.
- Edges are constructed inside official problem contexts: 2WikiMultiHopQA raw
  contexts, MuSiQue paragraph groups, or MultiHop-RAG evidence lists.
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

RAG_GRAPH_HOP_DEPTH > 0 (default 1)
  -> retrieve(query) for seeds
  -> deterministic NEXT/HOP expansion for the configured depth

empty context
  -> fixed "Insufficient evidence" result, no synthesis call

non-empty context
  -> one shared external synthesis call
```

`retrieval/hybrid.py` embeds the original query and runs vector plus Neo4j
full-text search for one channel (`body`, `q_minus`, or `q_plus`). The two
queries run on separate Neo4j sessions concurrently (`asyncio.gather`) rather
than sharing one session's serialized request/response cycle, then fuse into
one ordered list with weighted RRF (`k=60`, vector 1.3, text 1.0).

`retrieval/retrieve.py` has these channel branches:

- `HYPO_CHANNEL_VARIANT=qminus_only`: Q- only, Stage 2 disabled.
- `qplus_only`: Q+ only, Stage 2 disabled.
- `single_combined`: Q-/Q+ at 0.5/0.5, Stage 2 disabled.
- `full` with Q- enabled: Stage 1 Q-/body at 0.7/0.3.
- `full` with Q- disabled: Stage 1 body only.
- In full mode, Stage 2 always runs and adds Q+ and Q- support at 0.6/0.4.
  Disabling Q+ explicitly as an ablation also disables Stage 2.

Each stage's two independent channel calls (e.g. Q-/body, or Q+/Q- support)
run concurrently rather than sequentially. Stage 1 only scores/selects its
own candidates when it is the final result; when Stage 2 runs (the default),
Stage 1's candidates flow into Stage 2 unscored, since Stage 2's own final
scoring pass would otherwise immediately discard a first scoring pass over
the same pool. `core/vllm_client.py` caches single-text query embeddings
per client instance (keyed by exact text, gated to `encoding_type="query"`
single-item calls only, never document batches) so the same query string is
not re-embedded across every channel/scoring call that needs it within one
retrieval — text-to-embedding is a pure function of (text, model), so this
changes no values, only removes redundant network round trips.

`retrieval/scoring.py` uses external query/document embeddings and cosine
similarity to order candidates; there is no dedicated reranker model, rerank
prompt, query rewrite, metadata boost, boilerplate penalty, company filter,
or domain gate. Final top-k selection caps each source document at
`RAG_MAX_CHUNKS_PER_SOURCE_FRACTION` (default 0.34, `floor(top_k *
fraction)`, minimum 1) of the returned slots — pure global score order let
several near-duplicate high-scoring chunks from one document fill most of
the evidence set and crowd out the only chunk carrying a second gold
document, directly undermining multi-hop, cross-document evidence. Same
rule for every source/dataset. Candidates over the cap still backfill by
score if there are not enough distinct sources to fill top_k.

`retrieve.py`'s Stage 1 and `traversal.py`'s incremental collection each size
their candidate pool as `max(floor, top_k * multiplier)` —
`RAG_CANDIDATE_LIMIT_MULTIPLIER` (8), `RAG_SUPPORT_POOL_MULTIPLIER` (4),
`RAG_STAGE1_POOL_MULTIPLIER` (6), and `RAG_WIDE_POOL_MULTIPLIER` (6, shared
by `retrieve.py`'s Stage 2 cap and `traversal.py`'s `candidate_budget`) —
instead of the fixed literals used before this was config-driven. A 60-query
multihoprag fact_recall
sweep found the wide-pool default of 8 both slower and no better than 6
(fact_recall 0.611 vs 0.619, ~25% more traversal latency for no gain), so the
default moved to 6; the other three multipliers were swept too and showed no
improvement beyond run-to-run noise, so they kept their original values.

`retrieval/traversal.py` treats the wide Stage 2 RRF pool (not just the
top-k) as an ordered seed queue and expands one not-yet-expanded seed at a
time, incremental best-first, rather than all seeds at once. `NEXT` is
walked in both directions to recover preceding/following document context;
`HOP_ANSWER` is walked only in the Q+→answer-evidence direction. NEXT and
HOP paths are ranked separately per expansion step, then fused per target
chunk. A seed that was only passively swept up as another (higher-ranked,
same-document) seed's NEXT-neighbor still gets its own expansion turn —
gating that on "already discovered" instead of "already expanded" let one
early seed's same-document walk silently consume the shared candidate
budget before a lower-ranked, cross-document seed ever ran, at real cost to
cross-document evidence coverage on a live multihoprag A/B check. Only a
seed that has itself already been expanded is skipped. Candidates are
pruned to a shared reservoir (`candidate_budget = max(24, top_k*8)`) after
every step, then handed to `scoring.py` for final selection. HOP candidates
are scored independently against their preserved source bridge-Q+ and
target body, then the two cosine scores are averaged. This requires
agreement on both sides of the evidence path; concatenating them let a
strong bridge phrase mask an unrelated target, while discarding Q+ erased
why the edge existed. There is no query-time generation, continuation
prompt, heuristic stop gate, or runtime ANN supplement in Prehop retrieval.

`retrieval/text_utils.py` contains only normalization, Lucene sanitization,
context formatting, node identity/dedup, and RRF helpers.

## Prompt inventory

Project-owned prompt templates are deliberately limited to:

- `utils/prompts/indexing.py`: Prehop indexing-time Q-/Q+/summary generation.
- `utils/prompts/shared.py`: one dataset-neutral final-answer prompt shared by
  Prehop, Naive, and the HopRAG adapter.
- `utils/prompts/evaluation.py`: the offline benchmark judge.

The benchmark uses the OpenAI Batch API for LLM-as-a-judge by default
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
identical, paper tables and captions must state them. If a separate controlled
retrieval-only study is desired, add it as a clearly named ablation instead of
silently changing an official baseline.

## Measured full matrix

`scripts/run_index_matrix.py --clear-graph --max-parallel 3
--max-generation-parallel 3` runs all three datasets by four strategies in the
order `ms_graphrag → hoprag → naive → prehop`. It records per-target wall time, CPU, peak RSS,
structural integrity counts, payload estimates, endpoint/host pressure, and
failure logs beneath `artifacts/indexing/<run-id>`. It begins with bounded
parallelism and reduces width when sustained host-memory or inference-queue
pressure is observed. `VLLM_MAX_NUM_SEQS=120` is enforced and recorded as server capacity;
generation concurrency is one global per-target/event-loop semaphore, not one
limit per document. Embeddings default to two concurrent batches of 32 for one
target. `RAG_INFERENCE_CAPACITY_MODE` records whether the model names share an
accelerator scheduler. This matters when one gateway URL routes generation and
embedding to different servers: `separate` preserves their independent 120
sequence budgets, `shared` combines worst-case pressure, and `auto` infers from
URL equality. Generation pressure is multiplied by
`--max-generation-parallel`, not by embedding-only matrix slots.
Generation and embedding queues are sampled separately, and sustained pressure
still lowers target width for later work.
The scheduler additionally caps generation-heavy targets at one in flight
(`--max-generation-parallel 1`). It scans ahead for an embedding-only Naive
target to occupy the remaining width. This policy was selected after a cold
run showed Prehop throughput fall from roughly 7 to 2–3 documents/minute when
official HopRAG began generating concurrently on the generation server.
Naive uses document batches of 32: it parses every source, flattens all chunks
into one external embedding request stream, validates every vector, and
atomically replaces the batch's source nodes in one Neo4j transaction. This
turns short-document datasets into real embedding batches instead of thousands
of one-item requests. Live progress markers, current phase, and ETA are written
to `progress.json`.
The generation-heavy baselines then use the safe capacity left by serialization:
HopRAG runs 10 document workers with 4 chunk threads (at most 40 generation
calls), while MS GraphRAG runs 32 concurrent requests. The matrix previously
overrode MS's own 48-request default down to 8 and HopRAG down to 4, which
needlessly under-used the configured sequence budget once cross-method overlap was gone.

The measurement set directly addresses the indexing-time tradeoff: overall
and Prehop phase latency, logical storage, document/chunk/question/edge counts,
Q-/Q+ and Q+-direction coverage, provenance completeness, exact NEXT topology,
cross-document HOP invariants, and observed endpoint pressure. Retrieval and
answer-quality attribution remains a separate benchmark/ablation concern; an
index with valid topology is not reported as evidence that HOP improves QA.
For the semantic middle layer, the runner resolves every full-query evidence
title against indexed documents, then reports the fraction of fully resolved
gold queries and gold document pairs connected by at least one `HOP_ANSWER`.
