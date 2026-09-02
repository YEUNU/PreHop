# Architecture

This document defines the current indexing and query paths at module level.
All inference uses configured external OpenAI-compatible generation and
embedding endpoints; the repository does not start a local model. Historical
changes belong in `CHANGELOG.md`, and research claims belong in the local
`prehop_paper.md`.

## Shared input contract

`models/prehop/indexing/chunking.py` owns the in-repo parser and fixed
window splitter:

- `parse_pages_offline(filename, content)` reads an optional first-line
  `Title: ...` header and `--- Page N ---` markers.
- If page markers are absent, the remaining body is one logical page. This is
  a supported corpus format, not an LLM fallback.
- `split_fixed_sentence_windows(...)` sentence-splits each page, emits fixed
  six-sentence windows, and retains the final partial window.
- Page boundaries are never crossed. Pipe-delimited text is preserved exactly;
  there is no table-to-text branch.
- Prehop and Naive apply the same fixed window splitter after parsing. This
  makes the in-repo Naive path a controlled retrieval baseline rather than a
  claim that Naive RAG has one canonical chunker.
- Official HopRAG, MS GraphRAG, BrowseNet, and PropRAG retain their upstream
  indexing units because changing them would no longer be a full-system
  comparison.

## Strategy dispatch and indexing branches

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
  -> materialize reciprocal source-Q+ provenance on each HOP edge

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

strategy == browsenet
  -> isolated official BrowseNet revision
  -> GLiNER entities + ColBERT entity linking + Graph-of-Chunks
  -> NV-Embed-v2 passage embeddings + file-backed artifacts

strategy == proprag
  -> isolated official PropRAG revision
  -> proposition extraction + entity/proposition/passage graph
  -> NV-Embed-v2 stores + file-backed graph artifacts

anything else
  -> ValueError
```

BrowseNet and PropRAG use a process boundary rather than importing their
dependencies into the main environment. `scripts/setup_official_baselines.sh`
checks out exact upstream commits into an ignored directory. The parent stages
the prepared corpus, starts the official index or persistent retrieval worker,
and accepts only structured JSON responses. Snapshot metadata binds the
official revision, source digest, and corpus fingerprint before evaluation.
MuSiQue uses BrowseNet's native decomposition template. Because BrowseNet does
not provide a MultiHop-RAG template, MultiHop-RAG uses its official HotpotQA
template without changing the retrieval algorithm. PropRAG's official example
indexes and queries on one object; the persistent query worker therefore calls
its cache-aware index entry point once at startup to restore the transient
proposition maps from the completed artifacts.

The indexer selects all `.txt` and `.md` files. It applies no company or sample
filter and has no fallback grouping for unsupported datasets. HopRAG accepts
only the active known
corpus tags (`multihoprag`, `musique`) because each needs its
official problem-context grouping; another tag fails explicitly.

Both preparation scripts publish `corpus_manifest.json` beside the corpus.
The fingerprint binds the prepared source set and content; the manifest also
binds the complete query ID set, and MultiHop-RAG additionally records a
canonical prepared-query record digest. Full benchmarks require the manifest
fingerprint to match the completed index artifact. Query IDs, and query records
when that digest is present, are checked before any retrieval client starts.

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
- Returns `q_minus` and `q_plus`; missing keys, invalid JSON, schema-invalid
  records, or more than three questions raise after client retries. Empty
  Q-/Q+ lists are intentional valid outputs.
- Empty strings, within-channel duplicates, source-relative wording, and exact
  Q−/Q+ duplicates are removed deterministically before storage. The filter
  does not score semantic quality or introduce a dataset-tuned threshold.
- `RAG_QUESTION_SCHEMA=legacy` retains the string-only contract. The opt-in
  `grounded_v1` contract requires a verbatim source quote and source anchors,
  plus a quote-contained Q− answer or non-empty Q+ missing information.
  Invalid individual records are logged and removed without discarding valid
  siblings or failing the document.
- The experimental `linked_v2` schema retains that grounding contract and
  adds `continuation_anchor` to Q−. A non-empty value must equal the complete
  Q− answer and is requested only for a specific named entity that can anchor
  a relation in another document. Empty anchors are valid. Auxiliary
  `anchor_entities` that cannot be verified against the chunk are removed
  without discarding an otherwise grounded question; `grounded_v1` retains
  its stricter all-fields-valid behavior.

`indexing/embedding.py`

- Reuses cached vectors only when model, revision, endpoint, dimensions,
  encoding role, instruction, and normalized text all match.
- Batches cache misses through the external embedding endpoint.
- Cold timing runs disable both generation and embedding caches.
- Restores sparse results to original positions and verifies response count,
  non-empty vectors, and consistent dimensions. Every returned vector must
  match `NEO4J_VECTOR_DIMENSIONS` before it can be written or queried.
- Index artifacts record the served model, declared revision, and vector
  dimension. A measured run that changes any of these values builds a new
  cold index under a new run ID; an earlier index is not relabelled or reused.

The Q−/Q+ generation cache key includes the generation model, declared model
revision, sampling seed, question schema, prompt digest, source digest, and
chunking flags. It accepts early v1 records whose title was stored
only at document level and backfills that title on each in-memory chunk before
graph writing. This compatibility normalization does not rewrite the cache or
change its generated questions.

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
- `RAG_PRECOMPUTE_RECIPROCAL_HOPS=true` evaluates reverse Q−→Q+ nearest
  neighbors in bounded concurrent index-time pages, groups accepted IDs by
  HOP edge, and writes each edge once. The grouped write prevents lost list
  updates while retaining the same nearest-neighbour rule.

`indexing/answer_links.py`

- Runs only for `linked_v2`, after all chunks and grounded questions are
  visible. It token-normalizes complete continuation anchors and finds exact
  contiguous mentions in one corpus scan.
- Each normalized answer is stored once as an `AnswerAnchor`. `ANSWER_ANCHOR`
  joins grounded Q− nodes to that shared record and `MENTIONED_IN` joins it to
  exact corpus mentions. This preserves the same source-question-to-target
  paths without materializing their Cartesian product for common answers.
  Benchmark questions, gold paragraphs, hop labels, score thresholds, and
  semantic candidate widths are not inputs to this pass.

### Official baseline modules

`models/hoprag/official_indexer.py`

- Stages every corpus file, routes both model types externally, and preserves
  upstream node/question generation.
- Edges are constructed inside official problem contexts: MuSiQue paragraph
  groups or MultiHop-RAG evidence lists.
- Per-document caches and stage markers support safe resume for ordinary runs;
  the measured cold runner assigns a new run-specific output root instead of
  reusing or deleting another run's cache.
- Stage 2 inserts each document once and then streams each problem group to
  bound memory. No company metadata is stored.
- Upstream HopRAG can return no nodes when both question lists are empty. The
  adapter preserves this as an empty document representation rather than a
  runtime failure. The complete input manifest remains attached to the index;
  represented and omitted source counts and digests are stored separately, so
  the omission is retained as baseline behavior and affects all full queries.

`models/ms_graphrag/official_indexer.py`

- Stages every corpus file and calls the official `Standard` build pipeline.
- LiteLLM routes generation and embedding to external endpoints. Output is
  isolated under `data/ms_graphrag_output/<corpus_tag>`.
- Expected output tables are verified; workflow errors fail the target.

`models/browsenet/official_indexer.py` and
`models/proprag/official_indexer.py`

- Stage the complete prepared corpus into a run-specific file output and call
  the pinned official implementation through an isolated Python process.
- A complete snapshot records the upstream commit, source identities, staged
  content digest, and prepared-corpus fingerprint. Evaluation fails before
  retrieval if any identity differs.
- One persistent worker loads the completed official index and serializes its
  GPU retrieval calls. The parent evaluator receives the ordered passages and
  applies the common answer and metric boundary.

## Prehop query path and branches

`models/prehop/graphrag.py::run_workflow` strips only the benchmark output
format suffix and then:

```text
RAG_GRAPH_HOP_DEPTH == 0
  -> retrieve(query, top_k=12)

RAG_QUERY_REWRITE_VARIANT == role_aligned_evidence_iterative
  and input question has at most RAG_QUERY_REWRITE_MAX_WORDS words
  -> schema-constrained initial Q-/Q+ retrieval views
  -> each view searches only its matching representation channel
  -> retrieved evidence proposes non-duplicate Q-/Q+ views
  -> stop when no new view or selected chunk appears

question exceeds the rewrite limit
  -> use the original question without a rewrite call

RAG_GRAPH_HOP_DEPTH == 1 (default)
  -> retrieve(query) for seeds
  -> deterministic NEXT/HOP expansion for the configured depth
  -> RAG_GRAPH_EDGE_VARIANT selects the full, hop_only, or next_only path
  -> a matched Q+ paragraph exposes its stored outgoing connections
  -> HOP_EDGE_FILTER=none retains every activated HOP provenance item

RAG_SOURCE_SELECTION_VARIANT == role_body_list_ranking
  -> select from the complete candidate union by numbered paragraph IDs
  -> return the first top_k known IDs, deterministically completing omissions

empty context
  -> fixed "Insufficient evidence" result, no synthesis call

non-empty context
  -> one shared external synthesis call
  -> explicit answer boundary attached without rewriting the response
```

The current operational contract uses string Q−/Q+ questions, full Q−/body/Q+
retrieval, depth-one full NEXT/HOP traversal, matched-paragraph connection activation, unfiltered
stored HOP edges, evidence-conditioned iterative role rewriting for questions
of at most 32 words, and one complete candidate-ordering call. The one-pass,
rewrite-all, additive-view, `global`, reciprocal-filter, exact-activation,
edge-variant, channel-variant, and no-graph paths are explicit experimental
configurations rather than implicit fallbacks.

`retrieval/hybrid.py` embeds the original query and runs vector plus Neo4j
full-text search for one channel (`body`, `q_minus`, or `q_plus`). Vector and
full-text branches share one Cypher request per representation; enabled
representations run concurrently. Their results fuse into one ordered list
using equal reciprocal ranks `1 / (rank + 1)`. Because Cypher aggregation and
`UNION ALL` do not guarantee result order, Python explicitly sorts each
modality by its own raw score (descending) and stable chunk identity before
assigning ranks. Raw vector and lexical scores never cross modality boundaries,
and there is no modality weight or query-time fusion constant.

`retrieval/retrieve.py` searches the body representation with the original
question. When rewriting is active for a compact question, Q− and Q+ each use
their generated role-specific views; otherwise they also use the original
question. Multiple views are fused inside their role first, so Q−, body, and
Q+ each contribute one ranked list rather than gaining weight from the number
of generated views. Q- and body hits have the direct-evidence role; Q+ hits
have the dependency-seed role. Q− and Q+ vector and full-text rows retain the
exact matched question-node IDs while collapsing to owner chunks, and their
union is preserved when representation lists merge. Q+ IDs activate the
established dependency edges; under `linked_v2`, Q− IDs activate grounded
continuation edges. Enabled representation
results form a set union. Each owner retains
`1 / (rank + 1)` evidence from every representation list in which it appears;
these values define a representation order without mixing backend-specific
vector or lexical scores. Direction remains expressed by graph role rather
than a learned or fitted channel weight.

- `HYPO_CHANNEL_VARIANT=body_only`: body direct evidence only; this query-time
  channel control does not require rebuilding the complete index.
- `qminus_only`: Q- direct evidence only.
- `qplus_only`: Q+ dependency seeds only.
- `single_combined`: Q-/Q+ once each, set union, no body.
- `full`: Q-/body direct evidence plus Q+ dependency seeds.

The searches run concurrently. There is no second Q- support search: a Q-
hit already identifies its owner evidence chunk, while a Q+ hit reaches target
Q-/body evidence through the pre-built `HOP_ANSWER` relation. The query
embedding is created once before parallel channel search and passed unchanged
to every vector channel and final scoring call.

`retrieval/scoring.py` reuses the body and source-Q+ document embeddings stored
during indexing and embeds only the user query. Body similarity defines the
semantic score for direct/NEXT candidates; a HOP candidate uses
`min(body similarity, best individual source-Q+ similarity)`. Candidates are
ordered once by this semantic score and once by their retained representation
evidence. Equal reciprocal ranks from the two orders are summed for final
selection. `RAG_FINAL_RANK_VARIANT=semantic_only` and
`representation_only` retain one of those two orders for a declared
sensitivity analysis; `fused` is the default. HOP semantic evidence can also
be isolated with `RAG_HOP_SEMANTIC_VARIANT=body_only` or `bridge_only` instead
of the default `body_bridge_min`. These query-time switches are written to
benchmark ablation metadata. The default uses rank fusion rather than
calibrated raw-score interpolation. Role rewriting changes channel queries
before retrieval. The default selection passes the complete fused candidate
union to one paragraph-number candidate-selection prompt
and returns its first `top_k` known IDs. Unknown IDs are ignored, duplicates
collapse, and omitted known IDs retain the deterministic input order. Publisher, publication time,
author, and category are included only when present in the source-manifest
sidecar; no dataset identity, gold label, retrieval path, score, or rank is
exposed. `global`, `round_robin`, and the body-round policies remain explicit
query-time ablations.

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
When a `linked_v2` index is selected and
`RAG_CONTINUATION_EDGES_ENABLED=true`, the
`QMinus`→`AnswerAnchor`→`Chunk` path is exposed only by the exact Q− IDs matched
through the direct-evidence channel. The flag is query-time only, allowing an
on/off comparison on the identical stored graph.
Each source retains
at most the final evidence budget of continuation targets after query-to-body
cosine ordering inside Neo4j; this is the existing top-k contract rather than
a separate candidate-width setting. A continuation target inherits the
source's Q− rank evidence and uses the matched Q− embeddings as its stored
bridge representation.
The current operational default uses owner-wide activation: when an owner is
retrieved through any Q+ node, all stored source-Q+ provenance on that owner's
outgoing HOP edges is eligible. Exact matched-Q+ intersection remains
available through `RAG_QPLUS_HOP_ACTIVATION=exact`. Bridge embeddings and
emitted path provenance follow the selected activation mode.
The online `RAG_HOP_EDGE_FILTER=reciprocal` ablation leaves every stored
node, provenance relation, HOP edge, and index unchanged. Inside the same
frontier request, each activated Q+→Q− provenance pair is retained only when
the target Q− independently retrieves that exact source Q+ as its highest-ranked
cross-document Q+ representation. Its ANN pool is the number of Q+ nodes in
the target document plus one, which is the structural minimum needed to admit
one foreign-document result after exclusion; there is no acceptance threshold
or tunable candidate width. The default `none` policy performs no reverse ANN
and does not filter activated provenance. `reciprocal_offline` applies the
same reverse rule from materialized edge IDs and performs no query-time reverse
ANN. Traversal constructs only the selected NEXT/HOP/filter Cypher branches,
avoiding inactive ablations on the query hot path.
Q−/body-only seeds and graph-discovered nodes expose NEXT only, preventing an
unrelated Q+ attached to a direct-evidence chunk from triggering a HOP. NEXT and
HOP paths are ranked separately per expansion step, then fused per target
chunk. A NEXT target inherits the source's total representation evidence; a
HOP target inherits only the source's Q+ evidence. In either case the inherited
value is multiplied by `RAG_GRAPH_PATH_DECAY`, whose default is the reciprocal
one-edge value `1 / (depth + 1) = 0.5`. Values 0 and 1 are retained only as
declared propagation sensitivities. This attenuation prevents an expanded
target from tying its directly retrieved owner in the default configuration;
the switch also makes the assumption directly testable.
The structurally bounded results are retained without a candidate reservoir or
graph-search floor. HOP candidates compare the query against each indexed
source Q+ separately, take the best bridge similarity, and use
`min(body, bridge)` as the semantic score. This requires agreement on both
sides without a mixing weight. The default evidence-conditioned rewrite
repeats retrieval only while newly proposed role questions select at least one
unseen chunk. Exact normalized question and chunk identities provide the stop
rule; there is no fitted round count, score gate, hop label, dataset branch,
per-edge generation, or runtime ANN supplement.
`RAG_QUERY_REFINEMENT_MAX_ROUNDS=0` leaves that evidence-stability rule
unchanged. A positive value adds an operational upper bound on follow-up
generation calls. The executed count, configured cap, and stop reason are
written to the query-rewrite trace.
Targets already present in the representation-union pool retain their direct
rank evidence and semantic inputs; traversal only adds path provenance to
them. Graph-only targets receive inherited rank evidence and bridge semantics.

`retrieval/text_utils.py` contains only normalization, Lucene sanitization,
context formatting, node identity/dedup, and RRF helpers.

### Diagnostic controls and timing

#### Component evaluation contract

All component analyses use the 2,417-question MuSiQue split and the current
4,096-dimensional `qwen3-embedding-8b` index. Paired query-stage conditions
reuse one completed index and hold query IDs, model revisions, seed, top-k,
prompts, and judge state fixed. Results are joined by immutable query ID and
paired effects use 10,000 bootstrap resamples with seed 42. Latency is compared
only within one synchronized fixed-concurrency run; fixed-candidate analyses
remain separate from complete query-pipeline runs.

| ID | Stage | Intervention | Primary output |
|---|---|---|---|
| Ablation 1 | Query: graph expansion | One-step `NEXT` and `HOP_ANSWER` expansion on versus off | Answer, support, and retrieval passes |
| Ablation 2 | Query: refinement | Evidence-conditioned follow-up views on versus initial rewrite only | Answer and support |
| Ablation 3 | Query: candidate selection | Question-role selection versus integrated top 12 | Answer and support |
| Ablation 4 | Fixed candidates: ranking | Recompute rank signals and graph-distance weights | Support |
| Robustness | Fixed candidates: input order | Reference order versus deterministic shuffle | Selected-set overlap and support |
| Timing | Complete query path | Record non-overlapping stage timers | Within-run stage shares |

Ablations 1–3 rerun the complete query path while changing only the named
query-stage condition. Ablation 4 and the order robustness test reuse identical
candidate IDs, titles, texts, and annotations; they do not generate new
answers. The timing analysis separates query refinement, retrieval, graph
expansion, deterministic scoring, candidate selection, and synthesis.

The benchmark records `retrieve_ms`, `rewrite_ms`, `synthesis_ms`, and the
compatibility aggregate `traversal_ms`. It also splits the latter into
`graph_expand_ms`, `deterministic_score_ms`, and `candidate_order_ms`.
This prevents the generation-model list-ordering call from being reported as
database traversal time.

`RAG_CANDIDATE_ORDER_TRACE_PATH` is a diagnostic-only JSONL sink. When set, each
candidate-ordering call appends the query, canonical pre-call candidate pool,
stored source/paragraph identity, semantic and representation rank signals,
per-channel representation scores, model-returned IDs, and final selected IDs
under a process lock. The trace is
used by `scripts/replay_frozen_candidate_order.py` to replay the canonical
deterministic fused order and a deterministic hash-shuffled order over the
exact same questions and candidate texts. The replay reports selection stability and MuSiQue supporting-paragraph
metrics, but does not generate new answers. Duplicate question texts are joined
through the completed benchmark's stable query IDs and the generated Q−/Q+
views retained in both traces, not by question string alone. An
ambiguous assignment with different gold support labels is rejected.
Checkpointed replay artifacts can be continued
with `--resume`; their trace path, benchmark path, gold-query path, tested
orders, and shuffle seed must match. Normal benchmarks do not set this
variable.

`scripts/analyze_full_frozen_rank_variants.py` reconstructs deterministic rank
variants from every captured candidate pool and evaluates MuSiQue supporting
paragraphs. It does not call the answer model and therefore reports no answer
EM/F1. A result is eligible only when its query count and ID mapping match the
complete prepared split. When per-channel scores are present, decay 0.5 must
reproduce the captured graph-only score before decay 0 and 1 are evaluated.

`scripts/analyze_full_stage_profile.py` accepts one complete benchmark executed
at a declared fixed concurrency. It validates detail/trace alignment and
summarizes the non-overlapping rewrite, retrieval, graph expansion,
deterministic scoring, candidate-ordering, and synthesis timers. The
compatibility `traversal_ms` aggregate is excluded from this sum. Absolute values remain
specific to the declared concurrency and service load, so only the within-run
stage decomposition is used for interpretation.

`scripts/analyze_gold_hop_coverage.py` persists both aggregate coverage and a
query-level structural label. After a complete graph-on/off pair,
`scripts/analyze_graph_shortcut_effect.py` uses that fixed label to report
paired effects where gold paragraphs are or are not joined by a stored edge.
The grouping is retrospective, an edge need not have been activated, and
generation is rerun separately. The output is therefore a bounded exploratory
test of a local-shortcut interpretation, not evidence that a complete
multi-hop route was compressed into one hop.
`scripts/analyze_presentation_controls.py --exclude-latency` omits latency from
paired output when compared runs were resumed or did not share a controlled
load window. This is an analysis-time reporting guard; it does not modify the
source benchmark artifacts.

## Prompt inventory

Project-owned prompt templates are deliberately limited to:

- `utils/prompts/indexing.py`: indexing-time Q-/Q+ generation.
- `utils/prompts/query_rewrite.py`: bounded Q−/Q+ retrieval views for compact
  questions, evidence-conditioned follow-up views, and numbered-paragraph selection of
  the complete candidate union.
- `utils/prompts/shared.py`: one dataset-neutral final-answer prompt shared by
  Prehop, Naive, and the HopRAG adapter. It asks the model to connect required
  intermediate entities silently, return only the short final answer, and
  abstain only when a required evidence link is absent. It does not expose or
  request chain-of-thought.
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

Prehop begins rewriting only for questions within the fixed input-length
limit. It can add evidence-conditioned role views while exact identities keep
changing, then makes one complete-list selection call. Official HopRAG's upstream `bfs_node` traversal
includes its published LLM helpful/helpless node judgement. The adapter routes
that call externally. MS
GraphRAG also retains the official package's extraction/community/report/search
prompts. Those upstream prompts are baseline algorithms, not hidden Prehop
gates.

## Comparison settings

- Prehop and Naive use the same six-sentence chunks, top-k 12, and final
  synthesis prompt.
- HopRAG keeps the official repository's end-to-end top-k 20.
- MS GraphRAG uses the official LocalSearch API and context-budget
  configuration for entity-grounded passage QA. The adapter does not route
  between LocalSearch and GlobalSearch using query keywords.
- BrowseNet keeps its official five-subgraph retrieval, GLiNER extraction,
  ColBERT threshold 0.9, and NV-Embed-v2 encoder. The evaluator supplies the
  shared short-answer synthesis after official retrieval.
- PropRAG keeps proposition extraction, 200 retrieved passages, five passages
  for answer context, and NV-Embed-v2. The evaluator records the full ranking
  for retrieval metrics and uses the first five passages for synthesis.

Official full-system baselines retain their published budgets, so paper tables
and captions state the unequal settings. A one-source-one-vector Naive run
changes the evidence unit and is reported, if used, only as a separate
chunking sensitivity analysis.

Prehop, Naive, HopRAG, BrowseNet, and PropRAG return an explicit answer boundary
after synthesis; empty-context abstentions use the same boundary. MS GraphRAG
instead requests a short `Final Answer:` span through the official LocalSearch
response-type parameter. The canonical metric extractor recognizes these
explicit boundaries, preserves an unmarked response in full, and does not
truncate a marked prediction before scoring.

## Evaluation output contract

The benchmark emits deterministic normalized answer EM/F1 as a downstream
answer signal; benchmark-annotated gold-evidence retrieval is the primary
effectiveness endpoint. MuSiQue uses answer aliases. The LLM-as-a-judge
`score` is an optional exploratory semantic-correctness field for aliases and
equivalent wording;
`groundedness` and `hallucination` are separate context-directed diagnostics.
None replaces deterministic answer scoring, and without qualified-human
validation none enters quantitative submission results or system rankings.

Evidence metrics follow the prepared gold unit for each dataset:

- MultiHop-RAG reports official any-hit Hits@k, MRR@10, and MAP@10 on non-null
  queries. Normalized/token-overlap `evidence_fact_recall@k` and title-level
  document precision/recall/F1 remain explicitly diagnostic. Null queries
  report refusal and attempted-answer hallucination separately and do not
  enter retrieval denominators.
- MuSiQue reports supporting-paragraph/title precision, recall, and F1. Its
  paragraph-level gold evidence is not compared with the Prehop six-sentence
  fact matcher.

Missing gold units are emitted as `-1`, while an evaluated query with no match
is zero. Paper aggregates exclude failed, incomplete, and unreconciled rows.
Subset artifacts are development records only and do not enter reported
quantitative results. Complete-split paired analyses record the evaluated ID
digest and are interpreted as descriptive diagnostics, because the prepared
splits were also inspected during configuration development.
For query-only ablations, `--expected-ablation-difference` requires the named
metadata key and no other ablation key to differ. The active-index snapshot,
models and seed, code provenance, benchmark concurrency, and judge state must
also be identical. An index-changing paired analysis must opt into
`--allow-index-variant`; corpus fingerprints and stable query identities
remain mandatory even under that override.
`scripts/performance_gate.py` fixes the final effectiveness gate to the four
official MultiHop-RAG ranking metrics and the five official MuSiQue
answer/support metrics. It chooses the strongest supplied non-Prehop baseline
separately for every metric and requires the declared relative gain on all of
them. It rejects incomplete, fingerprint-mismatched, query-mismatched, and
non-full artifacts unless an explicitly non-paper exploratory override is
used.
The exact metric definitions, official evaluator references, paper-eligibility
rules, and reporting decisions are maintained in the local, intentionally
untracked `docs/prehop_paper.md`. This architecture document summarizes the
implemented evaluation contract but does not replace that paper specification.

The runner checkpoints its result and report artifacts every ten completed
queries by default and always writes once more at completion. This bounds lost
work after interruption without rewriting the growing result and trace files
after every query. `RAG_BENCHMARK_CHECKPOINT_EVERY` can change the interval;
the chosen value is recorded in the artifact and does not enter measured query
latency.

`RAG_BENCHMARK_RESUME=true` resumes only an existing `in_progress`
deterministic benchmark. It rejects an enabled supplemental judge, mismatched
query identity, configuration, model, corpus/index identity, duplicate or
foreign query IDs, and missing or misaligned traces. Successful rows are
retained and error rows are run again. The final artifact records the retained
and resumed query sets and their code provenance separately. This is query
execution recovery, not indexing recovery or an orchestration-level retry.

## Run measurements

Every paper run invokes one dataset/strategy target with a unique `RAG_RUN_ID`.
The run records wall time, phase timings exposed by the adapter, structural
integrity, and failures without an orchestration-level retry policy.
`VLLM_MAX_NUM_SEQS` records endpoint capacity. Clients that share a generation
endpoint and event loop share one request semaphore, and Prehop processes at
most one generation request per active file. Embeddings use bounded batches.
Naive flattens 32 source documents into each embedding/write batch. Official
adapters report only timing boundaries their upstream implementations expose.

The measurement set directly addresses the indexing-time tradeoff: overall
and Prehop phase latency, index-storage size, document/chunk/question/edge counts,
Q-/Q+ and Q+-direction coverage, provenance completeness, exact NEXT topology,
cross-document HOP invariants, and observed endpoint pressure. Retrieval and
answer-quality attribution remains a separate benchmark/ablation concern; an
index with valid topology contributes structural statistics; QA effects come
from the complete query-stage controls.
Before publishing the corpus snapshot as complete, `cli/index.py` reads the
live graph and enforces the index-quality contract: embeddings and ownership
are complete; question representations are non-empty, not source-relative,
deduplicated, and role-distinct; NEXT is exactly consecutive within each
document; every HOP is cross-document, channel-consistent, provenance-complete,
within the schema out-degree bound, and has the expected Q+→Q-→owner
provenance (or the explicit body-only ablation path); and all search indexes
are online. Coverage, linkage rate, and graph density remain descriptive and
do not become dataset-tuned pass thresholds. Held-out retrieval metrics test
effectiveness separately.

For official MS GraphRAG, HopRAG, BrowseNet, and PropRAG adapters, the stored
timing includes `official_pipeline_seconds` plus any stage boundaries exposed
by the adapter; the runner does not infer boundaries that the upstream package
does not expose. Prehop retains its finer phase timings; Naive reports its
aggregate pipeline and measurement timing only.
`scripts/run_paper_target.sh` creates a cold target without deleting shared
state: it disables the in-repo chunk and embedding caches, gives every
file-backed official baseline a new run-specific output root, clears Neo4j
when the selected strategy uses it, and runs the complete prepared split at
query concurrency 4. Existing run IDs and dirty tracked worktrees are rejected.
MS GraphRAG relationship drops caused by missing extracted entities are
recorded as integrity warnings in the target result rather than silently
treated as a clean graph.

`index_capacity` records the size of the persisted index that each strategy
uses during retrieval. Prehop, Naive RAG, and HopRAG retrieve from their
strategy-scoped Neo4j nodes, relationships, properties, and search indexes.
Their recorded value is a versioned logical-payload estimate: vector elements,
list elements, and graph records are counted at eight bytes, with selected text
property characters added directly. It does not represent the physical Neo4j
store size and excludes Neo4j record, page, transaction-log, and search-index
file overhead. MS GraphRAG does not use Neo4j in this repository; it retrieves
from files under `data/ms_graphrag_output/<corpus-tag>/`. Its recorded value is
the physical size of those local retrieval artifacts, excluding copied input,
cache, and log directories (`_input`, `_cache`, and `_logs`). Original corpus
files, debug output, run logs, and temporary files are not index storage for
any strategy.

These values share the reporting concept *index-storage size* but not the same
physical measurement method: the Neo4j values are logical estimates, whereas
the MS GraphRAG value is an on-disk file total. The measurement method and
definition version are stored with the value, so reports must retain that
distinction rather than describe the numbers as directly equivalent database
sizes. Capacity is measured after `timing_seconds.total_elapsed_seconds` is
frozen; reporting overhead is therefore excluded from indexing time. A
capacity-measurement failure marks the indexing run incomplete.

As a HOP-connectivity diagnostic, the runner resolves every full-query
evidence title against indexed documents, then reports the fraction of fully resolved
gold queries and gold document pairs connected by at least one `HOP_ANSWER`.
