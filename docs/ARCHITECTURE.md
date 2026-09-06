# Architecture

This document defines the current indexing and query paths at module level.
Remote generation and remote embeddings use one configured, fail-closed
OpenAI-compatible LiteLLM gateway; the repository does not start a local
generation model. Pinned local ancillary models remain part of methods that
define them. Historical changes belong in `CHANGELOG.md`, and research claims
belong in the local `prehop_paper.md`.

## External package installation

Pinned checkouts are runtime source references, not packaging work directories.
`scripts/build_official_package.py` exports the approved revision to a unique
strategy-local `artifacts/builds/` directory and passes that export to package
installation. It records the archive digest and rejects tracked, untracked,
or ignored changes in the original source before and after the build.
`RAG_OFFICIAL_BASELINE_HOME` selects the same runtime layout for setup and
worker source, interpreter, snapshot, and freeze resolution. Existing failed
attempts remain in place when a new runtime home is selected.

## Admission and transport identity

The typed transport checks the normalized gateway URL against the non-secret
`configs/paper_gateway.json` digest before client construction. Public vendor
endpoints, URL userinfo, and unregistered generation-model overrides fail
closed. Primary research workers receive canonical transport fields and only
necessary native client credentials; they do not inherit legacy routing aliases.

Target admission establishes the exact run ID, output root, namespace, and
method generation seed before comparing current and stored policy. Matrix
execution uses that target exit status, including admission failure. Standalone
target verification supports `--exact-run-id`. Admission revalidates current
v2 corpus manifests and source bytes and binds their identity along with the
result, detail rows, index statistics, runtime, and explicit evidence-contract version.

HippoRAG2 retains its native unprefixed query embeddings and native newline-to-space
normalization; its method policy records an empty instruction and `{query}`
template. The injected encoder uses the typed embedding request timeout.
Youtu's benchmark seed remains separate from its native unseeded generation;
a native-answer adapter cannot silently synthesize a replacement answer.

## Structured generation contracts

Prehop's Q−/Q+ indexing, role rewrite, evidence refinement, and candidate
ranking use registered `response_format.json_schema` requests with
`strict: true` through the same typed LiteLLM transport. The client requires
one completed, non-refused raw JSON response and validates every nested field.
It rejects fenced/prose-wrapped JSON, wrong types, extra fields, duplicate
keys, and incomplete generations. Ranking requires exactly the requested
number of distinct IDs from the actual candidate pool. It does not fill or
repair an invalid ranking. Empty question directions and linked-schema empty
continuation anchors retain their existing meaning. Final answer synthesis
keeps its text response contract.

`core/structured_outputs.py` owns these schemas. Profile
`prehop-json-schema-v1` and the schema-factory bundle digest enter index policy,
query metadata, and the v4 chunk-generation cache signature. Per-request
schema digests also enter inference telemetry. Fresh paper runs use separate
`data/index_cache/runs/<run>/<strategy>/<dataset>` directories; an existing
cache cannot satisfy a fresh-index step. Resume retains its own run cache.

HippoRAG2's controlled `hipporag2-paper-json-object-v2` profile sets the official
`BaseConfig.response_format` to `{"type":"json_object"}` for native NER and
triple extraction. Native prompts, parsers, seed, temperature and 512/2048-token
limits remain unchanged. Native QA receives no extraction response format.
The official cache and OpenIE state identity include this setting; project
snapshots also require the observed format and limits to match policy.

Youtu's controlled `youtu-extraction-json-schema-v2` profile wraps only the
constructor's public SDK client. It adds the frozen extraction JSON Schema,
checks raw completion and nested shape, and returns the identical SDK response
to the unmodified native parser. Dynamic entity/type maps and native schema
evolution remain available. Retrieval and QA clients receive no extraction
format. Construction retains temperature 0.3, no seed and no imposed token
cap. The extraction-response schema digest is distinct from the approved
starting ontology and evolved ontology digests. Old unstructured indexes
cannot be admitted under these new profiles.

The standard wire format follows the [vLLM structured-output interface](https://docs.vllm.ai/en/latest/features/structured_outputs/).
Unsupported gateway/backend schema requests fail without an unstructured
fallback. Local SDK acceptance is not evidence of a particular serving
backend or version.

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
- Primary external methods retain their upstream indexing units because
  changing them would no longer be a full-system comparison. BrowseNet,
  HopRAG, and PropRAG retain theirs as legacy/reserve adapters.

## Strategy dispatch and indexing branches

`core/strategy_registry.py` is the typed source of truth for primary order,
legacy support, external repository and revision, worker, output root, license
note, transport profile, and paper embedding identity. The CLI and shell
runners consume that registry rather than maintaining independent lists. The
primary order is Prehop, Naive RAG, MS GraphRAG, LightRAG, HippoRAG2, GFM-RAG,
LinearRAG, and Youtu-GraphRAG. BrowseNet, HopRAG, and PropRAG remain callable
legacy/reserve strategies.

`cli/index.py::run_indexing` obtains a namespace-aware strategy/corpus lock,
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

strategy == lightrag
  -> pinned official LightRAG storage initialization and insertion
  -> dual-level mix-mode retrieval and native answer path

strategy == hipporag2
  -> pinned official HippoRAG2 indexing and source metadata
  -> native rag_qa answer with source-bearing evidence

strategy == gfm_rag
  -> pinned official GFM-RAG worker
  -> validated model.pth/config.json and checkpoint-defined GNN components

strategy == linear_rag
  -> pinned process-isolated official relation-free Tri-Graph
  -> local MPNet snapshot in the primary official-faithful mode

strategy == youtu_graphrag
  -> pinned official knowledge-tree construction
  -> declared controlled public no-agent query API
  -> local MiniLM/NER plus observational provenance sidecars

strategy == browsenet
  -> isolated official BrowseNet revision
  -> GLiNER entities + ColBERT entity linking + Graph-of-Chunks
  -> LiteLLM passage embeddings + file-backed artifacts

strategy == proprag
  -> isolated official PropRAG revision
  -> proposition extraction + entity/proposition/passage graph
  -> LiteLLM embedding stores + file-backed graph artifacts

legacy browsenet | hoprag | proprag
  -> retained pinned official adapter; never a primary matrix target
```

File-backed external methods use a process boundary rather than importing
conflicting dependencies into the main environment.
`scripts/setup_official_baselines.sh` checks out exact upstream revisions into
an ignored directory and creates isolated runtimes. The preflight rejects a
revision mismatch or a tracked or untracked change in an upstream checkout.
The parent stages the prepared corpus, starts the official index or persistent
retrieval worker, and accepts only structured JSON responses. Snapshot metadata
binds the official revision, source/content digests, corpus fingerprint,
artifact inventory, exact indexed source coverage, and immutable
semantic-configuration hash before evaluation. Throughput controls are
operational configuration and do not silently change method semantics. The
external research worker loads one strategy-specific thin driver through the
typed central registry; it does not rely on placeholder modules in upstream
checkouts.

LinearRAG's GPL upstream source stays in its external checkout and is never
copied into the MIT repository. Youtu-GraphRAG's upstream
academic/research-use terms continue to apply. The primary LinearRAG mode uses
an exact local MPNet snapshot and pinned `en_core_web_trf`; Youtu-GraphRAG uses
an exact local MiniLM snapshot and pinned `en_core_web_lg`; GFM-RAG requires
content-addressed `model.pth` and `config.json`; and Youtu-GraphRAG requires a
content-addressed schema selected from the pinned checkout. MultiHop-RAG maps
to upstream `hotpot`; MuSiQue maps to `musique`. The registry records this
alias/no-chunk policy and the native effective retrieval/filter budget of 20.
`configs/paper_runtime_requirements.json` records the
machine-readable requirements, and `scripts/check_paper_runtime.py` validates
them without reading or printing secrets.
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

Both preparation scripts atomically publish `corpus_manifest.json` beside the
corpus. The current manifest schema binds distinct source IDs, source count,
file and corpus-record content digests, query IDs, query count, and query-record
digest. MuSiQue keeps paragraph IDs distinct from filename-derived source IDs;
the two digests are never aliases. Schema-v1 inputs are rejected in paper
mode. Full benchmarks recompute the active
corpus and query identities before a retrieval client starts and require them
to match the completed index artifact.

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

### Inference transport

`core/inference_transport.py` owns one typed transport: base URL, API key,
registered generation and embedding model IDs, timeout, retry policy, remote
embedding batch size, generation/embedding concurrency, seed semantics,
context/input limits, dimension, reserve, and exact query instruction/template.
The same base serves `/chat/completions` and `/embeddings`. Legacy `VLLM_*`,
ambient provider, and direct-vendor variables are rejected as public inputs.
Primary child processes receive canonical typed values and empty forbidden
aliases, preventing native dotenv imports from restoring competing settings.
Populated compatibility aliases are limited to isolated legacy children.
Internal clients do not interpret competing URL or credential variables. Empty bases, two different
bases, unregistered models, direct public-vendor routes, and unsupported
fallbacks fail closed in paper mode.

Remote embedding responses must contain exactly one finite, dimensionally
consistent vector per input with unique, gap-free response indices. Only
MessagePack/context-size errors and HTTP 413 trigger order-preserving
bisection; an unrelated HTTP 400 or a failing singleton is raised. The paper
operational configuration caps remote embedding batches at 16 and effective
concurrency at 1 without reducing the independent generation concurrency.

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
- There is no company property and no conditional index-recreation path. A
  paper cold run allocates a collision-resistant namespace and may clear only
  that namespace; it never globally deletes a concurrently usable graph.
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

### Primary external driver contracts

The common external worker owns lifecycle, structured requests, telemetry,
semantic provenance, and artifact inventory. Thin strategy drivers own only
method-specific staging and upstream calls:

- LightRAG supplies a NumPy-array embedding callback with exact count,
  dimension, and finite-value validation; storage initialization, insertion,
  insertion status, mix-mode retrieval, and finalization errors propagate.
- HippoRAG2 maps the canonical gateway into the upstream OpenAI-compatible
  fields, checks embedding width, calls native `rag_qa`, and retains both its
  answer and source-bearing evidence.
- GFM-RAG validates checkpoint/configuration hashes before constructing or
  loading the official index. Completion requires source-document coverage in
  the stored graph, not merely a staged corpus.
- LinearRAG maps the official numeric passage prefix back to source IDs through
  a sidecar, avoiding visible metadata markers in NER and embedding text. The
  native `qa()` owns retrieval, prompt construction and answer parsing; its
  configured `infer()` interface uses the typed gateway with the native
  2000-token limit. The pinned MPNet snapshot is downloaded at an exact revision and passed by local
  path for compatibility with the upstream sentence-transformers version.
- Youtu-GraphRAG stages one MuSiQue paragraph per source, maps the public
  corpus tag to the explicit upstream dataset alias, and uses native effective
  `top_k=top_k_filter=20`. The pinned agent batch entrypoint does not return
  structured answer/evidence, so the registered `controlled_adapter` calls the
  public native no-agent query API and preserves its evidence order. Each fresh
  worker calls the native `build_indices()` after retriever construction, as
  the pinned entrypoint does. It records
  staged coverage, native extraction success, and native source reachability
  as separate content-bound sidecars without injecting visible markers or
  changing retrieval/deduplication. Malformed, empty, sentinel, or swallowed
  failures are fatal.

GFM's entity-linker model snapshot remains under its pinned runtime artifacts.
Its mutable PLAID cache and metadata use the current run's
`artifacts/gfm_entity_linker` directory through the native public `root`
setting, including the graph constructor's interpolated entity linker.

### Legacy external modules

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
- The typed LiteLLM transport routes generation and embedding through the same
  gateway. Output is isolated under the run-specific MS GraphRAG root.
  LocalSearch receives the exact native community table; the adapter does not
  synthesize singleton communities or augment community membership.
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

LLM-as-a-judge is disabled for paper targets (`RAG_JUDGE_ENABLED=false`). Paper
mode prohibits the public OpenAI Batch API and any direct vendor fallback. An
explicit supplemental judge must use the same typed LiteLLM transport or fail
closed; incomplete, malformed, or self-judged output never enters primary
rankings.

Prehop begins rewriting only for questions within the fixed input-length
limit. It can add evidence-conditioned role views while exact identities keep
changing, then makes one complete-list selection call. The legacy HopRAG
adapter retains upstream `bfs_node` judgement. MS GraphRAG and the other
primary external methods retain their declared upstream extraction, indexing,
retrieval, and answer prompts. Those prompts are method behavior, not hidden
Prehop gates.

## Comparison settings

- Prehop and Naive use the same six-sentence chunks, top-k 12, and final
  synthesis prompt.
- MS GraphRAG retains Standard indexing and official LocalSearch context
  construction.
- LightRAG retains dual-level mix-mode retrieval and its native answer path.
- HippoRAG2 retains native retrieval and `rag_qa`, while the adapter preserves
  source-bearing evidence.
- GFM-RAG retains its validated checkpoint-defined GNN and local components.
- LinearRAG retains the official relation-free Tri-Graph path and pinned MPNet
  embeddings in the primary official-faithful mode. Its controlled Qwen mode
  is a distinct, currently unadmitted semantic configuration.
- Youtu-GraphRAG's declared controlled variant calls the pinned public
  `initial_question_decomposition` no-agent API, preserving its ordered
  retrieval, reranking, retry, and final answer with `top_k_filter=20`.
  MuSiQue staging preserves one prepared paragraph per source instead of
  silently re-chunking it. It does not claim the non-returning agent batch path.

Official systems retain their stated search and context budgets, so tables and
captions state unequal settings. A one-source-one-vector Naive run changes the
evidence unit and is, if used, a separate chunking sensitivity analysis.
BrowseNet, HopRAG, and PropRAG keep their previous contracts only as
legacy/reserve adapters and do not supply primary cells.

### Upstream immutability boundary

Pinned external source trees remain clean and immutable. Strategy adapters are
limited to input normalization, the declared LiteLLM transport, run-local
configuration or schema preparation, observational sidecars, and validation
after a native API call. They do not override upstream retrieval, graph
deduplication, serialization, or answer orchestration. A method-defining
override is a controlled deviation, receives a different semantic fingerprint,
and is ineligible for an official-faithful primary cell.

This distinction matters for Youtu-GraphRAG. Its native entity and triple
deduplication may retain only the first chunk identity for repeated evidence.
The adapter reports `input_chunk_coverage_complete` independently from
`native_source_reachability_complete`, plus the reachable and unreachable
counts. It does not add provenance to the graph or alter `_extract_chunk_ids_*`.
BrowseNet likewise keeps its legacy ColBERT checkpoint in the external
runtime's `artifacts/` directory and links that artifact into a run-local
layout; the pinned checkout itself stays clean.

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
is zero. The experiment ledger uses exactly `planned`, `canary_passed`,
`in_progress`, `completed_unadmitted`, `admitted`, and `failed`. A synthetic
canary is not a complete target. Completion is not admission: final admission
recomputes exact row order and count, error rows, query and ground-truth
identities, eligible counts, aggregates, corpus/index coverage, artifact
inventory, semantic configuration, model revisions, operational metadata,
exact index-stat bytes/path, runtime freeze/constraints, versioned effective method configuration,
and post-query retrieval-artifact inventory. Result JSON detail rows must equal
the complete JSONL bytes in manifest order.
Only `admitted` primary artifacts enter quantitative results. Subset and
legacy/reserve artifacts are development evidence only. Complete-split paired analyses record the evaluated ID
digest and are interpreted as descriptive diagnostics, because the prepared
splits were also inspected during configuration development.
The benchmark JSON keeps the machine execution value
`status=completed_unadmitted`. A successful per-target verifier writes the separate
run-level `admission.json` ledger entry with `status=admitted`. This keeps
execution completion distinct from permission to publish the artifact.
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
The run records wall time, service latency, worker-queue delay, end-to-end
latency, phase timings exposed by the adapter, effective concurrency,
structural integrity, and failures. Remote embeddings use batch 16 and
concurrency 1; generation has its own semaphore. Cancellation cannot leak a
global embedding permit. Official adapters report only timing and token/cost
fields their upstream implementations expose; unavailable telemetry is marked
incomplete and never estimated.

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

For MS GraphRAG and process-isolated adapters, the stored timing includes the
official pipeline boundary plus any stage boundaries exposed by the adapter;
the runner does not infer boundaries that the upstream package does not
expose. Prehop retains its finer phase timings; Naive reports its aggregate
pipeline and measurement timing only.
`scripts/run_paper_target.sh` creates a cold target without deleting shared
state: it disables the in-repo chunk and embedding caches, gives every
file-backed official baseline a new run-specific output root, allocates a
run-specific Neo4j namespace without invoking the global clear operation, and runs the complete prepared split at
query concurrency 4. A dirty tracked worktree is warned and recorded in code
provenance. A strictly verified completed target is skipped; a compatible
complete index or deterministic partial benchmark may resume; corrupt or
incompatible existing artifacts fail closed. The matrix continues after
independent target failures and exits nonzero if any target failed.
MS GraphRAG relationship drops caused by missing extracted entities are
recorded as integrity warnings in the target result rather than silently
treated as a clean graph.

`index_capacity` records the size of the persisted index that each strategy
uses during retrieval. Prehop, Naive RAG, and legacy HopRAG retrieve from their
strategy-scoped Neo4j nodes, relationships, properties, and search indexes.
Their recorded value is a versioned logical-payload estimate: vector elements,
list elements, and graph records are counted at eight bytes, with selected text
property characters added directly. It does not represent the physical Neo4j
store size and excludes Neo4j record, page, transaction-log, and search-index
file overhead. MS GraphRAG and the file-backed external methods record the
physical size and SHA-256 inventory of their local retrieval artifacts,
excluding copied input, cache, log, and temporary directories. Their completion
checks derive exact source coverage from stored chunks, passages, document
nodes, or source sidecars; staged input alone cannot satisfy coverage.

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

## Campaign process ownership

`paper_campaign.py` owns the ordered gate and full-matrix subprocesses. A frozen
plan binds effective model configuration, runtime content, the explicit evidence
contract and target order. Git/source/verifier hashes remain provenance metadata.
Changing only a commit, comment or document does not invalidate compatible
evidence. Each executed segment records its actual launch provenance; matching
settings do not imply that independently launched segments used identical code. The
systemd user unit must contain the supervisor's actual PID, and no other paper
unit may retain supervisor or native descendant processes. The shared user
resource lock is acquired before reading or replacing campaign status; a
losing launcher cannot overwrite the active owner's state. Native descendants
left after a stage block later stages and restart without being killed.

Atomic status and per-stage exit receipts complement content-bound gate
validation. Successful process exit is followed by actual evidence validation;
the final matrix requires all sixteen current admissions. Canonical secret and
URL values are redacted from logs. The harmless detachment regression proves
process/status behavior only; actual host unit and linger verification belong
to the release procedure in `RUNTIME_REQUIREMENTS.md`.

### Configuration compatibility maintenance

`core/paper_compatibility.py` resolves each method/dataset from the same typed
index, query and transport policies used by production validation. Explicit
runtime/interpreter and artifact paths retain their existing operational binding;
this contract does not promise relocation across runtime directories.
`core/generation_profiles.py` supplies the applied temperature/token settings
to owned consumers and the same values to policy identity. Pinned native
defaults retain their upstream owner and registered override metadata. Materialized
prompt and JSON-schema contents are hashed; Python source bytes are not schema
identity. Per-method semantic versions cover behavior not represented by those
settings and must be changed deliberately when such behavior changes. This is
a maintained contract, not automatic proof that arbitrary code edits preserve
behavior. Evidence protocol versions remain strict. Old admission records
require current revalidation; matching configuration never replaces corpus,
query, native artifact, result/detail or dependency checks. Historical query,
index and evaluation provenance is preserved unchanged.
