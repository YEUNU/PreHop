# Architecture

This document describes implementation behavior. `core/config.py` owns runtime
defaults; `core/strategy_registry.py` owns supported methods, primary order,
upstream revisions, and paper policies. Experimental controls are specified in
[PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md), measured evidence in
[RESULTS](RESULTS.md), and launch procedures in
[THROUGHPUT_EXECUTION](THROUGHPUT_EXECUTION.md).

## Strategy dispatch and indexing branches

The primary order is Prehop, Naive RAG, HopRAG, MS GraphRAG, LightRAG, GFM-RAG,
and LinearRAG. The two supported paper datasets are MultiHop-RAG and MuSiQue.
Removed methods do not supply primary comparison cells.

`cli/index.py::run_indexing` acquires a strategy/corpus lock in the working
checkout and calls `_run_indexing_unlocked`. Namespace isolation is separate
from that local lock; it is not a cross-worktree database lock.

| Method | Index construction | Query path |
|---|---|---|
| Prehop | Six-sentence passages, Q−/Q+ nodes, NEXT and directed HOP links | Representation search, activated links, LLM evidence selection |
| Naive RAG | The same six-sentence passages and body embeddings | Dense body search and shared answer synthesis |
| HopRAG | Native question-linked passage graph over the complete staged corpus | Native BFS, five hops, top-k eight |
| MS GraphRAG | Standard indexing: text units, entities, relations, communities, reports | Native LocalSearch |
| LightRAG | Native insertion and graph storage | Mix retrieval, top-k 40, chunk top-k 20, native answer |
| GFM-RAG | Native extraction and checkpoint-defined graph components | Native single-pass QA, top-k five |
| LinearRAG | Native relation-free Tri-Graph, local MPNet and spaCy | Native QA with top-k five and registered query parameters |

File-backed external methods run in isolated Python processes. Their adapters
stage input, configure transport and producer scheduling, preserve source IDs,
and observe native responses. They do not edit pinned upstream source.
[Runtime requirements](RUNTIME_REQUIREMENTS.md) define local models and declared
response interventions.

## Shared input contract

`models/prehop/indexing/chunking.py` owns the parser and window splitter:

- `parse_pages_offline` accepts a first-line `Title:` header and optional
  `--- Page N ---` markers. Without markers, the body is one logical page.
- `split_fixed_sentence_windows` forms six-sentence windows within each page,
  retaining the final partial window. It preserves pipe-delimited text.
- Prehop and Naive use the same splitter. External systems retain their native
  indexing units; equal rank cutoffs do not imply equal passage sizes.

Prepared `corpus_manifest.json` files bind source IDs, file and corpus-record
hashes, query IDs, counts, and query-record hashes. MuSiQue paragraph IDs and
filename-derived source IDs are distinct. Full benchmarks require a completed
index with matching corpus identity and verify the active source snapshot.
Gold evidence is used for evaluation, not index construction or retrieval.

## Prehop index construction

CPU parsing uses a spawn-based process pool. A bounded rolling document window
limits resident work; completed tasks release slots without waiting for a whole
batch. Document failures are recorded, and incomplete source coverage cannot
produce a complete index.

| Module under `models/prehop/indexing/` | Responsibility |
|---|---|
| `chunking.py` | Parsing, windowing, generation cache, intermediate output |
| `knowledge_mapping.py` | Question generation and question-record validation |
| `embedding.py` | Role-aware cached embeddings and bounded request batches |
| `graph_writer.py` | Document replacement, question ownership, indexes and NEXT |
| `hop_edges.py` | Question-based cross-source HOP construction and provenance |
| `answer_links.py` | Optional grounded continuation links for `linked_v2` |
| `body_links.py` | Explicit body-to-body representation ablation |

The primary legacy schema generates up to three Q− and three Q+ strings per
passage. Q− asks about facts answered by the passage; Q+ asks for information
elsewhere. Empty lists are valid. Invalid types and blank questions fail;
deduplication and source-relative wording filters remove unsuitable records.
Grounded and linked schemas remain explicit experimental settings.

Each body and individual question has a document embedding scoped by its title.
Q+ additionally stores an instructed query embedding for outgoing ANN search.
Embedding reuse requires matching model, revision, endpoint, role, instruction,
dimensions, and normalized input. Cold target wrappers disable generation and
embedding caches and allocate fresh storage rather than deleting shared state.

Document replacement writes complete Document/Chunk/question subgraphs in one
transaction; ordered NEXT edges are written separately. Graph batches respect a
document count and an 8 MiB logical parameter budget. Only transaction-memory
failures split a failed document group; successful prefixes are not replayed.
An exhausted write failure stops further work and prevents a complete index.

After corpus flushing, each source Q+ retrieves a Q− from another source file.
The ANN pool includes the source's own channel count plus one foreign slot;
filtering excludes the source file. The selected Q− owner is the destination.
This is the best eligible returned ANN candidate, not a guaranteed exact global
nearest neighbor or a verified answer. No second body ANN lookup is required.

Multiple questions resolving to the same passage pair merge into one
`HOP_ANSWER` edge. Question IDs/texts retain provenance; `ANSWERED_BY` and
`SUPPORTED_BY` retain question-to-evidence paths. The primary constructor skips
HOP construction if Q+ is disabled. The explicit body-link profile is the
exception described below. Reciprocal-hop precomputation is enabled by default;
the default query policy does not filter edges by reciprocity.

The `linked_v2` experiment also stores normalized answer anchors and exact
cross-source mentions. These optional relations do not change the legacy
primary graph. Prepared MuSiQue paragraphs are individual source files, so
cross-source exclusion need not imply different article titles.

## Structured generation contracts

`core/structured_outputs.py` defines schemas for question generation, role
rewriting, refinement, and candidate selection. Requests use strict JSON-schema
response formats through the common gateway. Validators reject extra fields,
wrong types, duplicate keys, malformed JSON, refusals, and incomplete responses.
Ranking requires exactly the requested number of distinct IDs from the supplied
candidate pool; it does not fill missing IDs or repair rankings.

`prehop-json-schema-v3` omits unsupported wire keywords such as `uniqueItems`
while enforcing uniqueness locally. Materialized schema and prompt digests are
part of semantic identity. The controlled format-retry profile retries the same
request under one shared maximum of five wire attempts, including transport
retries. Discarded responses and available usage remain recorded; retries do
not select among valid outputs by quality. Final synthesis remains text output.

## Prehop query path and branches

The primary settings use depth one, full NEXT/HOP expansion, Q−/body/Q+ search,
Q+-owner activation, unfiltered stored links, and final LLM evidence selection.
Questions of at most 32 words receive role-aligned rewriting and iterative
refinement; longer questions use their original text on all enabled channels.

1. Search each enabled representation with vector and full-text search.
2. Merge representation hits into owner passages while retaining question IDs.
3. Expand stored NEXT and activated HOP links from the starting-passage pool.
4. Score the complete candidate union and select up to 12 passages by LLM.
5. For refinement-eligible inputs, repeat while new views select new evidence.
6. Synthesize a short answer from selected evidence, or return the fixed
   insufficient-evidence response for empty context.

`retrieval/hybrid.py` sorts vector and lexical results independently by raw
score and stable identity, then combines reciprocal ranks `1 / (rank + 1)`.
Raw scores are not mixed across modalities. `retrieval/retrieve.py` fuses views
within each role before fusing roles, avoiding extra weight merely from having
more rewrites. The complete owner union becomes the base candidate pool.

The original question searches bodies. Role views search their question
channels and, under the default selection policy, also search bodies. Additional
role-body-only hits join the selection pool but do not seed graph expansion.
Each representation retains at most top-k owners with candidate multiplier one;
additional views can enlarge the full union. Question searches allocate three
times the owner budget before collapsing individual questions to their owners.

`retrieval/traversal.py` walks NEXT in both directions and HOP only in its
stored direction. Under the default `HOP_SEED_POLICY=qplus`, only passages
matched through Q+ expose HOP links. Owner activation exposes all provenance
on that passage; `RAG_QPLUS_HOP_ACTIVATION=exact` restricts it to matched Q+ IDs.
Graph-discovered targets are not expanded again within the one-step pass.

A graph-only NEXT target inherits its source's total representation score;
a HOP target inherits its source's Q+ score. Both use path decay 0.5. Direct
candidates retain their direct score when also reached by a graph path.
`retrieval/scoring.py` uses query-to-body similarity for direct/NEXT candidates
and the minimum of body similarity and best source-Q+ similarity for HOP
candidates. Semantic and representation orders are fused by reciprocal rank.
The default LLM selector receives all candidates as numbered passages, without
gold labels, retrieval scores, or path metadata.

Refinement stops when no new role view or selected passage appears.
`RAG_QUERY_REFINEMENT_MAX_ROUNDS=0` adds no numeric round cap; a positive value
limits follow-up generation. The trace records rounds and stop reasons.
Depth zero disables graph expansion but does not independently disable rewriting
or selection. Experimental channel, edge, reciprocal-filter, semantic-scoring,
and linked-continuation switches remain distinct from the primary defaults.

## Explicit representation ablations

`core/prehop_ablation.py` validates `question_full`, `question_body`, and
`body_body`. All use original queries, all-seed HOP activation, body-only
semantic scoring, depth one, and the existing final selector. Under all-seed
activation, HOP targets inherit the source's total representation score.
These settings are not the historical primary benchmark baseline.

A/B can read an existing compatible question graph through
`--reuse-existing-index`. They retain its namespace and exact index-stat
identity; writes and HOP rebuilds are disallowed for reused indexes. C uses a
fresh namespace and the frozen reference's per-passage degree budget.
`scripts/clone_prehop_body.py` can copy body properties and Document/Chunk,
CONTAINS, and NEXT structure without copying questions or HOP edges, then build
body links. Clone-only costs are not cold indexing costs.

The launcher prints a plan unless `--execute` is supplied. Its metadata uses
`prehop-representation-ablation-v1`. See
[PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md) for commands, controls, and
permitted interpretations.

## HopRAG indexing and reference checkout

`models/hoprag/native_runtime.py` loads the pinned prepared HopRAG installation.
The root `third_party/HopRAG` is reference-only and does not define the active
runtime. The adapter stages the complete corpus as one edge-construction group;
queries and gold evidence do not determine edge groups. Native generation may
omit a document when either question list is empty; represented and omitted
source IDs remain separately recorded.

The native constructor scores question pairs with vector dot products and
keyword Jaccard, then applies native destination selection and trimming.
Prehop instead resolves each Q+ through an ANN Q− index with source exclusion.
Both precompute connections; this architectural difference alone establishes
neither novelty nor a retrieval or cost advantage. The older chunked edge helper
functions in the adapter are not installed by the active setup.

<a id="legacy-external-modules"></a>
The historical external-module anchor resolves here for reference-checkout links.

## Adapter producer parallelism

| Method | Producer behavior |
|---|---|
| Prehop | Bounded documents, prefetch, and per-document chunk lookahead in `parallel_adapter.py` |
| Naive RAG | Batched body embeddings; no index-time generation |
| MS GraphRAG | Native extraction/community request concurrency |
| LightRAG | Native insertion concurrency separate from LLM request limits |
| GFM-RAG | Native OpenIE thread producers, then local linking/checkpoint work |
| LinearRAG | Native local NER/MPNet; QA batches preserve native retrieval |
| HopRAG | Ten document workers with native per-document chunk worker count one |

LinearRAG batches concurrent questions for a bounded 10 ms collection window,
up to `RAG_BENCHMARK_CONCURRENCY`, then invokes native `qa(questions)` once.
Retrieval is native and sequential; answer generation uses its native worker
pool. Batch failures propagate to all members without an adapter retry.
The registry sets `iteration_threshold=0.4`, `passage_ratio=2`, and
`top_k_sentence=3` to match the native entrypoint.

## Inference transport

`core/inference_transport.py` owns gateway identity, model aliases, dimensions,
timeouts, retries, and generation seed semantics. Generation and embedding use
one OpenAI-compatible LiteLLM base. Paper mode rejects unsupported public
provider aliases and omits LLM seeds; evaluation/sampling seed 42 is separate.
Existing result artifacts retain their historical generation settings.

Embedding responses must contain one finite, correctly sized vector per input
with unique, gap-free response indices. Context-size or HTTP 413 errors allow
order-preserving bisection; unrelated errors and failing singletons propagate.
External compatibility aliases are confined to validated native child processes.

## Evaluation output contract

`cli/benchmark.py` records official MultiHop-RAG retrieval and QA measures,
MuSiQue answer EM/F1, and separately named global paragraph-support metrics.
Normalized/fuzzy fact recall is diagnostic and differs from the manuscript's
literal exact-fact recall. Missing metric applicability is `-1`; evaluated
nonmatches are zero. Terminal failures receive zero primary quality scores and
remain visible in failure counts. Integrity failures stop the affected target.

The default checkpoint interval is ten completed queries. Resume requires an
`in_progress` deterministic result, the same query/model/index configuration,
and valid retained identities/traces. It runs missing IDs only, preserving both
successful and terminal-error rows. A resumed batch is not an uninterrupted
throughput measurement. The representation-ablation launcher itself does not
provide resume.

`record_paper_completion.py` records finished execution without final paper-policy
validation. It emits the legacy `admitted` receipt label with
`verification=disabled_by_user`; it does not recalculate metrics or certify
publication eligibility. Runtime and index checks remain separate. Optional
analysis tools are not automatic completion gates.

## Complete-index reuse for final benchmarks

`core/index_reuse.py` supports links from full-index supervisor completions
(version 2) and the separate legacy one-query matrix protocol (version 1).
Both retain original index statistics, corpus identity, method policy, and
measured construction costs. File-backed methods receive byte-verified query
copies; service-backed methods retain their source namespace. Copy preparation
cost is separate from original indexing cost. Query caches do not rewrite the
bound source index. A canary or index-only artifact cannot supply a full
benchmark score.

## Amortized throughput cost (evidence v3)

`core/execution_profile.py` binds transport and producer settings to provenance.
Version 3 uses one shared generation/embedding request semaphore in
`core/inference_queue.py`. Legacy versions remain readable for historical runs.
Profile limits are request bounds, not evidence of GPU saturation or exclusive
remote resources.

Index cost uses original successful pipeline wall time. Query batch cost uses
dispatch through the last answer/failure, distinct from individual response
latency and cumulative benchmark-segment wall time. Neo4j storage measurements
are logical-payload estimates; file-backed measurements are physical artifact
bytes. They are not equivalent physical database sizes. Full definitions belong
in [the measurement protocol](THROUGHPUT_EXECUTION.md#final-tables-and-measurement-definitions).

## Prehop tracing

`models/prehop/tracing.py` records stage inputs/outputs, embeddings, Cypher,
candidates, refinement, and answers. Tracing is enabled by default;
`RAG_PREHOP_TRACE=false` disables it and `RAG_PREHOP_TRACE_DIR` selects storage.
Each engine reserves `data/traces/<run-id>/prehop/<namespace>/<session-id>/`.
Ordered `events.jsonl` entries refer to hashed compressed payloads. Index and
query artifacts retain trace references.

Credentials and HTTP headers are not recorded; known secrets are redacted.
Payloads otherwise contain source and model data and remain private/ignored.
Trace directories use mode 0700 and files 0600. Writes occur inline and must
not wait for the executor used by embedding semaphore waiters. Storage errors
propagate. Trace I/O during a measured phase contributes to its wall time;
trace storage is excluded from retrieval-index size.

## Campaign process ownership

Persistent supervisors record PID/start/boot identities and own their process
sessions. Cleanup targets verified descendants, sends TERM with a bounded wait,
and does not escalate to KILL. Surviving owned processes block restart. Separate
logs and atomic status files survive the initiating shell; they do not imply
automatic reboot recovery or chat notifications.

Index dispatch does not reject edits solely because a source digest changed.
Running Python processes may retain loaded code; newly launched segments record
their actual provenance. The separate `paper_campaign.py`/`run_paper_matrix.sh`
legacy full-matrix path still has an explicit evidence ledger. Its checks must
not be described as automatic final policy validation or as the rolling
controller's dispatch policy. See [execution procedures](THROUGHPUT_EXECUTION.md).
