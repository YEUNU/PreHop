# Architecture

This document describes implementation behavior. `core/config.py` owns runtime
defaults; `core/strategy_registry.py` owns supported methods, primary order,
upstream revisions, and paper policies. Experimental controls are specified in
[PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md), measured evidence in
[RESULTS](RESULTS.md), and launch procedures in
[THROUGHPUT_EXECUTION](THROUGHPUT_EXECUTION.md).

## Strategy dispatch and indexing branches

The primary order is Prehop, Naive RAG, HopRAG, MS GraphRAG, LightRAG, GFM-RAG,
and LinearRAG. The paper targets MultiHop-RAG and HotpotQA (HippoRAG corpus). HotpotQA uses
the official introductory-paragraph corpus and complete development split, with
source and sentence identities preserved. Preparation and evaluation are defined
in [HOTPOTQA](HOTPOTQA.md).
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
hashes, query IDs, counts, and query-record hashes. HotpotQA evaluation requires
preserving original article titles and sentence indices through preparation. Full benchmarks require a completed
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
| `knowledge_mapping.py` | Question generation and record normalization |
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
primary graph. Source-file exclusion is defined by prepared file identity; it
does not by itself guarantee that linked passages have different article titles.

## Structured generation contracts

`core/structured_outputs.py` defines schemas for index question generation and
candidate selection. Requests use strict JSON-schema response formats through
the common gateway. The ranking schema requests the configured number of IDs
from the supplied candidate pool. The client decodes the returned JSON without
additional local schema or ranking-uniqueness validation; actual decoding and
consumer errors propagate. Initial question rewriting and document-conditioned
refinement, including their schemas, prompts, length gates and configuration
fields, have been removed.

`prehop-json-schema-v3` does not emit `uniqueItems`. Materialized schemas and
existing provenance digests retain their recorded identities; the historical
validation-contract label in the digest is not an active local validator.
The controlled format-retry profile retries the same
request under one shared maximum of five wire attempts, including transport
retries. Discarded responses and available usage remain recorded; retries do
not select among valid outputs by quality. Final synthesis remains text output.

## Prehop query path and branches

The primary settings use depth one, full NEXT/HOP expansion, Q−/body/Q+ search,
all-start HOP activation, body-only semantic scoring, unfiltered stored links,
and final LLM evidence selection.
Prehop uses the original query on every enabled channel at every input length.
Retrieval executes once. Retrieved evidence does not generate further questions
or trigger re-search. This query change reuses the existing index; construction
identity hashes cover index prompts and schemas independently of query prompts.

1. Search each enabled representation with vector and full-text search.
2. Merge representation hits into owner passages while retaining question IDs.
3. Expand stored NEXT and activated HOP links from the starting-passage pool.
4. Score the complete candidate union and select up to 12 passages by LLM.
5. Synthesize a short answer from selected evidence, or return the fixed
   insufficient-evidence response for empty context.

`retrieval/hybrid.py` sorts vector and lexical results independently by raw
score and stable identity, then combines reciprocal ranks `1 / (rank + 1)`.
Raw scores are not mixed across modalities. `retrieval/retrieve.py` sends the
original query once to each enabled role (body, Q−, and Q+), using the same
query embedding. Body and sentence hits are fused before owner aggregation.
The complete owner union becomes the base candidate pool.
Each representation retains at most top-k owners with candidate multiplier one.
Question searches allocate three times the owner budget before collapsing
individual questions to their owners.

`retrieval/traversal.py` walks NEXT in both directions and HOP only in its
stored direction. Under the default `HOP_SEED_POLICY=all`, every retrieved starting passage
exposes its HOP links. The historical `qplus` policy restricts starts to Q+ matches. Owner activation exposes all provenance
on that passage; `RAG_QPLUS_HOP_ACTIVATION=exact` restricts it to matched Q+ IDs.
Graph-discovered targets are not expanded again within the one-step pass.

A graph-only NEXT target inherits its source's total representation score;
a HOP target also inherits its source's total score under `all` (the historical `qplus` policy uses its Q+ score). Both use path decay 0.5. Direct
candidates retain their direct score when also reached by a graph path.
`retrieval/scoring.py` uses query-to-body similarity for all candidates by default
(`HOP_SEMANTIC_VARIANT=body_only`). The historical `body_bridge_min` option
uses the minimum of body and best source-Q+ similarity for HOP candidates. Semantic and representation orders are fused by reciprocal rank.
The default LLM selector receives all candidates as numbered passages, without
gold labels, retrieval scores, or path metadata.

Depth zero disables graph expansion while preserving passage selection.
Experimental channel, edge, reciprocal-filter, semantic-scoring, and
linked-continuation switches remain distinct from the primary defaults.

The primary component launcher supports `--expansion next_only`, `hop_only`,
and `none`. It retains the reference run’s activation and scoring policies;
for historical runs these are Q+-owner activation and bridge scoring. With `none`, traversal returns no neighbors
before opening a graph session. Every condition retains frozen initial
candidates, primary candidate scoring, and the common LLM selector. The index
is unchanged; each condition has a distinct run-local component identity.

## Explicit representation ablations

`core/prehop_ablation.py` describes `question_full`, `question_body`,
`body_body`, and `body_full`. All use original queries, all-seed HOP activation, body-only
semantic scoring, depth one, and the existing final selector. Under all-seed
activation, HOP targets inherit the source's total representation score.
These settings are not the historical primary benchmark baseline.

Both question-based linking conditions can read an existing compatible question graph through
`--reuse-existing-index`. They retain its namespace and exact index-stat
identity; writes and HOP rebuilds are disallowed for reused indexes. The body-based linking conditions share a
separate namespace and the frozen reference's per-passage degree budget.
`scripts/clone_prehop_body.py` can copy body properties and Document/Chunk,
CONTAINS, and NEXT structure without copying questions or HOP edges, then build
body links. Clone-only costs are not cold indexing costs. `body_full` uses frozen
multi-channel inputs from the question index with the body-linked graph; the
launcher requires benchmark mode and explicit frozen inputs for this profile.
Question representations supply initial retrieval but do not construct body links.

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
neither novelty nor a retrieval or cost advantage. Large edge groups use
`models/hoprag/exact_edges.py` to score all candidate pairs in bounded blocks
and retain native selection rules. The adapter creates the native-dtype answer
vector array once per edge group and reuses it across pending blocks.
`RAG_HOP_EDGE_BLOCK_SIZE` defaults to 128; only the current block of pair scores
is allocated. Sparse weights, same-node exclusion, tie ordering and final
sort/deduplication remain unchanged. No complete question cross join is built.
The older approximate dense-top-k helper
remains unused. Small groups use the native constructor.

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
External compatibility aliases are configured in native child processes.

## Evaluation output contract

`cli/benchmark.py` records official MultiHop-RAG retrieval and QA measures,
with legacy dataset metrics retained in code. The HotpotQA adapter projects complete returned corpus sentences to title/index pairs via
`utils/hotpotqa.py`, then applies Answer, Supporting Fact, and Joint scoring
rules. The projection is gold-independent and shared across systems. Official
scorer parity is tested; completed reduced-corpus experiments remain outstanding.
Normalized/fuzzy fact recall is diagnostic and differs from the manuscript's
literal exact-fact recall. Missing metric applicability is `-1`; evaluated
nonmatches are zero. Terminal failures receive zero primary quality scores and
remain visible in failure counts. Actual execution failures remain recorded.

The default checkpoint interval is ten completed queries. Resume reads the
existing result without configuration or identity validation. It runs missing IDs only, preserving both
successful and terminal-error rows. A resumed batch is not an uninterrupted
throughput measurement. The representation-ablation launcher itself does not
provide resume.

`record_paper_completion.py` records finished execution without final paper-policy
validation. It emits the legacy `admitted` receipt label with
`verification=disabled_by_user`; it does not recalculate metrics or certify
publication eligibility. Additional runtime and index verification gates are removed. Optional
analysis tools are not automatic completion gates. The synchronous benchmark
entrypoint finishes after query evaluation and resource cleanup; it does not
submit or reconcile an asynchronous Batch judge job.

## Complete-index reuse for final benchmarks

`core/index_reuse.py` supports links from full-index supervisor completions
(version 2) and the separate legacy one-query matrix protocol (version 1).
Both retain original index statistics, corpus identity, method policy, and
measured construction costs. File-backed methods receive query
copies without a byte-equality check; service-backed methods retain their source namespace. Copy preparation
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
candidates and answers. Tracing is enabled by default;
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

## Research diagrams

The method is illustrated by three editable schematics:
[overview](../fig/prehop_paper_overview.svg),
[query retrieval](../fig/prehop_paper_retrieval.svg), and
[passage-link example](../fig/prehop_paper_links.svg).
They describe the current single-pass paper design. HOP links come from
question matching; NEXT links come from within-source passage order. Original-
query retrieval and stored-link traversal contribute to a deduplicated candidate
set before selection. There is no input-length branch or query rewriting. Representation-ablation conditions
use the separate controls in [the ablation specification](PAPER_ABLATION_DESIGN.md).
Unreferenced legacy SVG/drawio schematics are retained under `fig/archive/`;
they document superseded designs and must not be used as current paper figures.
Result graphs are generated from completed evidence registered in
[RESULTS](RESULTS.md); they do not define runtime behavior.
Export and target-display checks are tracked in [the manuscript checklist](PAPER_CHECKLIST.md#rendering-and-cross-references).

### Direct retrieval and stored-link traversal

The query diagram distinguishes two routes into one candidate set. Manuscript
diagrams currently label the historical evaluated expansion configuration; that activation
restriction does not describe the updated runtime default. The original
query searches body, Q−, and Q+ representations; question hits map to owner
passages before rank fusion. All direct candidates remain in the pool. In the
current default, all retrieved starting passages expose stored outgoing HOP links;
NEXT supplies previous and next passages from retrieved starts. Both edge types
are written during indexing. Query-time traversal reads destinations, without
new link construction. Merge candidates by passage identity before final
selection; a passage can occur on both routes. Candidate membership is distinct
from inclusion in the final answer evidence. The timing ablation deliberately
changes when HOP destinations are resolved and is not the primary query flow.

## Controlled link-experiment modules

`models/prehop/ablation_inputs.py` provides explicit ablation-only frozen direct
inputs; primary retrieval does not consult them. representation ablation downstream measurements
carry their restricted latency scope. `models/prehop/connection_timing.py`
resolves Q+ destinations through the shared native matching wave and stores
experiment-scoped HOP_TIMING relationships in Neo4j. Both timing arms hydrate
identical destination fields. A pointer JSON records experiment identity and
preparation costs, not graph destinations.

`scripts/analyze_evidence_connections.py` reads gold only after construction and
measures co-evidence reachability plus a degree-matched random null. It never
writes gold-driven links. `scripts/ablation_statistics.py` resamples original
question clusters, retaining duplicate release occurrences. The task graph and
measurement constraints are owned by [PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md).


## Post-hoc connection analysis

`scripts/analyze_evidence_connections.py` reads a frozen graph and produces
`co-evidence-connectivity-v3` artifacts. It does not change retrieval or graph
construction. Per-query records retain original question IDs, relation-specific
scores, all random-graph realizations, and observed-minus-random differences.
Summaries expose released-row and original-question macro means with separate
question-cluster intervals. Random-graph variability occupies separate fields.

`scripts/ablation_statistics.py` pairs occurrence IDs before resampling original
questions. For version-3 connectivity inputs it discovers the complete metric
list, including signed differences, from `comparison_metrics`.
`scripts/analyze_ablation_links.py` uses the same inference function for
query-conditioned utility, with its successful-query denominator retained.
The scientific definitions belong in [PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md).

`scripts/prepare_reference_timing.py` reads original full-run traces to recover
actual connection starts, exclusions, settings and historical destinations.
`scripts/prehop_connection_timing.py --reference-inputs` measures both connection
arms on those recorded inputs using the prepared Neo4j store. It performs no
initial retrieval or answer generation. The original controlled replay remains
separate because its starts and settings can differ from the primary reference.

`scripts/estimate_connection_total.py` adds each query's online-minus-stored
connection delta to its existing measured full-query latency. It writes a new
`estimated_end_to_end` artifact, leaving the baseline untouched. Per-query
reasons explain unavailable estimates; paired-subset summaries remain distinct
from full-population summaries. Connection measurements retain the
`fixed_start_connection_replay` scope. These are analysis outputs, not new
production retrieval pipelines or runtime approval gates.


### Reference-specific expansion analysis

The experiment planner schedules NEXT-only, HOP-only and no-expansion controls
against each recorded reference. Updated-default MultiHop-RAG controls use a
separate task family and artifact namespace from historical controls. The
factorial analyzer reads the four saved conditions and computes paired
conditional effects and the additive interaction; it performs no retrieval.
Independently launched reference processes can be adopted into the same two-job
accounting without restarting them. Historical results keep their executed policy.
