# Architecture

This document describes implementation behavior. `core/config.py` resolves retrieval
and evaluation settings; `core/strategy_registry.py` owns shared method and transport
defaults, supported methods, primary order, upstream revisions, and benchmark
policies. `core/execution_profile.py` applies operational overrides at resolution
time; `core/inference_transport.py` resolves effective request and embedding settings.
Launch procedures are defined in
[THROUGHPUT_EXECUTION](THROUGHPUT_EXECUTION.md).

## Strategy dispatch and indexing branches

The registry execution order is Prehop, Naive RAG, HopRAG, MS GraphRAG, LightRAG,
GFM-RAG, and LinearRAG. Comparisons cover MultiHop-RAG and HotpotQA (HippoRAG
corpus). HotpotQA uses
the pinned HippoRAG v1 pooled retrieval corpus, with
source and sentence identities preserved. Preparation and evaluation are defined
in [HOTPOTQA](HOTPOTQA.md).
Removed methods do not supply primary comparison cells.

`cli/index.py::run_indexing` calls `_run_indexing_unlocked` without a preflight
or run lock. Run-scoped namespaces provide artifact isolation.
Native indexing branches share `_run_native_index` for elapsed time, capacity
and completion/failure artifacts; native calls and their parameters remain in
the method adapters.

| Method | Index construction | Query path |
|---|---|---|
| Prehop | Six-sentence passages, Q−/Q+ nodes, NEXT and directed HOP links | Representation search, activated links, LLM evidence selection |
| Naive RAG | The same six-sentence passages and body embeddings | Dense body search and the original answer synthesis prompt |
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
preserving original article titles and sentence indices through preparation.
Provenance records the selected index and source identities; dispatch does not
verify snapshot or corpus equality. Scientific comparisons still require the
declared inputs and a completed index.
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
elsewhere. Empty lists are valid. The decoder strips question strings and removes
duplicates and source-relative wording. It has no additional blank-string or
type-validation gate; incompatible values can fail during native string use.
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

`prehop-json-schema-v3` does not emit `uniqueItems`. Schema bundle hashes cover
the requested schemas; existing artifacts retain their recorded identities.
The `prehop-native-json-retry-v3` profile retries JSON decoding failures with the
same request under one shared retry budget, including transport retries. The
default is five wire attempts. Question generation and candidate selection call
this client directly. Consumer errors propagate without starting another retry
loop or rewriting the prompt. Decodable duplicate properties, nonfinite values
and schema-mismatched values do not trigger local validation retries. Discarded
responses and available usage remain recorded; retries do not select among valid
outputs by quality. Final synthesis remains text output. Transport settings are
owned by the [common inference gateway configuration](RUNTIME_REQUIREMENTS.md#common-inference-gateway).

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

Prehop's answer instructions live in `utils/prompts/prehop_answer.py`
(`prehop-evidence-check-v3`). A system message asks the reader to check up to
three decisive facts, connect them to the question, and end with a labelled
`Final Answer:`. The user message puts the original question before the
selected passages. Instructions preserve entity names, time periods and units;
binary answers use bare `Yes` or `No`. Missing essential evidence calls for
abstention. The reader uses one non-thinking call at temperature zero with a
256-token output cap. Context fitting counts both messages and accepts whole
passages in rank order, including the client's 1,024-token completion reserve.
Naive RAG and generic adapters retain the
original prompt in `utils/prompts/shared.py`; native external readers retain
their own procedures. Answer scores across these readers do not isolate
retrieval quality.

The materialized messages, output budget and prompt version contribute to query identity.
Index identity excludes answer prompts. A synthesis-only replay can therefore
reuse recorded selected passages and their retrieval scores, provided their
text and order are preserved. Its generation time is a separate measurement
from the original full query.

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
The default LLM selector receives the complete candidate union as passages with
opaque IDs, titles and available source metadata, without gold labels, retrieval
scores or path metadata. The requested ranking length is the smaller of the
candidate count and final top-k. Returned IDs are mapped directly to candidates;
unknown IDs fail at lookup. A short ranking is filled from the existing candidate
order until top-k, then the query adapter builds unique source records. There
is no separate local uniqueness or length validation of the model's ranking.
The runtime and recorded-input selection comparison share candidate ordering and
short-ranking completion in `retrieval/scoring.py` to keep those behaviors aligned.

Depth zero disables graph expansion while preserving passage selection.
Experimental channel, edge, reciprocal-filter, semantic-scoring, and
linked-continuation switches remain distinct from the primary defaults.

The primary component launcher supports `--expansion next_only`, `hop_only`,
and `none`. It retains the reference run’s activation and scoring policies;
for historical runs these can be Q+-owner activation and bridge scoring.
With `none`, traversal returns no neighbors before opening a graph session.
Every condition retains frozen initial candidates, reference scoring rules,
and the common LLM selector. The index
is unchanged; each condition has a distinct run-local component identity.

## Explicit representation ablations

`core/prehop_ablation.py` describes `question_full`, `question_body`,
`body_body`, and `body_full`. All use original queries, all-seed HOP activation, body-only
semantic scoring, depth one, and the existing final selector. Under all-seed
activation, HOP targets inherit the source's total representation score.
The question-link/multi-channel profile has the current primary retrieval
settings. A matching configuration does not make separately executed source
artifacts interchangeable. Older Q+-restricted runs retain their recorded policy.

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
[experiment reproduction](REPRODUCING.md#compare-link-and-search-representations)
for commands, controls and comparison scope.

## HopRAG indexing

HopRAG is part of the current primary comparison set. See
[the runtime guide](RUNTIME_REQUIREMENTS.md#hoprag-runtime) for setup.
`models/hoprag/native_runtime.py` loads the pinned prepared installation.
`models/hoprag/runtime_paths.py` shares its location across the coordinator, POS
worker and provenance collection. The adapter stages the complete corpus as one
edge-construction group; queries and gold evidence do not determine edge groups.
Native generation may omit a document when either question list is empty;
represented and omitted source IDs remain separately recorded.

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
Small groups use the native constructor.

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
one OpenAI-compatible LiteLLM base. Shell launchers reject legacy provider
input variables. Paper mode omits LLM seeds; evaluation/sampling seed 42 is separate.
Existing result artifacts retain their historical generation settings.

The shared embedding decoder orders returned vectors by response index without
an additional count, dimension, finite-value or uniqueness validator. Consumers
retain their own shape requirements and errors. Context-size or HTTP 413 errors
allow order-preserving bisection; unrelated errors and failing singletons
propagate. External compatibility aliases are configured in native child processes.

## Evaluation output contract

`cli/benchmark.py` owns query dispatch and adapter lifetime.
`core/benchmark_evaluation.py` owns scope, aggregation and completion labels;
`core/benchmark_checkpoint.py` owns persistence and resume. Cancellation or
checkpoint failure cancels and drains query tasks before closing the adapter.
External worker EOF, broken pipes and request timeouts fail the target. CLI
commands propagate a recorded failed execution as a nonzero exit status.

`cli/benchmark.py` records official MultiHop-RAG retrieval and QA measures,
with legacy dataset metrics retained in code. The HotpotQA adapter projects complete returned corpus sentences to title/index pairs via
`utils/hotpotqa.py`, then applies Answer, Supporting Fact, and Joint scoring
rules. The projection is gold-independent and shared across systems. Official
scorer parity is tested. Saved-output common retrieval analysis uses
`scripts/evaluate_saved_retrieval.py`; its HotpotQA identity projection and
rank measures are defined in [HOTPOTQA](HOTPOTQA.md#common-passage-retrieval-metrics).
Normalized/fuzzy and literal exact-fact recall are separate diagnostics; neither
redefines the official benchmark metrics. Missing metric applicability is `-1`; evaluated
nonmatches are zero. Terminal failures receive zero primary quality scores and
remain visible in failure counts. Actual execution failures remain recorded.

`utils/official_results.py` writes dataset-specific `<stem>.official.json`
alongside each benchmark result. MultiHop-RAG stores Hits@4, Hits@10, MRR@10,
MAP@10 and official QA precision/recall/F1/accuracy. Its QA values are the same
per-question success rate under the upstream token-intersection rule. QA
includes null questions and also reports question-type and non-null groups;
retrieval excludes null questions. HotpotQA stores the twelve official Answer,
Supporting Fact and Joint EM/F1/precision/recall values. Repeated original IDs
retain distinct release-occurrence IDs and their original denominator.

`<stem>.diagnostics.json` holds auxiliary measures, including MultiHop-RAG
normalized answer EM/F1, refusal and evidence-coverage diagnostics. Adapted
HotpotQA passage-ranking measures belong here when present in the source;
they are not official HotpotQA metrics. Values in both files remain unscaled.
The official report records the pinned upstream evaluator revision and the
shared final-answer extraction and sentence-projection adapters. It preserves
the source status and expected/evaluated counts, so a checkpoint is not
presented as a completed benchmark.

To export the same two files from an existing result without changing it or
calling a model, run:

```bash
python -m scripts.export_official_results path/to/result.json --output-dir path/to/reports
```

Offline exports bind the original result with its SHA256. Historical main
artifacts retain their existing fields; official and auxiliary sidecars provide
separate reporting contracts. The same command accepts multiple explicit final
result paths and writes separate dataset comparison tables with only official
metrics and per-method official files; auxiliary sidecars remain with their
original results. See
[saved-result evaluation](REPRODUCING.md#dataset-specific-official-results).

The default checkpoint interval is ten completed queries. Resume restores saved
rows and their traces in the current input order without configuration or
identity validation. It runs missing IDs only, preserving both successful and
terminal-error rows, prior provenance and accumulated segment timing. A resumed
batch is not an uninterrupted throughput measurement. The representation-ablation
launcher itself does not provide resume.

Each checkpoint atomically writes the trace JSONL first and then the main
result JSON as its commit point. Resume joins traces by query ID, so interruption
between those writes can leave extra uncommitted traces without misassigning
them to completed queries. Missing committed traces or mandatory write failures
are errors. Derived report I/O failures are logged after saving resume evidence.
Shared JSON and text writers use `utils/io.py::atomic_text_writer`.

`record_paper_completion.py` records finished execution without final paper-policy
validation. Receipt fields, accepted result states and resume behavior are
defined in [completion and continuation](THROUGHPUT_EXECUTION.md#completion-and-continuation).
Optional analysis tools are not automatic completion gates. The synchronous benchmark
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
Clients call the configured gateway directly. Profiles set per-client
generation and embedding limits and per-target benchmark concurrency. Campaigns
run targets sequentially by default; separate processes do not share a request
semaphore. These limits do not establish GPU saturation or exclusive remote
resources.

Cost definitions and storage measurement boundaries belong in
[the measurement protocol](THROUGHPUT_EXECUTION.md#final-tables-and-measurement-definitions).

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

`scripts/campaign_runtime.py` owns shared process identities, resource locks,
atomic status writes and redacted child-log draining. The index matrix, link
experiments and full campaign use these helpers. `scripts/runner_environment.py`
owns environment loading, interpreter selection and child-environment filtering.


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

## Controlled link-experiment modules

`models/prehop/ablation_inputs.py` provides explicit ablation-only frozen direct
inputs; primary retrieval does not consult them. Representation-ablation downstream measurements
carry their restricted latency scope. `models/prehop/connection_timing.py`
resolves Q+ destinations through the shared native matching wave and stores
experiment-scoped HOP_TIMING relationships in Neo4j. Both timing arms hydrate
identical destination fields. A pointer JSON records experiment identity and
preparation costs, not graph destinations.

`scripts/analyze_evidence_connections.py` reads gold only after construction and
measures co-evidence reachability plus a degree-matched random null. It never
writes gold-driven links. `scripts/ablation_statistics.py` resamples original
question clusters, retaining duplicate release occurrences. Public experiment
commands and measurement boundaries are in [Reproducing experiments](REPRODUCING.md).


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
The [analysis guide](REPRODUCING.md#additional-analyses-and-release-scope)
distinguishes observed link use from gold-evidence connectivity.

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

The experiment planner schedules Direct + NEXT, Direct + HOP and Direct-only controls
against each recorded reference. Updated-default MultiHop-RAG controls use a
separate task family and artifact namespace from historical controls. The
factorial analyzer reads the four saved conditions and computes paired
conditional effects and the additive interaction; it performs no retrieval.
Independently launched reference processes can be adopted into the same two-job
accounting without restarting them. Historical results keep their executed policy.

## Fixed-candidate selection analysis

`scripts/compare_prehop_selectors.py` reads a complete, failure-free reference,
its query annotations and hashed scoring/selection traces. The recorded LLM
selection is compared with the saved fused order and representation-rank order
over the same candidate pool. An optional raw hybrid-score condition reads saved
channel records or explicitly replays search against the original Neo4j index
using recorded query embeddings. Replay checks owner order before propagating
scores along the recorded graph paths; it makes no generation or embedding calls.

Alternatives take the recorded top-k before source deduplication, so a returned
list can be shorter than that budget. Outputs contain selected sources, common
retrieval measures, HotpotQA support scores and paired original-question cluster
intervals. They supply no generated answers or measured query latency. Input
consistency checks belong to this optional analysis, not benchmark dispatch.
See [selector reproduction](REPRODUCING.md#fixed-candidate-final-selection) for
commands, supported references and output filenames.

## Answer-only replay

`core/synthesis_replay.py` supplies saved evidence through `TraceContextProvider`.
`cli/synthesis.py` consumes that hook to regenerate only final answers, with
sequential dataset/system groups and concurrent questions within a group.
The `synthesize` mode returns before any benchmark or indexing dispatch.
The [replay workflow](SYNTHESIS_REPLAY.md) owns input preparation, resume identity
and request handling.
