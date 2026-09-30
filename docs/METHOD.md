# Method and implementation

For readers studying the retrieval method or changing its implementation.
Start with the [README](../README.md) to run it; use
[Reproducing](REPRODUCING.md) for experiment controls and evaluation, and
[Setup](SETUP.md) for infrastructure and configuration. Exact defaults, native
budgets and upstream revisions belong to the [strategy registry](../core/strategy_registry.py)
and [configuration](../core/config.py), rather than duplicated tables here.

## Code ownership

| Entry or module | Responsibility |
| --- | --- |
| `main.py`, `cli/index.py`, `cli/benchmark.py` | CLI dispatch, indexing coordination, query execution and adapter lifetime |
| `core/benchmark_evaluation.py`, `core/benchmark_checkpoint.py` | Metric aggregation, completion labels, persistence and resume |
| `models/prehop/graphrag.py` | Index/retrieval mixins and `run_workflow` |
| `models/prehop/indexing/`, `models/prehop/retrieval/` | Prehop construction and query algorithms |
| `models/naive/`, `models/hoprag/`, `models/ms_graphrag/` | Naive implementation and pinned native adapters |
| `models/external_research/`, `models/official_baseline_runtime.py` | Isolated native drivers and worker transport |
| `cli/synthesis.py`, `core/synthesis_replay.py` | Common reader over saved passages |
| `utils/metrics.py`, `utils/hotpotqa.py`, `utils/official_results.py` | Question scoring, support projection and official reports |
| `scripts/campaign_runtime.py`, `scripts/runner_environment.py` | Owned processes, logs, interpreter and environment selection |

## Prehop index construction

`indexing/chunking.py` accepts an optional first-line `Title:` and `--- Page N ---`
markers. Without markers the body is one page. Prehop and Naive split within
pages into six-sentence windows, retaining the final partial window and pipe
delimited text. External methods retain native indexing units.

The default question schema generates up to three Q− and three Q+ strings per
chunk. Q− asks about information inside the chunk; Q+ requests outside
information. Empty lists are valid. Decoding strips question strings, removes
duplicates and source-relative wording; incompatible values can fail in the
consumer. `knowledge_mapping.py` owns generation and normalization.

`embedding.py` stores title-scoped document embeddings for bodies and question
representations. Q+ also gets an instructed query embedding for outgoing links.
Cache keys distinguish model/revision, endpoint, role, instruction, dimensions
and normalized input. Cold target launchers disable generation/embedding caches
and allocate new storage.

`graph_writer.py` replaces a document's Document/Chunk/question subgraph in one
transaction and writes ordered NEXT edges separately. Parsing uses spawn-based
processes; a bounded document window and chunk producers limit resident work.
Graph writes bound both document count and logical parameter bytes. Only
transaction-memory errors split a failed group; successful prefixes are not
replayed. Exhausted failures prevent a complete index.

After corpus flushing, `hop_edges.py` searches each Q+ against Q− from another
source file. Its ANN pool includes the source's own channel count plus a
foreign slot. The best eligible returned Q− identifies its owner chunk; no
second body lookup is needed. This is an ANN-selected connection, not a proven
answer, exact global nearest neighbor or completed reasoning path. Prepared
file identity defines source exclusion; distinct files can share a title.

Questions selecting the same source/destination chunks merge into one
`HOP_ANSWER` edge with question provenance. `ANSWERED_BY` and `SUPPORTED_BY`
retain question-to-evidence paths. Both question roles are indexed in every
retained condition; body-only search changes query-time channels, not the index.
Reciprocal-link metadata remains materialized to preserve the published index
construction and timing conditions; query expansion does not filter by reciprocity.

## Prehop query path

The original question searches body, Q− and Q+ once. There is no question
rewriting, document-conditioned refinement or recursive search in the default
method. The complete union of directly retrieved owner chunks supplies starts:

1. `hybrid.py` orders vector and lexical hits separately and combines reciprocal
   ranks `1 / (rank + 1)`, preserving stable identity tie breaks.
2. `retrieve.py` merges role hits into owner chunks and retains question IDs.
   Enabled roles share the query embedding. Question search budgets account for
   multiple questions per owner; role budgets do not cap the total owner union.
3. `traversal.py` expands NEXT in both directions and stored HOP in its directed
   direction from every start. Graph-discovered chunks are not re-expanded in
   the default one-step traversal.
4. Graph-only candidates inherit the strongest incoming starting score with
   path decay. Direct candidates keep their direct score. `scoring.py` combines
   query-to-body similarity and representation ranks, then invokes the LLM
   selector on the full union.
5. The selected passages supply one final-answer request. Empty context returns
   the fixed insufficient-evidence response.

The selector receives opaque candidate IDs, titles, metadata and passage text,
without gold labels, numeric retrieval scores or path metadata. It requests up
to the final passage budget. Unknown returned IDs fail lookup; a short list is
filled from the existing order. The adapter then creates unique source records.
No additional local uniqueness or ranking-length validator retries the model.
The runtime and fixed-candidate comparisons share ordering/completion code.

The paper compares this expansion with deeper original-query retrieval in the
same index under matched selector input token budgets. Further controls vary
body-only versus full-channel search, HOP/NEXT expansion, link construction,
and LLM versus deterministic selection. Historical `body_bridge_min`
scoring remains available for the reported search-channel comparison. Saved
results retain their recorded policy and are not relabelled with current defaults.

## Generation and reader boundaries

`core/structured_outputs.py` owns requested question/selection schemas and their
hashes. The client requests strict JSON-schema output, decodes JSON and passes
it to consumers. Transport and JSON-decoding failures share one wire-attempt
budget with unchanged requests and SDK retries disabled. Decodable duplicate
properties, nonfinite values or schema mismatches do not trigger a new local
validation loop. Consumer errors propagate; valid outputs are not resampled
for quality. Discarded response/usage evidence is retained when available.

`utils/prompts/prehop_answer.py` owns Prehop's versioned evidence-checking system
instructions and question-first user message. The reader checks decisive facts,
connects them to the question and ends with `Final Answer:`; insufficient
support calls for abstention. Whole passages are fitted in rank order for the
native Prehop answer call. Effective output settings come from
`core/generation_profiles.py`. Index identity excludes answer-only prompts.

Naive uses the original shared prompt; external methods retain native readers.
Differences in their answer scores therefore do not isolate retrieval quality.
The [common-reader experiment](REPRODUCING.md#compare-a-common-reader-over-saved-evidence)
preserves every exposed returned passage without native Prehop's context fitting.
It is a separate measurement, not native-reader reproduction.

## Native method boundaries

Adapters stage the common corpus, configure transport, preserve source aliases
and observe native responses. Pinned upstream source is unchanged. Native
retrieval, prompts, local models and budgets remain method-specific; equal
rank cutoffs do not mean equal passage sizes or context amounts.

- HopRAG stages the complete corpus as one edge group, never query-specific gold
  contexts. Its native question-pair scoring uses vector dot products and
  keyword Jaccard. `exact_edges.py` scores large groups in bounded exhaustive
  blocks while retaining native dtype, exclusion, ties, trimming and deduplication;
  small groups use the native implementation. There is no dense-only prefilter.
  Represented and omitted source IDs are recorded when native generation omits
  documents with empty question lists. Native-dtype answer arrays are reused per
  group; block size controls memory, not candidate selection.
- MS GraphRAG retains native extraction, communities and LocalSearch. Citation
  decoding preserves all source aliases when native content-derived IDs merge
  documents; it does not rewrite parquet membership or answers.
- LightRAG retains native insertion, mix retrieval and document statuses. Resume
  inserts absent sources and explicitly retries existing failed sources once per
  invocation. Status is checked by source ID, not a reused track ID; repeated
  native failure remains an error. LLM and embedding wrapper deadlines use the
  configured transport timeout.
- GFM-RAG retains native extraction, checkpoint retrieval and single-pass QA.
  Its local linker uses run-local mutable caches, preserving model snapshots.
  The QA transport timeout follows the common contract instead of the upstream
  call default; a client timeout does not prove server cancellation.
- LinearRAG retains local NER/MPNet and native QA. Concurrent questions may be
  collected into a bounded batch and passed to native `qa(questions)`; retrieval
  remains native/sequential and answer generation uses native workers. A batch
  failure reaches all members without an adapter retry.

HopRAG's declared JSON recovery parses valid plain/fenced JSON before the native
cleaner, without inventing facts or adding completion retries. GFM-RAG retains
native JSON/empty-list behavior; other adapters retain native parsing and record
responses or exceptions. Observation alone is not evidence of indexing success.
Missing usage and monetary costs remain unavailable. Fixture parity tests do
not certify every numerical boundary or the full adapted pipeline.

## Persistence and ownership

Native indexing branches share elapsed-time, capacity and failure recording in
`cli/index.py::_run_native_index`; actual native calls stay in adapters. Benchmark
cancellation and checkpoint failure cancel/drain query tasks before adapter
closure. Worker EOF, broken input pipes and absolute request timeout fail the
target, including when upstream logs continue arriving. CLI commands propagate
recorded execution failure as a nonzero exit status.

`core/benchmark_checkpoint.py` atomically publishes trace JSONL before the main
JSON commit point. Resume joins traces by query ID, restoring only committed
rows; interruption can leave extra uncommitted traces without misassociation.
Missing committed traces or mandatory write failures are errors. Derived-report
I/O failures are logged after saving resume evidence. Shared writers use
`utils/io.py::atomic_text_writer`.

Dispatch has no additional corpus/configuration/index-reuse approval gate.
Recorded provenance supports later interpretation; it does not guarantee that
reused files are semantically compatible. See [continuation](REPRODUCING.md#completion-and-continuation)
for user-facing resume and receipt rules.

## Prehop tracing

`models/prehop/tracing.py` records stage inputs/outputs, embeddings, Cypher,
candidates and answers in session-specific `data/traces/` directories. Ordered
`events.jsonl` entries refer to hashed compressed payloads; index/query artifacts
retain references. `RAG_PREHOP_TRACE` enables recording and
`RAG_PREHOP_TRACE_DIR` selects storage.

Credentials/HTTP headers are excluded and known secrets redacted, but payloads
contain source and model data. Directories use mode 0700 and files 0600. Inline
writes must not wait for the embedding waiters' executor. Storage errors
propagate. Trace I/O contributes to phase wall time; trace bytes are excluded
from retrieval-index capacity.

## Controlled analysis modules

`ablation_inputs.py` freezes direct candidates for the HOP/NEXT controls.
`ablation_statistics.py` pairs occurrence IDs before resampling original-question
clusters. Link usefulness retains its successful-query denominator, while the
shuffled-link comparison reads frozen candidates and sampling realizations.
`compare_link_representations.py` retains the five-condition appendix pilot;
its Q+ to passage search uses the shared ANN helper without replacing the main
question-link index. `analyze_evidence_accessibility.py` evaluates frozen graph
and deeper-direct pools, selection retention and reader-development exclusions.
Commands and the full experiment inventory are owned by [Reproducing](REPRODUCING.md).

Registry fields for retired experimental controls are fixed historical identity
values, not selectable methods. Question-role indexing, one-step depth, path
decay and reciprocal metadata materialization are fixed to the reported index
and method settings. Prompt and structured-schema identities retain historical bundle aliases
while the active templates and schemas are unchanged; editing a retained
prompt or schema produces a new identity. This preserves
existing index and cache identities without retaining experimental generators.
