# Method and implementation

This guide describes standard Prehop, its implementation, and the controlled
comparisons of graph expansion, direct retrieval, and candidate admission.
Start with the [README](../README.md) to run it; use
[Reproducing](REPRODUCING.md) for experiment controls and evaluation, and
[Setup](SETUP.md) for infrastructure and configuration. Exact defaults belong to
the [strategy registry](../core/strategy_registry.py) and
[configuration](../core/config.py), rather than duplicated tables here.

## Code ownership

| Entry or module | Responsibility |
| --- | --- |
| `main.py`, `cli/index.py`, `cli/benchmark.py` | CLI dispatch, indexing coordination, query execution and engine lifetime |
| `core/benchmark_evaluation.py`, `core/benchmark_checkpoint.py` | Metric aggregation, completion labels, persistence and resume |
| `models/prehop/graphrag.py` | Index/retrieval mixins and `run_workflow` |
| `models/prehop/indexing/`, `models/prehop/retrieval/` | Prehop construction and query algorithms |
| `models/naive/` | Dense baseline over the same fixed windows, embeddings and Neo4j vector index |
| `cli/synthesis.py`, `core/synthesis_replay.py` | Shared answer generation over saved passages |
| `utils/metrics.py`, `utils/hotpotqa.py`, `utils/official_results.py` | Question scoring, support projection and official reports |
| `scripts/campaign_runtime.py`, `scripts/runner_environment.py` | Owned processes, logs, interpreter and environment selection |

## Prehop index construction

`indexing/chunking.py` accepts an optional first-line `Title:` and `--- Page N ---`
markers. Without markers the body is one page. Prehop and Naive split within
pages into six-sentence windows, retaining the final partial window and pipe
delimited text.

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

The original question searches body, Q− and Q+ once. The complete union of
directly retrieved passages forms the initial candidate pool. Prehop expands
from every initial candidate without iterative LLM-guided traversal:

1. `hybrid.py` orders vector and full-text hits separately and combines reciprocal
   ranks `1 / (rank + 1)` for zero-based `rank`, preserving stable identity tie breaks.
2. `retrieve.py` merges role hits into owner chunks and retains question IDs.
   Enabled roles share the query embedding. Question search budgets account for
   multiple questions per owner; role budgets do not cap the total owner union.
3. `traversal.py` expands NEXT in both directions and stored HOP in its directed
   direction from every start. Graph-discovered chunks are not re-expanded in
   the default one-step traversal.
4. Candidates added by expansion inherit the strongest incoming starting score with
   path decay. Direct candidates keep their direct score. `scoring.py` combines
   query-to-body similarity and representation ranks, then invokes the LLM
   reranker on the full union.
5. The selected passages supply one final-answer request. Empty context returns
   the fixed insufficient-evidence response.

The reranker receives opaque candidate IDs, titles, metadata and passage text,
without gold labels, numeric retrieval scores or path metadata. It requests up
to the final passage budget. Unknown returned IDs fail lookup; a short list is
filled from the existing order. The adapter then creates unique source records,
so candidate IDs repeated by the model are deduplicated after the fill and fewer
than 12 passages can be returned.
No additional local uniqueness or ranking-length validator retries the model.
The runtime and fixed-candidate comparisons share ordering/completion code.

The historical `body_bridge_min` scoring variant remains an explicit
query-policy setting for reading archived results. Saved results retain their
recorded policy and are not relabelled with current defaults.

## Controlled retrieval comparisons

Standard Prehop submits its complete one-step pool to LLM reranking. The
comparisons below separately control initial candidates, input tokens, or
admitted counts. They are experimental conditions rather than changes to that
default query path. The [experiment guide](REPRODUCING.md#paper-experiment-inventory)
identifies the required saved artifacts and commands.

### One-step expansion and direct retrieval

The HOP/NEXT ablation retains the index and saved initial passages while varying
which edge types are expanded. The comparison of LLM reranking with score-based
ordering instead retains each complete candidate pool. Neither comparison
equalizes the input tokens across different expansion conditions.

The direct-retrieval comparison uses the original one-step reranking prompt's
token count as a per-query ceiling. Multi-channel direct retrieval increases
the body, Q−, and Q+ search limits while retaining the original initial union.
Body-only direct retrieval searches passage bodies and retains its original
12 body-channel passages. Both omit graph expansion and add an ordered prefix
of whole passages under the ceiling, using query-to-body cosine fused with their
respective representation scores. Counts include the prompt and chat template.
The final three-condition comparison evaluates candidate coverage, LLM
reranking, and common answer generation. The ceiling does not equalize realized
tokens, final answer-context length, or total retrieval cost. Body-only search
reuses the constructed corpus and index; it does not measure body-only indexing.

### Expansion depth and candidate admission

The depth analysis repeatedly expands the saved HOP/NEXT graph to construct
nested one-to-four-step neighborhoods. Candidate admission selects which of
these passages enter LLM reranking: all original initial passages remain,
followed by a whole-passage prefix under the original one-step token ceiling.
Within each neighborhood, rank-fusion and cosine admission share
passages, initial scores, query embeddings, and the ceiling. Rank fusion combines
query-to-body rank with representation rank; cosine admission orders by
query-to-body cosine alone. The name distinguishes the admission policy from the
body-similarity graph and from body-only direct retrieval. A newly reached passage inherits half the strongest
representation score among predecessors at the preceding distance. Scores of
previously reached passages remain fixed.

Coverage is evaluated before and after admission at every depth. Downstream
LLM reranking and QA compare the two admitted four-step sets from the original
question-link graph, with a common query-to-body cosine presentation order and
the same answer prompt. That order differs from the fused-score order of the
standard one-step query path, so contrasts against the one-step policies compare
runs with different presentation rules. This comparison measures an
admission-policy change at four steps, not an isolated depth effect. The runtime query path still
requires exactly one expansion step; the saved-graph analysis is separate.

### Graph construction and inherited scores

Question-link and body-similarity graphs share passages, initial candidates,
NEXT links, and each passage's outgoing HOP degree. Comparing both admission
policies in both four-step neighborhoods changes candidate membership and
inherited scores together. Equal outgoing degree does not equalize incoming
degree, neighborhood size, or search cost.

A separate control takes the intersection of the two neighborhoods and fixes
the number of admitted non-initial passages to the smallest available count
across the original four graph–policy conditions, bounded by the intersection's
non-initial passage count. All initial passages remain. It recomputes fusion
ranks over this common pool using each graph's inherited scores, with fixed query-to-body similarities
and tie rules. Scores retain their original full-graph paths, including
predecessors outside the intersection. Exchanging their shortest-distance and
starting-score components provides additional scalar score controls. These
experiments measure candidate recall at a common count; they do not match exact
tokens, evaluate QA, or decompose the original token-limited QA effect.

Source-level controls also distinguish passages included in a one-step pool
from all passages in the sources it reaches. The source oracle is unbudgeted;
the passage-selection controls retain the initial passages and apply the same
token-limited cosine-order rule within each source set. Coverage analyses
separately track newly recovered and lost annotations, initial-coverage groups,
edge-type reachability, the graph distance of gold evidence missing from the
one-step pool, and sensitivity to repeated fact annotations. Gold
annotations are used only to evaluate these constructed sets and define the
reported analysis groups.

<a id="generation-and-reader-boundaries"></a>

## Generation settings and answer generation

Documentation uses **LLM reranking** for ordering the candidate pool and
**answer generation** for producing the final answer. Existing code and saved
records also use `selector`, `reader` and `common-reader`; those identifiers
are retained for compatibility with commands, result files and recorded experiments.

`core/structured_outputs.py` owns the question-generation and reranking schemas and their
hashes. The client requests strict JSON-schema output, decodes JSON and passes
it to consumers. Transport and JSON-decoding failures share one wire-attempt
budget with unchanged requests and SDK retries disabled. Decodable duplicate
properties, nonfinite values or schema mismatches do not trigger a new local
validation loop. Consumer errors propagate; valid outputs are not resampled
for quality. Discarded response/usage evidence is retained when available.

`utils/prompts/prehop_answer.py` owns Prehop's versioned evidence-checking system
instructions and question-first user message. The answer generator checks decisive facts,
connects them to the question and ends with `Final Answer:`; insufficient
support calls for abstention. Whole passages are fitted in rank order for the
benchmark Prehop answer call. Effective output settings come from
`core/generation_profiles.py`. Index identity excludes answer-only prompts.

Naive uses the original shared prompt, so Prehop/Naive answer-score differences
do not isolate retrieval quality. The
[shared answer-generation experiment](REPRODUCING.md#compare-answer-generation-over-saved-evidence)
preserves every exposed returned passage without the benchmark Prehop answer
call's context fitting. It measures answer quality from the returned passages
under a common model and prompt, separately from each system's benchmark answer
pipeline. Equal rank cutoffs do not mean equal passage sizes or context amounts.

## Persistence and ownership

Benchmark cancellation and checkpoint failure cancel/drain query tasks before
engine closure. CLI commands propagate recorded execution failure as a nonzero
exit status.

`core/benchmark_checkpoint.py` atomically publishes trace JSONL before the main
JSON commit point. Resume joins traces by query ID, restoring only committed
rows; interruption can leave extra uncommitted traces without misassociation.
Missing committed traces or mandatory write failures are errors. Derived-report
I/O failures are logged after saving resume evidence. Shared writers use
`utils/io.py::atomic_text_writer`.

Resume uses the original corpus, configuration and index for that run. See
[continuation](REPRODUCING.md#completion-and-continuation) for checkpoint and
run-identity rules.

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
`analyze_evidence_accessibility.py` evaluates frozen graph and pools from direct
retrieval with more candidates, recorded search channels of newly supplied evidence,
retention after reranking, initial-coverage groups and prompt-development exclusions.
The depth, admission, graph-score, source-selection, and sensitivity analyses
reconstruct their specified conditions from saved graphs and experiment
artifacts. They are distinct from the standard benchmark command and the
one-step comparison modules above. Commands, artifact requirements, and the
full experiment inventory are owned by [Reproducing](REPRODUCING.md).

Registry fields for retired experimental controls are fixed historical identity
values, not selectable methods. Question-role indexing, one-step depth, path
decay and reciprocal metadata materialization are fixed to the reported index
and method settings. Prompt and structured-schema identities retain historical bundle aliases
while the active templates and schemas are unchanged; editing a retained
prompt or schema produces a new identity. This preserves
existing index and cache identities without retaining experimental generators.
