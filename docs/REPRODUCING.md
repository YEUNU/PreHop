# Reproducing and evaluating experiments

For researchers reproducing comparisons or interpreting saved results. Follow
the [README](../README.md) for installation, data preparation, the small live
smoke and full system runs. This guide owns data/evaluation definitions,
continuation, controlled experiments and measurement boundaries. Commands run
from the repository root in Bash using the selected `PYTHON_BIN`.

## Data and evaluation

| Dataset | Prepared corpus and queries | Evaluation population |
| --- | --- | --- |
| MultiHop-RAG | `data/multihoprag_corpus/`, `data/multihoprag_queries.json` | 609 documents; 2,556 QA questions, including 301 null questions; 2,255 evidence-bearing retrieval questions |
| HotpotQA (HippoRAG release) | `data/hotpotqa_corpus/`, `data/hotpotqa_queries.json` | 9,221 passages; 1,000 released occurrences representing 944 original questions |

MultiHop-RAG preparation downloads the upstream JSON, normalizes titles and
removes the fixed article boilerplate patterns defined in its script. It
rebuilds the prepared corpus directory. Prepare data once before a run and do
not rebuild a corpus in use. Manifests retain source/query IDs and content hashes.

### HotpotQA source and population

The source is [HippoRAG v1.0.0](https://github.com/OSU-NLP-Group/HippoRAG/tree/b144c46df14cabe5f5822d8caded4bec5f709461),
using its `data/hotpotqa.json` and `data/hotpotqa_corpus.json`. The pinned revision
and download URLs are owned by
[the preparation script](../scripts/datasets/prepare_hotpotqa_hipporag.py).
The [HippoRAG paper, Section 3.1](https://arxiv.org/html/2405.14831v1#S3.SS1)
describes pooling selected validation questions' supporting/distractor contexts.
Retrieval sees this common pool, not per-question gold contexts or all Wikipedia.
This is not the official HotpotQA fullwiki search setting.

Preserve every released row and its order. Prepared `_id` combines the original
ID and occurrence ordinal; `original_query_id` retains upstream identity.
Row-weighted scores preserve release comparability. Original-question macro
scores are sensitivity analyses; bootstrap intervals resample original-question
clusters with all their occurrences.

Preparation stores the pinned raw files in `data/hotpotqa_hipporag_raw/`, corpus
text and `sentences.sqlite3` in `data/hotpotqa_corpus/`, and provenance/population
metadata in `corpus_manifest.json`. Use the script's `--output` and
`--queries-output` for an additional copy with fresh destinations. Titles,
sentence text and order are preserved; source IDs derive deterministically from
titles. Gold annotations never repair corpus content. Missing supporting
sentences remain in the denominator and are recorded in `annotation_coverage`.
A complete run is labelled `released_benchmark`, not fullwiki completion.

### Official metrics and prediction adapters

MultiHop-RAG reports Hits@4/10, MRR@10, MAP@10 and its official token-intersection
QA success rule. QA includes null questions and reports question-type/non-null
subgroups. Successful null rows are excluded from retrieval; terminal failure
handling in legacy benchmark aggregates can change that denominator, so report
failed null rows explicitly. Official exports exclude null rows consistently.

HotpotQA reports Answer, Supporting Fact and Joint EM/F1/precision/recall through
`utils/hotpotqa.py`. The shared support adapter predicts all complete original
corpus sentences found in returned passages, mapped to original title/index
pairs. It never uses gold to choose support. Report mapping coverage and missing
identities. These are official formulas under the declared reduced-corpus
protocol, not leaderboard scores or native sentence-selection measurements.

Both datasets use shared final-answer extraction before official scoring.
`utils/official_results.py` records the evaluator revisions and these adapters.
Scores remain in native unscaled units. Auxiliary normalized/fuzzy answer and
passage-coverage diagnostics do not replace official metrics. Terminal failures
remain in applicable populations with zero quality; unknown usage, cost or judge
scores remain unavailable. LLM judging is optional/synchronous and not part of
the primary comparison; asynchronous Batch judging is unsupported.

### Common passage-retrieval metrics

The optional saved-output analysis applies Hits@4/10, MRR@10, the MultiHop-RAG
MAP@10 formula and distinct-gold Recall@10 to both datasets. MultiHop-RAG matches
facts case-sensitively after removing spaces/newlines. HotpotQA matches original
support title/sentence IDs through the same complete-sentence projection.
Unknown identities get no credit. Keep passage order and actual counts without
padding. These adapted ranking measures are not official HotpotQA metrics.

At rank r only newly recovered gold items receive credit. MAP sums their count
divided by r and normalizes by min(gold count, 10); Recall@10 divides distinct
recovered items by the full gold count. Hits is any-hit success; MRR uses the
first hit. The capped MAP denominator can yield values above one when there
are more than ten gold items: do not clip or replace the formula.

## Completion and continuation

The README target command allocates run-scoped storage and disables repository
generation/embedding caches. `--check` resolves launch settings without testing
runtime readiness. It skips indexing if that run's index-statistics file exists
and resumes missing query IDs if its result exists. File presence does not prove
compatibility. Reuse an ID only with the original corpus/settings/index; choose
a new ID for an independent experiment. No global graph clear is needed.

Resume preserves successful and terminal-error rows, their traces, prior
provenance and checkpointed segment times. It does not rerun failed query rows.
Missing committed traces or mandatory checkpoint writes are errors; see
[the persistence contract](METHOD.md#persistence-and-ownership). An index-only
completion or one-query smoke is not a full benchmark result.

`record_paper_completion.py` accepts completed result states and emits the legacy
`admitted` receipt with `verification=disabled_by_user`, preserving an existing
receipt. This status-based record does not recount questions, rescore answers or
certify research validity. Inspect population, failures, source index and
measurement scope. Source edits do not update already-loaded Python code;
new segments retain their actual provenance.

The smoke's `query.json` holds its answer/passages; `index_evidence.json` and
`evidence.json` bind execution sources. Full results hold `details`, including
answers, ordered `retrieved_sources`, scores and `index_manifest_stats_path`.
Retain the graph and trace references for the analyses below.

## Select a reference

The following examples assume the completed MultiHop-RAG target from the
README (`BENCH_RUN` is still exported). To analyze another saved run, assign
these three paths explicitly; no command selects the latest file:

```bash
export PREHOP_RESULT="$PWD/data/results/$BENCH_RUN-multihoprag/prehop/multihoprag/seed_42/prehop_multihoprag.json"
export PREHOP_QUERIES="$PWD/data/multihoprag_queries.json"
export PREHOP_INDEX="$PWD/data/index_stats/prehop_multihoprag_$BENCH_RUN-multihoprag.json"
```

For each experiment specify the hypothesis, matched query IDs, changed factor,
fixed factors and measures before execution. Gold is evaluation-only. The
[method](METHOD.md) describes primary behavior; the [runtime manifest](../configs/paper_runtime_requirements.json)
and registry own native model/revision/budget requirements. Use common declared
serving conditions for time comparisons. Preserve historical conditions instead
of relabelling them with current defaults.

## Compare HOP and NEXT expansion

This experiment changes expansion while keeping the reference graph, initial
retrieval, scoring and LLM selection settings fixed. Every condition retains
the direct passages:

| Condition | HOP | NEXT | `--expansion` |
| --- | --- | --- | --- |
| Prehop | On | On | Use the reference result |
| Direct-only | Off | Off | `none` |
| Direct + NEXT | Off | On | `next_only` |
| Direct + HOP | On | Off | `hop_only` |

After the reference finishes, prepare its saved initial candidates and run one
intervention. Preparation reads trace files; the worker performs expansion,
selection and answer generation:

```bash
"$PYTHON_BIN" scripts/run_primary_hop_ablation.py prepare \
  --reference "$PREHOP_RESULT" --queries "$PREHOP_QUERIES" \
  --expansion next_only --output data/results/reproduce-mhr-next
"$PYTHON_BIN" scripts/run_primary_hop_ablation.py worker \
  --reference "$PREHOP_RESULT" --queries "$PREHOP_QUERIES" \
  --output data/results/reproduce-mhr-next
```

Repeat preparation and the worker with `none` and `hop_only`, each in a new
output directory. The worker reads the expansion setting from the prepared
plan. Its result path ends in `prehop/multihoprag/seed_42/prehop_multihoprag.json`.
For HotpotQA, use that dataset's own reference and query file.

Compare common retrieval scores as described below. Intervention latency
excludes initial retrieval; it is not interchangeable with full-query latency.
Expansion changes the candidate pool while holding the final selection budget
fixed. Its effect is conditional on the remaining retrieval system.

## Compare link and search representations

| Index-time links | Query-time search | Profile |
| --- | --- | --- |
| Question-based | Body, Q−, Q+ | `question_full` (Prehop configuration) |
| Question-based | Body only | `question_body` |
| Body-based | Body only | `body_body` |
| Body-based | Body, Q−, Q+ | `body_full` |

Use the same Ours source in system and representation comparisons. A separately
executed `question_full` is another realization of the same configuration, not
another method. For a strictly fixed-input comparison, prepare one input set
for each search policy and use it for both link constructions. Report that
matched reference separately if it differs from the system-comparison run.

Set the index and read its actual namespace; keep using the reference query file:

```bash
export PREHOP_NAMESPACE=$("$PYTHON_BIN" -c \
  'import json,os; print(json.load(open(os.environ["PREHOP_INDEX"]))["index_policy"]["index_namespace"])')
export REPRO_DIR="$PWD/data/results/reproduce-mhr-representations"
mkdir -p "$REPRO_DIR"

for profile in question_full question_body; do
  "$PYTHON_BIN" scripts/prepare_ablation_inputs.py \
    --index-stats "$PREHOP_INDEX" --queries "$PREHOP_QUERIES" \
    --profile "$profile" --output "$REPRO_DIR/$profile-inputs"
done
```

Build body-based links in a separate namespace, preserving passages, body
embeddings, NEXT and the question graph's outgoing degree budget:

```bash
"$PYTHON_BIN" scripts/prehop_ablation.py \
  --mode index --profile body_body --namespace reproduce_mhr_body \
  --run-id reproduce-mhr-body-index --corpus-tag multihoprag \
  --dataset data/multihoprag_corpus --queries "$PREHOP_QUERIES" \
  --clone-body-from "$PREHOP_INDEX" \
  --reference "$REPRO_DIR/body-reference.json" --execute
```

Run the four profiles using the corresponding graph and shared search inputs:

```bash
for profile in question_full question_body body_body body_full; do
  source_index="$PREHOP_INDEX"
  namespace="$PREHOP_NAMESPACE"
  search_profile=question_body
  extra_args=(--reuse-existing-index)
  case "$profile" in
    question_full|body_full) search_profile=question_full ;;
  esac
  case "$profile" in
    body_body|body_full)
      source_index="$PWD/data/index_stats/prehop_multihoprag_reproduce-mhr-body-index.json"
      namespace=reproduce_mhr_body
      extra_args=(--reference "$REPRO_DIR/body-reference.json")
      ;;
  esac
  "$PYTHON_BIN" scripts/prehop_ablation.py \
    --mode benchmark --profile "$profile" --namespace "$namespace" \
    --run-id "reproduce-mhr-$profile" --corpus-tag multihoprag \
    --dataset data/multihoprag_corpus --queries "$PREHOP_QUERIES" \
    --index-stats "$source_index" --direct-inputs "$REPRO_DIR/$search_profile-inputs" \
    "${extra_args[@]}" --execute
done
```

Results are under
`data/results/ablations/reproduce-mhr-<profile>/<profile>/prehop/multihoprag/seed_42/`.
All four profiles use the same expansion and selection policy. Search-channel
changes also change candidate breadth: 12 owners per enabled channel does not
mean 12 total initial candidates. Body links use the question graph's degree
budget; this is not an independently optimized body-link system. Clone/build
cost is incremental preparation, not cold total indexing cost.

## Measure stored versus online connection processing

Use a completed full-query reference with retained traces and its index. These
commands resolve the same connections offline and online without new answer
runs. Each timing arm executes once per query, with no warm-up:

```bash
export REPRO_TIMING="$PWD/data/results/reproduce-mhr-timing"
mkdir -p "$REPRO_TIMING"
"$PYTHON_BIN" scripts/prepare_reference_timing.py \
  --reference-result "$PREHOP_RESULT" --output "$REPRO_TIMING/starts.json"
"$PYTHON_BIN" scripts/prehop_connection_timing.py \
  --mode build --index-stats "$PREHOP_INDEX" \
  --store "$REPRO_TIMING/store.json" --output "$REPRO_TIMING/build.json" --execute
"$PYTHON_BIN" scripts/prehop_connection_timing.py \
  --mode replay --index-stats "$PREHOP_INDEX" \
  --store "$REPRO_TIMING/store.json" --reference-inputs "$REPRO_TIMING/starts.json" \
  --queries "$PREHOP_QUERIES" --output "$REPRO_TIMING/paired.json" --execute
"$PYTHON_BIN" scripts/estimate_connection_total.py \
  --reference-result "$PREHOP_RESULT" --timing "$REPRO_TIMING/paired.json" \
  --queries "$PREHOP_QUERIES" --output "$REPRO_TIMING/estimated-total.json"
```

Destination agreement determines whether the comparison moves identical work
offline. Connection timing includes lookup/matching and destination hydration;
it excludes initial retrieval, evidence selection and answer generation.
Estimated online full-query time adds each paired connection-time difference
to that question's measured full-query time. It is an estimate, not a measured
online execution. Break-even query count is the ceiling of link-build time
divided by positive mean connection savings, with shared indexing costs cancelled.
It concerns cumulative connection processing, not parallel batch wall time.

## Compare a common reader over saved evidence

This compares exposed returned passages with one final-answer prompt. It never
reruns indexing, search, expansion, evidence selection or embeddings. Preparation
reads every `details[].retrieved_sources` passage in its original order with
common title/page/chunk labels; it adds no passage or token cutoff. Gold and
previous answers are excluded. Native-only structured graph context is not
added, so this does not reproduce each system's native answer pipeline.

Pass one completed result per dataset/system, in the intended execution order:

```bash
"$PYTHON_BIN" -m scripts.prepare_synthesis_inputs \
  "$PREHOP_RESULT" \
  --output data/synthesis/inputs.jsonl
"$PYTHON_BIN" main.py --mode synthesize \
  --trace-inputs data/synthesis/inputs.jsonl \
  --output-dir data/results/common-reader --concurrency 24
```

Add the other native result paths to the preparation command for comparisons.
Groups run sequentially; questions within a group run concurrently. The reader
uses the shared evidence-checking messages and configured generation model.
Whole contexts are sent unchanged. Context-length rejection is an execution
error; empty successful output is retained without quality-based regeneration.

Outputs are `responses.jsonl`, `status.json`, `events.jsonl` and
`execution_config.json`. The latter controls request spacing, concurrency and
`pause` between request windows. Transport timeout and the total attempt budget
come from the common inference contract and are recorded with responses. Rate
limits cause a shared cooldown respecting `Retry-After`; retries retain the exact input. Resume with the same command;
only matching messages, reader settings and model reuse successful responses.
Changed-input responses are removed from the current response file; original
retrieval results stay untouched. `generation_complete` means response records
exist, not that answer accuracy or research validity has been established.

Score the common-reader responses against the same original result files:

```bash
"$PYTHON_BIN" -m scripts.export_official_results "$PREHOP_RESULT" \
  --synthesis-responses data/results/common-reader/responses.jsonl \
  --output-dir data/results/common-reader/scores
```

With HotpotQA, supply its prepared `--hotpot-sentence-store` if it is outside
the default data directory. Original query identities, annotations and retrieval
predictions are retained; source and response hashes identify the new score
artifacts. Incomplete, duplicated or mismatched response populations are rejected by this
offline evaluator; original retrieval failures retain zero quality. This does
not add an execution gate. A single source writes
`<source-stem>.common-reader.official.json` under the score directory; multiple
sources also produce dataset-specific comparison CSVs. For the example:

```bash
"$PYTHON_BIN" - <<'PYCODE'
import json
from pathlib import Path
report = json.loads(Path('data/results/common-reader/scores/prehop_multihoprag.common-reader.official.json').read_text())
print(report['qa']['overall'])
PYCODE
```

Reader times are separate from original end-to-end query latency.

## Evaluate saved results

### Dataset-specific official results

Each completed benchmark writes `.official.json` and `.diagnostics.json`
beside its result. To collect final results in one command, pass the explicit
result paths for all methods and both datasets:

```bash
"$PYTHON_BIN" -m scripts.export_official_results \
  path/to/prehop_multihoprag.json path/to/hoprag_multihoprag.json \
  path/to/prehop_hotpotqa.json path/to/hoprag_hotpotqa.json \
  --output-dir data/results/final-comparison
```

Add the remaining methods' result paths to the same command. Select one final
source per dataset and method; the command does not discover the newest run.
Multiple inputs must be complete full or released-population results with
matching question IDs and annotations within each dataset. Partial runs,
subset runs and duplicate method entries are rejected by this offline comparison.

Each dataset directory contains `comparison.json`, `comparison.csv` and
per-method official files, retaining native units and source hashes. Datasets
are not averaged together. Auxiliary diagnostics remain beside the source
results. A single input exports per-result official/diagnostic JSON files.

This command aggregates recorded scores without modifying source results or
calling a model. It does not recompute predictions or certify official-code
parity. The runtime implements the pinned official scoring formulas after
shared answer extraction and, for HotpotQA, sentence projection; these adapters
are recorded in each official report. See the
[method implementation](METHOD.md#persistence-and-ownership) for their
boundaries.

### Common passage-retrieval analysis

This optional diagnostic analysis is separate from the official-only final
comparison above. It does not supply official HotpotQA ranking scores.

Use the [shared diagnostic definitions](#common-passage-retrieval-metrics)
without changing the native benchmark's official fields.

Create a manifest from your result paths and hashes. This example adds only the
MultiHop-RAG reference; add named conditions or a `hotpotqa` object for other runs:

```bash
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from pathlib import Path
result = Path(os.environ['PREHOP_RESULT'])
manifest = {'multihoprag': {'prehop': {
    'path': str(result), 'sha256': hashlib.sha256(result.read_bytes()).hexdigest()
}}}
Path('data/results/retrieval-manifest.json').write_text(json.dumps(manifest, indent=2))
PY
"$PYTHON_BIN" -m scripts.evaluate_saved_retrieval \
  --manifest data/results/retrieval-manifest.json \
  --multihop-queries data/multihoprag_queries.json \
  --hotpot-queries data/hotpotqa_queries.json \
  --sentence-store data/hotpotqa_corpus/sentences.sqlite3 \
  --output data/results/common-retrieval-scores.json
```

The evaluator currently requires both prepared query files and the sentence
store even for a one-dataset manifest. It expects full-population result files;
MultiHop-RAG's null questions are then excluded from retrieval scores. Outputs
include source hashes, per-question values, means, 95% condition-mean intervals
and MultiHop-RAG fact-count strata. Intervals use the recorded original-question cluster
bootstrap draws and seed; they do not rerun retrieval. Paired effect intervals
must be calculated from paired query outcomes, not differences of interval endpoints.

## Fixed-candidate final selection

To test whether stored HOP/NEXT connections help the LLM select complementary
evidence, run the graph-selection comparison on your own Prehop result:

```bash
"$PYTHON_BIN" -m scripts.compare_prehop_graph_selection \
  --reference "$PREHOP_RESULT"
```

The command loads `.env`, recovers the complete candidates and original ranking
prompt from the reference's trace files, and runs both the original prompt and
the prompt with stored connections. It uses the existing generation settings
and decoder. It requires generation service access; it makes no graph,
embedding or answer-generation calls. HotpotQA additionally needs its prepared
`data/hotpotqa_corpus/sentences.sqlite3` mapping. Run each dataset separately.

Each invocation writes a new `data/results/graph-selection-<timestamp>/`
directory. `selections.jsonl` retains requests, returned rankings, failures,
usage and selection time. The two condition result files have separate
`.official.json` and `.diagnostics.json` exports; `comparison.json` contains
paired original-question cluster intervals. MultiHop-RAG exports official
ranking metrics, while HotpotQA exports official supporting-fact metrics and
labels its adapted ranking measures separately. Answer and joint outcomes are
unmeasured. This comparison does not change the production selector or its
configuration. Retain the reference's traces; they contain the candidate pools
that cannot be recovered from the final selected passages alone.

[The selector comparison](../scripts/compare_prehop_selectors.py) compares the
recorded LLM selection against deterministic orders over the same complete
candidate pool. It uses the recorded selection limit and produces all five
common retrieval measures, condition means and paired differences from LLM
selection, with original-question cluster bootstrap intervals.
HotpotQA also receives full-return Supporting Fact EM, precision, recall and
F1. No answers are generated and no new latency is measured.

From the repository root, supply a complete Prehop benchmark result, its query
annotations, and the associated trace payloads:

```bash
"$PYTHON_BIN" -m scripts.compare_prehop_selectors \
  --dataset multihoprag \
  --reference "$PREHOP_RESULT" \
  --queries "$PREHOP_QUERIES" \
  --output data/results/reproduce-mhr-selectors
```

The reference must cover every query ID without terminal failures and record
`final_rank_variant=fused`. The output directory must be new. For HotpotQA, use
`--dataset hotpotqa`, the
matching reference and query file, and
`--sentence-store data/hotpotqa_corpus/sentences.sqlite3`. If traces have moved,
pass their `events.jsonl` paths with `--trace-events`; each event's relative
payload path must still resolve beside that event file.

The default comparison includes the recorded LLM order, fused
semantic/representation order, and representation-rank order. The latter sums
channel reciprocal ranks, retains the recorded graph-score propagation, and
breaks ties by descending passage identity. The fused alternative retains the
recorded order, including ties. Each deterministic condition takes the first
recorded `top_k` candidates before applying the same source deduplication.
This can return fewer than `top_k` unique passages; there is no refill after
deduplication in these deterministic conditions.

For the raw hybrid-score alternative, add either:

- `--raw-score-records PATH`: consume a saved replay JSONL. The script checks
  the scoring payload hash and each channel's owner order before propagating
  scores along the recorded graph paths.
- `--replay-raw --index-stats PATH`: use the original namespace and recorded
  query embeddings to replay search. It makes database reads without generation,
  embedding or index writes, and saves `raw-score-records.jsonl`. An owner-order
  mismatch stops comparison; partial analysis is not reported as complete.

Load the existing `.env` without replacing exported settings before a live replay:

```bash
source scripts/lib.sh
load_project_env "$PWD/.env"
```

The live replay implements body/Q−/Q+ search using the reference's default
top-k and recorded query vectors. It is intended for the documented full-channel,
multiplier-one, sentence-channel-disabled reference. A different search policy
requires a matching replay implementation; matching model names alone is
insufficient. Offline fused and representation conditions need no database.

Conditions write `llm.json`, `fused.json`, `representation.json`, and optional
`raw.json`, each with benchmark-shaped `details` and `retrieved_sources`.
`comparison.json` contains metrics, paired intervals,
query counts and source hashes. These outputs also work with the saved-result
evaluator above. Generated results and traces remain local and ignored.

## Measurement definitions

Report datasets separately with metric populations and failures. Matched quality
comparisons use paired query IDs; HotpotQA intervals cluster original question
IDs. Query bootstrap does not capture model/index-build variability or condition
selection bias. Compare latency only under a declared common serving/load window.

| Measurement | Scope |
| --- | --- |
| Index wall time | Original index pipeline, including waiting/retries/native work; amortized over manifest sources. Post-timer reporting/capacity collection is separate. |
| Query batch wall time | Full-batch dispatch to last answer/terminal failure, including queues and interleaved checkpoints; excludes initialization/trailing reports. Dividing by questions gives inverse throughput. |
| Query latency | Individual response time; inspect service, worker queue and wall-latency fields separately. It is not batch time divided by questions. |
| Benchmark segment wall time | Accumulated checkpointed segments, including setup/checkpoints. Resumed segments are not uninterrupted throughput. |
| Reuse/clone preparation | Separate from original cold construction and prior failed attempts. |
| Storage | Neo4j logical payload estimates and file-backed physical bytes are different measures. Trace bytes are separate; trace I/O remains in phase time. |

Partial, target-failed or resumed batches do not supply continuous throughput.
Fully executed batches retain terminal query failures. Missing native usage and
cost are unavailable, not zero. Connection-delta estimates and common-reader
replay times are not measured online end-to-end latency.

## Managed campaigns and additional analyses

For an index-only batch after README setup:

```bash
"$PYTHON_BIN" scripts/index_matrix.py launch indexing-01
```

It owns a detached session and writes `data/results/<campaign>/index-supervisor/`
plan/status and named logs. Smoke/full indexes run sequentially over registry
targets. Smoke failure isolates a target; index completion does not run full
query benchmarks. Complete receipts can supply `core/index_reuse.py` version-2
links; the legacy one-query matrix uses version 1. File-backed query copies
preserve source index identity and keep copy costs separate.

`plan_link_experiments.py` and `link_experiment_campaign.py` handle dependencies,
explicit concurrency, exclusive timing jobs and adoption of owned processes.
Use their `--help` for plan/launch arguments. Resume retains recorded
completed/failed tasks, adopting live owned processes instead of silently
retrying; failed dependencies remain explicit. Timing exclusivity covers that
campaign, not unrelated gateway clients.

Supervisors record PID/start/boot identity and clean only verified descendants
with bounded TERM waits. Surviving owned processes block restart. Heartbeats
show liveness, not completed source work. Reboot recovery is not automatic.
The separate `paper_campaign.py`/`run_paper_matrix.sh` evidence-ledger workflow
records executed stages; its historical labels are not publication approval or
extra dispatch gates. Keep progress counters, ETAs and temporary analysis in
run artifacts, outside public method documentation.

HopRAG recovery retains document caches and committed edge groups; uncommitted
interrupted groups are rescored. Preserve earlier statistics and attempt costs.
`benchmark_hoprag_edges.py` measures block-scoring parity, throughput and RSS
against the pinned native implementation; these are not corpus indexing costs.

[Link usefulness](../scripts/analyze_ablation_links.py) measures evidence added
beyond direct retrieval and retained by selection.
[Co-evidence connectivity](../scripts/analyze_evidence_connections.py) reads a
frozen graph and gold after construction with a degree-matched random null.
It does not establish query-time retrieval success or correctness of every
link. `analyze_expansion_factorial.py` computes conditional HOP/NEXT effects and
additive interaction from saved conditions. All retain explicit source identities
and analysis denominators; each exposes required inputs through `--help`.
