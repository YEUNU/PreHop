# Reproducing and evaluating experiments

For researchers reproducing comparisons or interpreting saved results. Follow
the [README](../README.md) for installation, data preparation, the small live
smoke and full system runs. This guide owns data/evaluation definitions,
continuation, controlled experiments and measurement boundaries. Commands run
from the repository root in Bash using the selected `PYTHON_BIN`.

## Paper experiment inventory

The final paper, *Graph Expansion versus Deeper Direct Retrieval for Multi-Hop
RAG*, uses the following comparisons. Appendix analyses remain part of the
reproduction scope, including inconclusive and negative results.

| Comparison or analysis | Reproduction path |
| --- | --- |
| Seven systems on MultiHop-RAG and reduced-corpus HotpotQA | README benchmark commands, saved-result exporter and common reader below |
| Fixed-start HOP/NEXT factorial | `run_primary_hop_ablation.py`, `analyze_expansion_factorial.py` |
| Fixed-start expansion QA | `compare_prehop_direct_qa.py` with `init-fixed-start` |
| Graph expansion versus deeper direct retrieval, token and count controls | Frozen preparation/selection/reader protocols; see the matched-budget section below |
| Evidence overlap, retrieval-depth prefixes and selection retention | `analyze_evidence_accessibility.py`, `analyze_ablation_links.py` |
| Body-only versus body/Q−/Q+ search in the same question index | `prepare_ablation_inputs.py`, `prehop_ablation.py` |
| LLM, fused, representation and raw-score selection | `compare_prehop_selectors.py` |
| Matched-count shuffled HOP evidence supply | `analyze_link_supply.py` over the archived samples |
| Five-condition link-construction pilot | `compare_link_representations.py` |
| LightRAG passage-exposure and HopRAG retrieval-budget sensitivity | Archived condition-specific retrieval outputs and common-reader replay |
| Reader-development exclusion and question/evidence subgroups | `reader_development_groups.json`, saved per-query scores and paired analyses |
| Index wall time and native query latency | Original run statistics under the measurement definitions below |

Saved indexes, results, trace payloads, prepared pools and run-specific protocol
snapshots are local artifacts, not bundled with the source checkout. Replaying
published measurements requires those artifacts. Implementation tests use mocks
and fixtures to check correctness; they are distinct from paper experiments.

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

These datasets use shared final-answer extraction before official scoring.
`utils/official_results.py` records the evaluator revisions and these adapters.
Scores remain in native unscaled units. Auxiliary normalized/fuzzy answer and
passage-coverage diagnostics do not replace official metrics. Terminal failures
remain in applicable populations with zero quality; unknown usage and cost
remain unavailable. Evaluation uses deterministic dataset metrics and contains
no LLM-judge stage.

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

To measure this fixed-start expansion's effect on MultiHop-RAG answers, set
`DIRECT_ONLY_RESULT` to the completed `--expansion none` result. Reuse the
original selected passages from both conditions with a common reader:

```bash
export FIXED_QA_DIR="data/results/fixed-start-qa-$(date +%Y%m%d-%H%M%S)-$$"
"$PYTHON_BIN" -m scripts.compare_prehop_direct_qa init-fixed-start \
  --reference "$PREHOP_RESULT" --direct-reference "$DIRECT_ONLY_RESULT" \
  --queries "$PREHOP_QUERIES" --execution-profile "$RAG_EXECUTION_PROFILE" \
  --output "$FIXED_QA_DIR"
for stage in prepare canary generate evaluate; do
  "$PYTHON_BIN" -m scripts.compare_prehop_direct_qa "$stage" \
    --output "$FIXED_QA_DIR" || break
done
```

Preparation verifies the executed Direct-only candidates, their scores and query
embeddings against the primary retrieval traces. The reader receives all selected
passages in their saved order, with no gold annotations or previous answers.
Both arms receive fresh reader calls, interleaved in a fixed random order;
the first eight query pairs form a retained canary. Official QA includes all
questions, including null questions and terminal failures. This tests fixed-start
expansion with different selector candidate pools; it does not match their input
budgets. The protocol and official exports distinguish it from the comparison below.
Auxiliary answer EM/F1 use the 2,255 non-null questions, retaining their terminal
failures as zero. These metrics are unmeasured for null questions; official QA
still uses all 2,556 questions.

## Compare search representations

Both profiles use the same question-link index. `question_full` searches body,
Q− and Q+; `question_body` searches body only. Keep the source index and reference
query file fixed. A new execution is a new realization, not a relabeling of the
paper's archived results. The historical channel comparison uses
`body_bridge_min` scoring; current default scoring uses `body_only`. Read the
source result's recorded policy when reproducing that comparison.

```bash
export PREHOP_NAMESPACE=$("$PYTHON_BIN" -c \
  'import json,os; print(json.load(open(os.environ["PREHOP_INDEX"]))["index_policy"]["index_namespace"])')
export REPRO_DIR="$PWD/data/results/reproduce-mhr-representations"
mkdir -p "$REPRO_DIR"
for profile in question_full question_body; do
  "$PYTHON_BIN" scripts/prepare_ablation_inputs.py \
    --index-stats "$PREHOP_INDEX" --queries "$PREHOP_QUERIES" \
    --profile "$profile" --output "$REPRO_DIR/$profile-inputs"
  "$PYTHON_BIN" scripts/prehop_ablation.py \
    --mode benchmark --profile "$profile" --namespace "$PREHOP_NAMESPACE" \
    --run-id "reproduce-mhr-$profile" --corpus-tag multihoprag \
    --dataset data/multihoprag_corpus --queries "$PREHOP_QUERIES" \
    --index-stats "$PREHOP_INDEX" --direct-inputs "$REPRO_DIR/$profile-inputs" \
    --hop-semantic-variant body_bridge_min --reuse-existing-index --execute
done
```

Results are under
`data/results/ablations/reproduce-mhr-<profile>/<profile>/prehop/multihoprag/seed_42/`.
Search-channel changes also change candidate breadth: 12 owners per channel
need not mean 12 total initial candidates.

## Compare HOP evidence supply at matched candidate counts

The paper's question-link versus degree-preserving shuffled-link comparison
is evaluated from its archived `protocol.json`, `prepared.json`, `snapshot.json`
and `samples.npz`. Set `SUPPLY_ARCHIVE` to that prepared run and provide a new
output directory:

```bash
"$PYTHON_BIN" -m scripts.analyze_link_supply \
  --prepared "$SUPPLY_ARCHIVE" \
  --output data/results/reproduce-hop-supply
```

The evaluator verifies the prepared hashes and retains the exact budgets and
random samples of the question and shuffled conditions. The archived experiment
also used a body-link condition when setting budgets and shared sampling
priorities. The paper evaluator reports only the two retained conditions;
removing the historical body arm must not change those samples or relabel the published scores.
Gold is read only to evaluate the saved candidates. The report measures
pre-selector evidence supply, not selected evidence or answer quality.
Original-question bootstrap intervals remain conditional on the saved graph
and sample realizations, with variation across realizations reported separately.

For an exploratory comparison of link representations, use
[`compare_link_representations.py`](../scripts/compare_link_representations.py).
It reuses the archived corpus, question links, passage-neighbor links, and
direct inputs; only Q+ to passage ANN lookups are new. Preparation reads the
original Neo4j namespace without writing to it or calling an inference model.

```bash
export LINK_PILOT="data/results/link-representation-$(date +%Y%m%d-%H%M%S)-$$"
export RAG_RUN_ID="${LINK_PILOT##*/}"
"$PYTHON_BIN" -m scripts.compare_link_representations prepare \
  --archive "$SUPPLY_ARCHIVE" --output "$LINK_PILOT-four-arms" --limit 256 --seed 42
"$PYTHON_BIN" -m scripts.compare_link_representations add-base \
  --prepared "$LINK_PILOT-four-arms" --output "$LINK_PILOT"
"$PYTHON_BIN" -m scripts.compare_link_representations evaluate \
  --output "$LINK_PILOT" --fused-only
"$PYTHON_BIN" -m scripts.compare_link_representations select \
  --output "$LINK_PILOT" --workers 8 --limit 8
"$PYTHON_BIN" -m scripts.compare_link_representations select \
  --output "$LINK_PILOT" --workers 8
"$PYTHON_BIN" -m scripts.compare_link_representations evaluate --output "$LINK_PILOT"
```

The four expansion graphs use the same unique outgoing degree per source passage, bounded
by the smallest of the three semantic graphs; the random control rewires the
matched question graph while preserving its in/out degrees. Every query retains
its saved direct starts and bidirectional NEXT base, then adds the same number
of unique HOP passages in each arm, at most 12. The shared fused scorer uses
body semantics without bridge embeddings, followed by the unchanged selector
prompt and a final limit of 12. These are controlled variants, not the unmodified
main method. Passage counts do not guarantee equal token counts.

The fifth condition is Direct+NEXT-only. `add-base` extends an existing frozen
four-arm run into a new directory without changing those arms or their query
IDs, budgets, passage identities, orders, or fused selections. It reconstructs
the common-base scores from saved direct inputs and trace embeddings, omitting
all HOP score contributions and recomputing fusion ranks within the smaller
pool. It performs no database access or inference. Comparing an expansion arm
with this baseline intentionally changes input counts; only the four expansion
arms have matched added-passage counts.

The pilot fixes query IDs before new matching or scoring, retains null queries
and zero-budget cases, and makes at most 1,280 logical selector requests for
256 questions. Transport/JSON retries can increase wire attempts. The first
eight questions remain part of the run; resume preserves successes and terminal
failures. No answer generation is needed. Terminal failures receive zero quality
and null questions remain outside the retrieval denominator. Reports include
paired intervals and graph-by-selector interactions, conditional on the frozen
index and one random graph/sample realization. Historical results remain in
`historical-audit.json`; this pilot is exploratory, not a blind confirmation.

The five-condition protocol records the revised comparison priorities before
LLM selection, while acknowledging the already observed candidate-supply and
fused results. It reports all paired contrasts on the same evidence-bearing
query population, with pointwise exploratory intervals. Intervals including
zero are inconclusive rather than evidence of equivalence. Negative contrasts
and terminal failures remain in the report.

The descriptive evidence funnel identifies distinct gold facts absent from the
entire common base, their supply in additional candidates, their survival in
selected top-10 passages, and complete selected evidence after that survival.
It also records all completeness gains and losses versus the base-only selector,
including changes without any new fact. Supply-conditioned subsets differ by
arm and are not used as the population for paired performance claims. Zero
supplied facts gives an undefined conditional retention rate. Gold is used only
by the evaluator and never enters prepared selector inputs. The LLM-versus-fused
interaction does not separate joint comparison from other LLM capabilities.

## Compare graph expansion with deeper direct retrieval

The matched-budget comparison uses the same question index, original query,
query embedding and initial candidates in both conditions. Deeper direct
retrieval increases the passage limit to 256 per body/Q−/Q+ channel, ranks the
full pool, retains every initial candidate and adds the longest prefix that
fits Prehop's selector input token budget for that question. Both pools use
body-based scoring, the same rank fusion, selector and common reader, with new
selector and reader outputs. The comparison includes the input order induced
by each retrieval procedure. A separate MultiHop-RAG control matches candidate
counts exactly; it is not a token-matched condition.

Use the archived protocol and preparation code snapshots for these executed
conditions. They pin the candidate identities and order, token counter, budget
rule and selector/reader outputs. The fixed-start QA runner above does not
construct deeper-direct pools. The source checkout provides the saved-output
evaluators below; it does not independently recreate the archived matched-budget
run from a benchmark result alone.

### Evidence accessibility in saved candidate pools

[`analyze_evidence_accessibility.py`](../scripts/analyze_evidence_accessibility.py)
audits the paper's archived token-matched graph and deeper-direct pools. It
requires an explicit run directory with `protocol.json`, the original saved
inputs, query annotations, HotpotQA sentence mapping and frozen pool-analysis
manifest. The source checkout alone does not contain these generated artifacts.
The protocol pins source hashes, populations, prefix fractions and uncertainty
settings before analysis. Use a new output directory for each analysis; an
existing `analysis.json` is not overwritten.

```bash
"$PYTHON_BIN" -m scripts.analyze_evidence_accessibility --run "$ACCESSIBILITY_RUN"
```

Every original start is retained. Additional candidates are taken in the saved
deeper-direct order, with the requested fraction rounded down. Gold facts and
support sentences are matched only after constructing these prefixes, using the
existing evaluation rules. The audit verifies all endpoint coverage sets against
the frozen report, preserves null-query identities, and clusters HotpotQA
intervals by original question. `verified-inputs.json` records the checked input
hashes; `analysis.json` contains per-query curves and aggregates.

The same audit records every passage ID supporting each graph-new gold unit.
Its exhaustive cases are a witness shared by both pools, witnesses with only
different IDs across pools, and evidence absent from deeper direct. The shared
case is additionally checked for graph-only witnesses; it is not counted twice.
Passage-ID novelty and gold-unit overlap alone cannot distinguish these cases.

When the protocol specifies `reader_comparisons` and `development_groups`, the
audit also reaggregates the pinned per-query scores before and after removing
all occurrences of each reader-development question. Query hashes and both arm
populations must agree; unavailable required metrics are errors, while terminal
failures retain zero quality. This recomputes score aggregates and paired
intervals, not model outputs, and remains a retrospective sensitivity analysis.
An offline replay package can supply the frozen texts, sentence mapping, score
records, analysis snapshot, protocol and checksums independently of the source
checkout. Its `replay.py` verifies inputs and compares all per-query records and
aggregates with the packaged expected result; no database or model is needed.

These are candidate-depth diagnostics inside one frozen maximum pool. Only its
endpoint had a matched selector-token ceiling. Intermediate prefixes have no new
selector or reader results, and their token budgets are not matched. Graph-only
evidence means absent from this recorded direct pool, not inaccessible to every
retriever. A union-coverage gain does not predict answer quality or the result
of fusion under a fixed budget. This audit makes no model or retrieval calls.

## Compare a common reader over saved evidence

Using the same reader, model and prompt controls answer-generation differences
and compares the downstream usefulness of each system's exposed passages.
Identical instructions need not be equally optimal for every retrieval output. It never
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

If a prompt was developed on evaluation questions, report that development
population and distinguish its exclusion from a previously untouched test set.
The paper's original development IDs and query-file hashes are in
[`reader_development_groups.json`](../configs/reader_development_groups.json).
Exclude every released occurrence of each listed original question, using the
same retained IDs for all systems. The file controls this answer-evaluation
sensitivity analysis only; it is not an input to retrieval or evidence selection.

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
cost are unavailable, not zero. Common-reader replay times are separate from
measured native online end-to-end latency.

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
explicit concurrency and adoption of owned processes for the native systems,
search-channel comparison and HOP/NEXT expansion conditions.
Use their `--help` for plan/launch arguments. Resume retains recorded
completed/failed tasks, adopting live owned processes instead of silently
retrying; failed dependencies remain explicit.

Supervisors record PID/start/boot identity and clean only verified descendants
with bounded TERM waits. Surviving owned processes block restart. Heartbeats
show liveness, not completed source work. Reboot recovery is not automatic.
The separate `paper_campaign.py`/`run_paper_matrix.sh` evidence-ledger workflow
records executed stages; its historical labels are not publication approval or
extra dispatch gates. Keep progress counters, ETAs and temporary analysis in
run artifacts, outside public method documentation.

HopRAG recovery retains document caches and committed edge groups; uncommitted
interrupted groups are rescored. Preserve earlier statistics and attempt costs.

[Link usefulness](../scripts/analyze_ablation_links.py) measures evidence added
beyond direct retrieval and retained by selection.
`analyze_expansion_factorial.py` computes conditional HOP/NEXT effects and
additive interaction from saved conditions. All retain explicit source identities
and analysis denominators; each exposes required inputs through `--help`.
