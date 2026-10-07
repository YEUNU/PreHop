# Reproducing and evaluating experiments

For researchers reproducing comparisons or interpreting saved results. Follow
the [README](../README.md) for installation, data preparation, the small live
smoke and full system runs. This guide describes data and evaluation definitions,
continuation, controlled experiments and measurement boundaries.
[Setup](SETUP.md) owns service and environment configuration;
[Method](METHOD.md) defines retrieval behavior. Commands run
from the repository root in Bash using the selected `PYTHON_BIN`.

| Task | Section |
| --- | --- |
| Compare Prehop with the dense baseline | [Baseline comparison](#run-a-baseline-comparison) |
| Understand datasets, populations and metrics | [Data and evaluation](#data-and-evaluation) |
| Resume an existing benchmark run | [Completion and continuation](#completion-and-continuation) |
| Compare HOP/NEXT expansion | [Expansion ablation](#compare-hop-and-next-expansion) |
| Compare expansion with a larger direct-retrieval pool | [Matched-budget retrieval](#compare-graph-expansion-against-direct-retrieval-with-more-candidates) |
| Evaluate saved passages with one answer generator | [Answer-generation replay](#compare-answer-generation-over-saved-evidence) |
| Reconstruct depth and admission comparisons | [Fixed-input analyses](#fixed-input-analyses) |
| Export dataset scores | [Evaluate saved results](#evaluate-saved-results) |

## Paper experiment inventory

*Reached but Not Retained: Candidate Admission under Reranker Input Ceilings in Graph-Based Multi-Hop RAG* studies which
reachable evidence enters a limited LLM reranking input. Standard Prehop remains
the one-step method described in [Method](METHOD.md); the deeper neighborhoods
and changed admission policies are experimental conditions. Appendix analyses,
including inconclusive and negative results, remain in the reproduction scope.
Archived graph-construction and link-control analyses remain available even where
they are no longer included in the manuscript.

The main paper evaluates reachability, admission and returned evidence. Appendix H
contains the fixed-initial-passage QA ablation, evidence partition and direct-prefix
coverage. The ablation uses a separate generation run and does not match input
tokens; intermediate direct prefixes measure coverage without new QA evaluations.

The candidate-acquisition controls and supporting reference analyses are:

| Comparison or analysis | Reproduction path |
| --- | --- |
| HOP/NEXT factorial with fixed initial passages | `run_primary_hop_ablation.py`, `analyze_expansion_factorial.py` |
| Expansion QA with fixed initial passages (Appendix H.1) | `compare_prehop_direct_qa.py` with `init-fixed-start` |
| Graph expansion versus direct retrieval with more candidates, token and count controls | Archived candidate preparation, reranking and answer-generation protocols; see the matched-budget section below |
| Evidence overlap and candidate-count prefixes (Appendix H.2–H.3), with supporting channel and retention analyses | `analyze_evidence_accessibility.py`, `analyze_ablation_links.py` |
| Body-only direct retrieval under matched token ceilings | Archived body-search outputs, original body-channel starts and token-budget preparation; the three-policy reranking and answer-generation run and its repeat with identical reranking requests and the same answer-generation procedure (Appendix C) |
| Reranker input-ceiling sweep at a quarter and a half of the additional token allowance (Appendix F) | Saved one-step and direct pools, mandatory initial passages, whole-passage prefix fitter and tokenizer; fused-order, cosine-order and direct prefixes reranked and answered under common per-query ceilings |
| Repeated-fact sensitivity of the recall contrasts (Appendix D) | Fixed fact-contribution ranking and whole-question exclusion masks; an additional paired bootstrap resamples connected components of questions sharing a fact string over the same saved outputs; no reranking or answer regeneration |
| Graph distance of gold evidence missing from the one-step pool (supporting acquisition analysis) | Saved graph, initial passages and corpus witnesses; shortest HOP/NEXT paths without an input budget; no model calls |
| Direct answer context versus the entire Prehop candidate pool | Paired reaggregation of the archived per-question pool and final-context gold-coverage sets |
| LLM reranking, fused-score, retrieval-rank-score and hybrid-score ordering | `compare_prehop_selectors.py` |
| Matched-count question, body and shuffled HOP evidence supply | `analyze_link_supply.py --include-body` over the archived samples |
| Prompt-development exclusion of the controlled contrasts and initial-coverage groups | `reader_development_groups.json`, saved per-query scores and paired analyses |
| Archived seven-system QA, indexing time and query latency (cross-system comparison) | `data/final_experiments/provenance/uniform_reader_result_index.json` and `data/results/timing-audit-20260928/paper_timing_sources.json`; common-answer QA and native pipeline timing are separate measurements |
| Indexing time and query latency of Prehop and the dense baseline | Original indexing statistics and timings; Prehop query latency remeasured using existing indexes |

The depth and admission comparisons use the following fixed-input analyses:

| Comparison or analysis | Reproduction inputs and scope |
| --- | --- |
| One-to-four-step HOP/NEXT neighborhoods and token-limited admission | Saved graph, initial passages/scores, query–body similarities, full prompt metadata and tokenizer; reconstruct neighborhoods, orders and whole-passage prefixes |
| Four-step rank-fusion versus similarity-only admission, compared against the one-step policies | The same admitted sets and recorded downstream requests from two runs: the primary run presents each set in its own admission order, the secondary run presents both in query-to-body cosine order; re-score saved passages and answers, including prompt-selection exclusions; post hoc contrasts against the one-step three-policy run pair both runs' per-query scores with `ablation_statistics.cluster_interval` |
| Question-link versus body-similarity graphs crossed with admission policy | Saved graph-specific neighborhoods, scores and admitted sets; evaluate coverage, shared evidence and count-matched prefixes |
| Inherited-score exchange on a common passage pool | Both full-graph score vectors, the intersection of passage IDs and fixed addition counts; preserve graph paths outside the intersection |
| Distance and starting-score components | Saved shortest distances and starting scores for both graphs, with the same common pool, counts and scoring rules |
| Input ceiling, distance decay, rank-fusion offset and score ties | Fixed graph/candidate inputs and each condition's protocol; candidate coverage only |
| Repeated-fact sensitivity of the graph and score comparisons | The same five exclusion masks applied to the graph-admission and common-pool conditions |
| Dataset differences, source selection and changed evidence | Saved graph structure, source identities, initial coverage, HOP/NEXT reachability, and passage-to-annotation matches |

See [Fixed-input analyses](#fixed-input-analyses) for the available replay
packages and the distinct limits of each one. The repository keeps only the
implementation and tooling behind these retained experiments.

The source checkout supports new Prehop and dense-baseline benchmark runs,
saved-passage answer generation, HOP/NEXT controls, and analysis of their saved
outputs.
Exact replay of the archived matched-budget comparison, shuffled-link supply
analysis and depth/admission analyses additionally requires their saved indexes, candidate pools, samples and
protocol/code snapshots.
These generated artifacts are not bundled with the source. Their requirements
are specified in the respective sections below. Implementation tests use mocks
and fixtures; they do not reproduce paper scores.

## Data and evaluation

| Dataset | Prepared corpus and queries | Evaluation population |
| --- | --- | --- |
| MultiHop-RAG | `data/multihoprag_corpus/`, `data/multihoprag_queries.json` | 609 documents; 2,556 QA questions, including 301 null questions; 2,255 evidence-bearing retrieval questions |
| HotpotQA (HippoRAG release) | `data/hotpotqa_corpus/`, `data/hotpotqa_queries.json` | 9,221 source paragraphs; 1,000 released occurrences representing 944 original questions |

The retained Prehop indexes contain 8,529 six-sentence passages on MultiHop-RAG
and 10,885 on HotpotQA. Source counts and indexed passage counts are different
units; HotpotQA's released source paragraphs can be split into multiple passages.

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

`record_paper_completion.py` records a completed result in `admission.json`,
preserving an existing record. Assess a run using its question population,
terminal failures, source index and measurement scope. Source edits do not
update already-loaded Python code; new segments retain their actual provenance.

The smoke's `query.json` holds its answer/passages; `index_evidence.json` and
`evidence.json` bind execution sources. Full results hold `details`, including
answers, ordered `retrieved_sources`, scores and `index_manifest_stats_path`.
Retain the graph and trace references for the analyses below.

## Run a baseline comparison

After the [README benchmark setup](../README.md#run-the-benchmarks-and-inspect-scores),
compare Prehop with the dense baseline (`naive`). Both use the same environment,
fixed passage windows, embedding model, Neo4j vector-index infrastructure and gateway. Run and export both
dataset/system pairs for the comparison:

```bash
export COMPARISON_RUN="comparison-$(date +%Y%m%d-%H%M%S)-$$"
results=()
for dataset in multihoprag hotpotqa; do
  for method in prehop naive; do
    run_id="$COMPARISON_RUN-$dataset-$method"
    bash scripts/run_paper_target.sh "$dataset" "$method" "$run_id" || break 2
    results+=("data/results/$run_id/$method/$dataset/seed_42/${method}_${dataset}.json")
  done
done
if [ "${#results[@]}" -eq 4 ]; then
  "$PYTHON_BIN" scripts/export_official_results.py "${results[@]}" \
    --output-dir "data/results/$COMPARISON_RUN-tables"
fi
```

These commands evaluate each system's benchmark answer pipeline. The paper's
comparison with a shared answer generator needs the separate
[saved-evidence answer replay](#compare-answer-generation-over-saved-evidence). New runs produce their own results;
identical settings do not guarantee identical generated answers or times.

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
[method](METHOD.md) describes primary behavior; the
[registry](../core/strategy_registry.py) owns model and budget defaults. Use
common declared serving conditions for time comparisons. Preserve historical
conditions instead of relabelling them with current defaults.

## Compare HOP and NEXT expansion

This experiment changes expansion while keeping the reference graph, initial
retrieval, scoring and LLM reranking settings fixed. Every condition retains
the direct passages:

| Condition | HOP | NEXT | `--expansion` |
| --- | --- | --- | --- |
| Prehop | On | On | Use the reference result |
| Direct-only | Off | Off | `none` |
| Direct + NEXT | Off | On | `next_only` |
| Direct + HOP | On | Off | `hop_only` |

After the reference finishes, prepare its saved initial passages and run one
intervention. Preparation reads trace files; the worker performs expansion,
reranking and answer generation:

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

To measure expansion's effect on MultiHop-RAG answers with fixed initial passages, set
`DIRECT_ONLY_RESULT` to the completed `--expansion none` result. Reuse the
original selected passages from both conditions with the shared answer generator:

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
embeddings against the primary retrieval traces. The answer generator receives all selected
passages in their saved order, with no gold annotations or previous answers.
Both conditions receive new answer-generation calls, interleaved in a fixed random order;
the first eight query pairs form a retained canary. Official QA includes all
questions, including null questions and terminal failures. This tests expansion
with fixed initial passages and different reranker pools; it does not match their input
budgets. The protocol and official exports distinguish it from the comparison below.
Auxiliary answer EM/F1 use the 2,255 non-null questions, retaining their terminal
failures as zero. These metrics are unmeasured for null questions; official QA
still uses all 2,556 questions.

## Compare HOP evidence supply at matched candidate counts

The question-link, body-link and degree-preserving shuffled-link comparison
is evaluated from its archived `protocol.json`, `prepared.json`, `snapshot.json`
and `samples.npz`. Set `SUPPLY_ARCHIVE` to that prepared run and provide a new
output directory:

```bash
"$PYTHON_BIN" -m scripts.analyze_link_supply \
  --prepared "$SUPPLY_ARCHIVE" \
  --include-body \
  --output data/results/reproduce-hop-supply
```

The evaluator verifies the prepared hashes and retains the original shared
samples and per-query budgets without resampling. `--include-body` reports all
three archived conditions and the body-minus-question and body-minus-shuffled
contrasts. Without this flag, the default question/shuffled report is unchanged.
Gold is read only to evaluate the saved candidates. The report measures
evidence supply before reranking, separately from selected evidence or answer quality.
Original-question bootstrap intervals remain conditional on the saved graph
and sample realizations, with variation across realizations reported separately.

## Compare graph expansion against direct retrieval with more candidates

The matched-budget comparison uses the same question index, original query, query
embedding and initial passages in both conditions. Direct retrieval with more
candidates increases the passage limit to 256 per body/Q−/Q+ channel, ranks the full
pool, retains every initial candidate and adds the longest prefix that fits Prehop's
reranker input token budget for that question. Both pools use body-based scoring, the
same rank fusion, LLM reranking and shared answer generation, with new rankings and
answers. The comparison includes the input order induced by each retrieval procedure. A
separate MultiHop-RAG control matches candidate counts exactly and evaluates
retrieval after new rankings; it does not generate answers or match input tokens.

Use the archived protocol and preparation code snapshots for these executed conditions.
They pin the candidate identities and order, token counter, budget rule, rankings and
generated answers. The expansion QA runner above does not construct the pools for direct
retrieval with more candidates. The source checkout provides the saved-output evaluators
below; it does not independently recreate the archived matched-budget run from a
benchmark result alone.

Complete reranker requests in the matched-budget comparison are tokenized
with `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, including its bundled chat template.
This identifies the tokenizer used to measure inputs, separately from the
served generation model's recorded revision, which stays in saved run
provenance.

### Reranker input-ceiling sweep

The sweep reuses the saved one-step Prehop pool and the saved multi-channel
direct-retrieval pool of the matched-budget comparison on MultiHop-RAG. For each
question the ceiling is `T0 + floor(lambda * (T1 - T0))`, where `T1` is Prehop's
original reranker prompt length and `T0` the largest prompt length of the
mandatory initial passages alone under the three presentations; `lambda` takes
0.25 and 0.5, so only the allowance beyond the initial passages is scaled.
Three candidate policies fill each ceiling with whole passages: the Prehop pool
in its fused order, the same pool in query–body cosine order (a query-conditioned
pruning of the expansion), and the direct pool in its saved order. At
`lambda = 1` the fused and direct prefixes reproduce the matched-budget requests
exactly, which the preparation verifies per question before any model call. Each
condition is reranked and answered once with the shared settings; the evaluation
reports official QA, MAP@10, normalized EM/F1, candidate recall and returned
recall with paired question bootstraps, and the follow-up analysis adds the
budget interaction, within-policy budget effects, final-answer-label rates and
label-restricted contrasts. Reproducing the sweep requires the archived pools,
protocol and preparation snapshots in addition to the source checkout.

### Body-only direct retrieval under the same token ceiling

This candidate-supply control uses the original corpus, body index, query text
and saved query embeddings. Native body vector and full-text search request a
passage limit of 256. Preparation retains the original 12 body-channel starts,
reconstructs body-only reciprocal-rank scores, and supplements those starts
with the body search results. Body-based semantic scoring and rank fusion order
the merged pool. The longest prefix of additional whole passages that fits the
saved Prehop reranker-input token ceiling is retained, preserving the full
ranking's relative order without recomputing ranks after taking the prefix.

Use the archived preparation protocol, code snapshots, original starts, raw
body search results and tokenizer to reproduce the executed pools. These
artifacts are required in addition to the source checkout. The retained
candidate sets are fixed before gold evaluation, which uses the same fact
matching, sentence projection and paired original-question bootstrap as the
primary comparison. Retrieval does not query generated questions or follow
graph links. Candidate coverage is evaluated before LLM reranking. The completed
three-policy comparison (Appendix C) uses saved rankings and
generated answers for graph expansion, direct retrieval with more candidates and
body-only direct retrieval. It is a separate run from the primary matched-budget
comparison; its repeated Prehop and multi-channel conditions, and a later
repeat with identical reranking requests and the same answer-generation procedure,
quantify observed generation variability at
temperature zero (Appendix C). Repeating requests also exposes the saved per-condition
final-answer-label rates and the reranker's repeated-ID counts reported in Appendix C. Use those final full-population outputs, not a
partial sample or the coverage-only preparation, for downstream scores. All three use the same LLM reranking and
answer-generation settings and return at most 12 passages. Body-only direct
retrieval differs from a body-only Prehop search that still follows graph links.
The input ceiling is shared, while the initial passages and
retrieval-rank scores differ from the multi-channel conditions. This is not an
isolated intervention on question generation or a measurement of body-only index
construction cost.

### Evidence accessibility in saved candidate pools

[`analyze_evidence_accessibility.py`](../scripts/analyze_evidence_accessibility.py)
analyzes the recorded token-matched pools from graph expansion and direct
retrieval with more candidates. It requires an explicit run directory with
`protocol.json`, the original saved inputs, query annotations, HotpotQA sentence mapping
and frozen pool-analysis manifest. The source checkout alone does not contain these
generated artifacts. The protocol pins source hashes, populations, prefix fractions and
uncertainty settings before analysis. Use a new output directory for each analysis; an
existing `analysis.json` is not overwritten.

```bash
"$PYTHON_BIN" -m scripts.analyze_evidence_accessibility --run "$ACCESSIBILITY_RUN"
```

Every original start is retained. Additional candidates are taken in the saved order
from direct retrieval with more candidates, with the requested fraction rounded down.
Gold facts and support sentences are matched only after constructing these prefixes,
using the existing evaluation rules. The analysis checks all endpoint coverage sets
against the frozen report, preserves null-query identities, and clusters HotpotQA
intervals by original question. `verified-inputs.json` records the checked input hashes;
`analysis.json` contains per-query curves and aggregates.

The same analysis records every passage ID covering each graph-new gold unit. Its
exhaustive cases are a supporting passage shared by both pools, supporting
passages with only different IDs across pools, and evidence absent from direct
retrieval with more candidates. The shared case is additionally checked for
supporting passages found only in the graph pool; it is not counted twice.
Passage-ID novelty and gold-unit overlap alone cannot distinguish these cases.

The final-context versus full-pool comparison pairs the direct condition's
saved answer-context coverage with coverage of every Prehop candidate on each
question. It reports their mean recall difference and complete-evidence rate,
using the same evaluation populations and original-question bootstrap. Any
selected subset of the Prehop pool has coverage bounded by that full pool.
This comparison therefore bounds annotated-evidence recall attainable by
reranking the original pool, without imposing a bound on QA or MAP. It requires
the archived per-question coverage sets and makes no new model calls.

When the protocol specifies `reader_comparisons` and `development_groups`, the
analysis also reaggregates the pinned per-query scores before and after removing
all occurrences of each prompt-development question. Query hashes and both arm
populations must agree; unavailable required metrics are errors, while terminal
failures retain zero quality. This recomputes score aggregates and paired
intervals, not model outputs, and remains a retrospective sensitivity analysis.
An offline replay package can supply the frozen texts, sentence mapping, score
records, analysis snapshot, protocol and checksums independently of the source
checkout. Its `replay.py` verifies inputs and compares all per-query records and
aggregates with the packaged expected result; no database or model is needed.

These are candidate-count diagnostics inside one frozen maximum pool. Only its
endpoint had a matched reranker input token budget. Intermediate prefixes have no new
rankings or generated answers, and their token budgets are not matched. Graph-only
evidence means absent from this recorded direct pool, not inaccessible to every
retriever. A union-coverage gain does not predict answer quality or the result
of fusion under a fixed budget. This analysis makes no model or retrieval calls.

### Search channels of newly supplied evidence

Adding `channel_analysis` with the recorded `depth_limit` to the accessibility protocol
partitions every gold unit in the saved pool from direct retrieval with more candidates
that is absent from the initial pool. `representation_scores` records each non-initial
passage's reciprocal owner rank in body, Q− and Q+ search. The analysis takes the union
of channels across every supporting passage in the budgeted pool, then assigns each unit
to body only, questions only, or both. Initial passages retain their original search
ranks and supply none of these initially missing units; their metadata is not treated as
a measurement at the larger retrieval limit.

The report preserves all seven channel combinations, unit and passage IDs,
recorded ranks, and retention in the saved top ten and full answer context.
Category totals count question–gold-unit pairs; a question can contain several
categories. HotpotQA retains release occurrences and also reports original
question counts. Zero-supply categories have undefined retention rates.
These are descriptive counts of existing outputs. They characterize channel
membership within the saved budgeted pool, without reconstructing the full
unretained search union or a body-only reranking/QA condition. Source hashes
and the recorded evidence sets must agree before results are saved.

### Quality by initial evidence coverage

The same command accepts a protocol with `analysis: "initial_coverage"`.
It reaggregates the saved `pool_report` and saved matched-budget scores,
grouping questions by exact gold-unit counts in the shared initial pool:
none, partial or complete coverage. The protocol's `datasets` specifies each
query file, both `results` paths (`prehop_replay` and `direct_tokens`), eligible
and null-query counts, and score `metrics`. Its `inputs` maps every source
path to its SHA-256 hash; `uncertainty` specifies the seed and resample count.
The protocol is recorded before calculating grouped results, while identifying
the analysis as retrospective to the original experiments.

Each group reports candidate-pool, selected-top-ten and answer-context recall,
saved quality metrics, and paired Prehop-minus-direct differences. All eligible
questions are retained, including terminal failures with zero quality; missing
metrics or inconsistent identities are errors. MultiHop-RAG null questions are
listed separately because their gold recall is undefined. HotpotQA preserves
occurrence weights and clusters by original question within each group. Empty
groups remain undefined; a one-question group is descriptive because bootstrap
resampling cannot estimate between-question variability. `analysis.json`
contains the group assignments and all metric summaries. This analysis makes
no new model or retrieval calls and does not change the main full-population QA.

When comparing evidence gains across datasets, distinguish absolute recall
changes from the initial evidence deficit. If `r_initial` and `r_expanded`
are mean gold-evidence recalls over the same eligible rows, report
`(r_expanded - r_initial) / (1 - r_initial)` as a descriptive fraction of the
initial mean recall deficit recovered. Preserve HotpotQA occurrence weights;
do not substitute a mean of per-question ratios or pooled annotation counts.
A zero denominator leaves this ratio undefined. It is not an answer-quality
metric or evidence of a causal dataset effect. For admission diagnostics,
reachable recall minus admitted recall bounds further recall gains
from that neighborhood; it need not be achievable under the token ceiling.
Compare initial-coverage strata and report recovered and displaced annotations
separately before attributing different effects to corpus or graph structure.

Dataset conditions matter. Initial evidence recall is 66.98% on MultiHop-RAG
and 94.83% on the reduced HotpotQA corpus. Relative to the initial mean recall
deficit, one-step expansion recovers 29.50% and 45.11%, respectively. The smaller
absolute HotpotQA gain does not imply less recovery relative to initially
missing evidence. These are descriptive retrieval ratios, not QA metrics or a
causal comparison of datasets.

<a id="compare-a-common-reader-over-saved-evidence"></a>

## Compare answer generation over saved evidence

Using the same answer-generation model and prompt controls generation differences
and compares the downstream usefulness of each system's exposed passages.
Identical instructions need not be equally optimal for every retrieval output. It never
reruns indexing, search, expansion, evidence selection or embeddings. Preparation
reads every `details[].retrieved_sources` passage in its original order with
common title/page/chunk labels; it adds no passage or token cutoff. Gold and
previous answers are excluded. This replays saved passages through one answer
generator; it does not reproduce either system's benchmark answer pipeline.

Pass one completed result per dataset/system, in the intended execution order:

```bash
"$PYTHON_BIN" -m scripts.prepare_synthesis_inputs \
  "$PREHOP_RESULT" \
  --output data/synthesis/inputs.jsonl
"$PYTHON_BIN" main.py --mode synthesize \
  --trace-inputs data/synthesis/inputs.jsonl \
  --output-dir data/results/common-reader --concurrency 24
```

Add the Naive result path to the preparation command for comparisons.
Groups run sequentially; questions within a group run concurrently. The answer generator
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
only matching messages, generation settings and model reuse successful responses.
Changed-input responses are removed from the current response file; original
retrieval results stay untouched. `generation_complete` means response records
exist, not that answer accuracy or research validity has been established.

Score the generated answers against the same original result files:

```bash
"$PYTHON_BIN" -m scripts.export_official_results "$PREHOP_RESULT" \
  --synthesis-responses data/results/common-reader/responses.jsonl \
  --output-dir data/results/common-reader/scores
```

With HotpotQA, supply its prepared `--hotpot-sentence-store` if it is outside
the default data directory. Original query identities, annotations and retrieval
predictions are retained; source and response hashes identify the new score
artifacts. Incomplete, duplicated or mismatched response populations are rejected by this
offline evaluator; original retrieval failures retain zero quality. A single source writes
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

Answer-generation replay times are separate from original end-to-end query latency.

<a id="v2-fixed-input-analyses"></a>

## Fixed-input analyses

Archived package READMEs and hash manifests describe the outputs bundled with
each package, including historical comparisons outside the final manuscript.
Preserve those files when reanalyzing outputs; current paper scope and appendix
locations are defined by the inventory above.

The depth, admission, graph-construction and inherited-score comparisons are
replays over saved experiment artifacts rather than new benchmark runs. Each
package pins its inputs by hash in a `protocol.json` before measurement and
uses gold annotations only to evaluate the constructed sets.

| Package | Required saved inputs | What is regenerated |
| --- | --- | --- |
| Depth and admission | Saved HOP/NEXT graph, initial passages and scores, query–body similarities, reranking prompt metadata, tokenizer at the pinned revision | Neighborhoods, admission orders, whole-passage prefixes, coverage |
| Graph crossing | Both graphs' neighborhoods, inherited scores and admitted sets | Coverage, shared/unique evidence, count-matched prefixes |
| Common-pool score exchange | Both full-graph score vectors, passage-ID intersection, fixed addition counts | Fusion ranks over the common pool, distance/start-score components |
| Budget, decay, offset and ties | Fixed four-step inputs and each condition's protocol | Candidate coverage only |
| Four-step downstream | Recorded admitted sets, reranking and answer requests and responses of the primary (own-order) and secondary (common cosine order) runs | Scores, paired intervals, prompt-selection exclusions, presentation-order contrast |
| Repeated-fact exclusions | Fixed fact ranking and masks | Reduced-population means and intervals |

Graph construction, embeddings, direct search and model inference are not
regenerated by these packages. The primary four-step downstream run presents
each admitted set in its own admission order, so the rank-fusion set follows the
fused order of the standard one-step pipeline; a secondary run presents both sets
in query-to-body cosine order. The fusion-set contrast between them combines
presentation-order and generation-run differences. Comparisons with the one-step three-policy run pair the
same questions across different generation runs and are reported as post hoc. Intervals use
10,000 paired resamples with seed 42 and original-question clusters; adjusted
families are stated per comparison in the manuscript.

The additional MultiHop-RAG fact-cluster bootstrap links questions sharing an
annotated fact string into connected components, then resamples those components
while retaining all questions in each sampled component. Report pointwise 95%
intervals distinctly from family-adjusted intervals. For graph and score-component
sensitivity, preserve the original separate comparison families: two graph gaps,
three admission gains/interactions, and six score-component contrasts across the
two datasets (including identically zero HotpotQA contrasts). Resample component
sums and counts together so the estimate remains weighted by questions. Source-article
components are dominated by one large component; their intervals are reported
as sensitivity results rather than used as the primary analysis. Fact clustering accounts for that specified dependence;
it neither bounds all dependence nor measures generation or index variability.

## Evaluate saved results

### Dataset-specific official results

Each completed benchmark writes `.official.json` and `.diagnostics.json`
beside its result. To collect final results in one command, pass the explicit
result paths for all methods and both datasets:

```bash
"$PYTHON_BIN" -m scripts.export_official_results \
  path/to/prehop_multihoprag.json path/to/naive_multihoprag.json \
  path/to/prehop_hotpotqa.json path/to/naive_hotpotqa.json \
  --output-dir data/results/final-comparison
```

Select one final source per dataset and method; the command does not discover
the newest run.
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

<a id="fixed-candidate-final-selection"></a>

## Compare ranking methods with fixed candidates

[The ranking comparison](../scripts/compare_prehop_selectors.py) compares the
recorded LLM ranking against score-based orders over the same complete
candidate pool. It uses the recorded selection limit and produces all five
common retrieval measures, condition means and paired differences from LLM
reranking, with original-question cluster bootstrap intervals.
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

The default comparison includes the recorded LLM order, fusion of
query-to-body similarity and retrieval-rank scores, and retrieval-rank order. The latter sums
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
insufficient. Offline fused and retrieval-rank conditions need no database.

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
| Index wall time | Original index pipeline, including waiting/retries; amortized over manifest sources. Post-timer reporting/capacity collection is separate. |
| Query batch wall time | Full-batch dispatch to last answer/terminal failure, including queues and interleaved checkpoints; excludes initialization/trailing reports. Dividing by questions gives inverse throughput. |
| Query latency | Individual response time; inspect service, worker queue and wall-latency fields separately. It is not batch time divided by questions. |
| Benchmark segment wall time | Accumulated checkpointed segments, including setup/checkpoints. Resumed segments are not uninterrupted throughput. |
| Reuse preparation | Separate from original cold construction and prior failed attempts. |
| Storage | Neo4j logical payload estimates and file-backed physical bytes are different measures. Trace bytes are separate; trace I/O remains in phase time. |

Partial, target-failed or resumed batches do not supply continuous throughput.
Fully executed batches retain terminal query failures. Missing usage and cost
are unavailable, not zero. Shared answer-generation replay times are separate
from measured online end-to-end latency.

The paper's Prehop query latency was remeasured on every benchmark query using
the existing indexes and eight concurrent queries. The dense baseline retains
its original timing run. The paper reports available full indexing-phase
measurements. These are descriptive measurements under uncontrolled serving
loads, not a matched efficiency comparison.

## Managed campaigns and additional analyses

For an index-only batch after README setup:

```bash
"$PYTHON_BIN" scripts/index_matrix.py launch indexing-01
```

It owns a detached session and writes `data/results/<campaign>/index-supervisor/`
plan/status and named logs. Smoke/full indexes run sequentially over registry
targets. Smoke failure isolates a target; index completion does not run full
query benchmarks. Complete receipts can supply `core/index_reuse.py` version-2
links; the legacy one-query matrix uses version 1. A link reuses the source
Neo4j namespace in place and records its original index identity and timing.

`plan_link_experiments.py` and `link_experiment_campaign.py` handle dependencies,
explicit concurrency and adoption of owned processes for the primary Prehop and
Naive runs and the HOP/NEXT expansion conditions.
Use their `--help` for plan/launch arguments. Resume retains recorded
completed/failed tasks, adopting live owned processes instead of silently
retrying; failed dependencies remain explicit.

Supervisors record PID/start/boot identity and clean only verified descendants
with bounded TERM waits. Surviving owned processes block restart. Heartbeats
show liveness, not completed source work. Reboot recovery is not automatic.
The separate `paper_campaign.py`/`run_paper_matrix.sh` workflow records executed
stages and their outputs in run-local files. Preserve earlier statistics and
attempt costs.

[Link usefulness](../scripts/analyze_ablation_links.py) measures evidence added
beyond direct retrieval and retained after reranking.
`analyze_expansion_factorial.py` computes conditional HOP/NEXT effects and
additive interaction from saved conditions. All retain explicit source identities
and analysis denominators; each exposes required inputs through `--help`.
