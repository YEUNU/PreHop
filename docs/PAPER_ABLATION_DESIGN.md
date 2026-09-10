# Prehop Ablation and Link-Analysis Design

This document defines the A/B/C representation comparisons, link-usefulness
analyses, and a separate precomputed-versus-online matching comparison. It is
an experiment specification, not a record of measured performance. Manuscript
results must come from completed, compatible benchmark artifacts.

The manuscript places the design in Section 5.4 (Table 5) and reserves
unmeasured outcomes in Tables 6–7. Result cells remain `—` until compatible
full-query runs complete. Figures 1–2 describe the revised single-pass Prehop pipeline,
not the common A/B/C activation and scoring policy below.

## Conditions and research questions

| Condition | Index-time passage connections | Starting-passage retrieval |
|---|---|---|
| A: `question_full` | Q+ to Q− matching | Body + Q− + Q+ |
| B: `question_body` | The same frozen nodes, embeddings, and edges as A | Body only |
| C: `body_body` | Body-to-body similarity | Body only |

**A–B: Do question-search channels improve final evidence retrieval on a fixed
graph?** B does not delete questions or rebuild connections. It searches only
passage bodies, then follows stored connections from the retrieved passages.

**A–C: Is question-based linking and search useful compared with body-based
linking and search?** C neither generates nor stores questions. Both connection
construction and starting-passage retrieval change, so this comparison measures
the joint design rather than an isolated indexing effect.

HOP removal, activation-policy comparisons, and ranker replacement are outside
these two comparisons.

## Shared controls

- Use the same corpus, six-sentence passage boundaries, body embedding model and
  revision, query population, and answer-generation model.
- Every retrieved starting passage exposes its stored outgoing HOP links.
  This does not mean traversing all passages in the corpus. Keep NEXT expansion
  and graph depth one in every condition.
- Set `HOP_SEED_POLICY=all`. Graph-only candidates inherit the starting passage's
  total representation score multiplied by 0.5.
- Use `HOP_SEMANTIC_VARIANT=body_only`, since C has no question vectors. Keep
  the existing final LLM selection and the 12-passage output budget.
- Use the original question and single-pass retrieval in every condition. Initial
  rewriting and evidence-conditioned refinement have been removed from Prehop.
- A retrieves up to 12 candidates per channel; B and C retrieve up to 12 from
  the body channel. Final evidence budgets match, but initial candidate counts
  and search work do not. A–B does not establish superiority at equal compute.

These settings differ from primary Prehop's Q+-owner activation and bridge
scoring. Run A under the shared controls; do not copy primary benchmark scores
into its result row. Original-question single-pass retrieval is shared with
primary Prehop.

## Body-link construction

Use a completed compatible question-link graph as a frozen reference. Export each
passage's body/embedding digest and outgoing HOP degree. Neither evaluation
queries nor gold evidence participate in this export or in link construction.

Build C in a fresh namespace. Verify identical passage IDs, source IDs, titles,
text, and body embeddings before writing links. Query the body ANN index using
each passage's body vector, exclude its source document, and select distinct
destinations by similarity. Match A's outgoing degree for each passage, including
zero-degree passages. Fail if the requested number of eligible neighbors cannot
be obtained. C's degree budget therefore depends on A; it is not an independently
optimized body baseline.

The ANN request size is the source document's passage count plus the required
outgoing degree, capped by corpus size. Approximate retrieval does not guarantee
exact global nearest neighbors. `--clone-body-from` copies stored Document/Chunk
properties and CONTAINS/NEXT structure to a fresh namespace, without questions
or existing HOP edges, then builds body links. It avoids new generation and
embedding requests. Fresh body indexing is also supported, but recomputed
embeddings must match the reference; do not disable the identity check.

## Evaluation and reporting

Use identical query IDs within each paired comparison. Report retrieval and QA
populations separately and retain failure accounting. Report A–B and A–C score
differences with paired query bootstrap intervals.

| Benchmark | Official scoring rules | Additional measures |
|---|---|---|
| MultiHop-RAG | Hits@4, Hits@10, MRR@10, MAP@10, QA Accuracy | Exact-fact recall, AllFacts@10 |
| HotpotQA fullwiki | Answer EM/F1, Supporting Fact EM/F1, Joint EM/F1 | Separately labelled sentence coverage and document recall |

MultiHop-RAG QA Precision, Recall, F1, and Accuracy coincide under the decision
rule used by the repository. HotpotQA uses the [official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
and the released fullwiki corpus. The common adapter projects complete retrieved sentences to original article
titles and sentence indices without gold labels. Official support scores apply
to this predicted set; passage-level coverage is a separate diagnostic. The
same adapter must be used in every arm. Scorer integration and the complete
prepared corpus are verified. All original labels are retained, including the
one unavailable upstream sentence label documented in
[HOTPOTQA_FULLWIKI](HOTPOTQA_FULLWIKI.md). Ablation results remain unmeasured;
see [Appendix C](prehop_paper.md#appendix-c-hotpotqa-fullwiki-evaluation-protocol).

Hits@k alone does not establish complete evidence recovery or correct reasoning
transitions. Use fact recall and AllFacts for claims about evidence completeness.
Separate index time from query latency, and report question-generation calls and
tokens, embedding work, link-construction time, degrees, and candidate counts.
Reference export, body cloning, and verification are experimental preparation
costs. The clone artifact records `body_reuse` and source costs separately; its
clone-and-link wall time must not be compared with a full cold index build. Measure
query times under matched serving and concurrency conditions; do not equate
shared-resource measurements with exclusive execution.

## Running the experiments

`scripts/prehop_ablation.py` prints a plan by default. Add `--execute` to perform
service work. Use explicit gateway, model, revision, and Neo4j settings from the
execution environment. The script does not print credentials. Source edits do
not require a separate worktree or restore source-change/final-validation gates.
Use distinct index namespaces and output directories to preserve other runs.

Select `PYTHON_BIN` for the prepared main environment. The following commands
print plans; replace input paths and namespaces with the exact source artifact.

```bash
# A: reuse a completed question graph without changing it.
"$PYTHON_BIN" scripts/prehop_ablation.py --mode benchmark --profile question_full \
  --reuse-existing-index --namespace SOURCE_NAMESPACE --run-id mhr_A_v1 \
  --corpus-tag multihoprag --dataset /absolute/path/to/corpus \
  --queries /absolute/path/to/queries.json --index-stats /absolute/path/to/question_index_stats.json

# C: copy frozen bodies, export a new degree reference, and build body links.
"$PYTHON_BIN" scripts/prehop_ablation.py --mode index --profile body_body \
  --clone-body-from /absolute/path/to/question_index_stats.json \
  --namespace ablation_mhr_body_v1 --run-id mhr_body_index_v1 \
  --corpus-tag multihoprag --dataset /absolute/path/to/corpus \
  --queries /absolute/path/to/queries.json --reference /absolute/path/to/new_reference.json

# C query evaluation uses the new body's completed index artifact.
"$PYTHON_BIN" scripts/prehop_ablation.py --mode benchmark --profile body_body \
  --namespace ablation_mhr_body_v1 --run-id mhr_C_v1 \
  --corpus-tag multihoprag --dataset /absolute/path/to/corpus \
  --queries /absolute/path/to/queries.json --index-stats /absolute/path/to/body_index_stats.json \
  --reference /absolute/path/to/new_reference.json
```

For B, change A's profile to `question_body` and use a new run ID, keeping the
same namespace and index-stats file. `--reuse-existing-index` supports only A/B
benchmarking; the source remains read-only. The launcher clears the LLM generation
seed. A completed index-stat file supplies the original index run identity,
while the requested run ID selects a fresh benchmark output directory.

The clone command requires a **new reference filename**: it exports the source
reference itself. Do not run `export-reference` to that file beforehand. To build
C from raw corpus input instead, first use `--mode export-reference` with the
source namespace and a new `--reference` file, then use C's index command without
`--clone-body-from`. To build a new question graph, use A in index mode with a
fresh `ablation_` namespace and omit reuse/index-stats options.

Repeat each comparison on the same query IDs for each dataset. Results use
`data/results/ablations/<run-id>/<profile>/`. Existing output directories and
reference files are not overwritten. The launcher does not provide resume;
restart interrupted indexing with a fresh namespace and run ID. Indexing and
benchmarking need different run IDs.

Result metadata records `prehop-representation-ablation-v1`, the profile,
connection and activation policies, and the body reference SHA-256 where
applicable. Corpus/index fingerprints, index-policy hashes, and resume
compatibility remain distinct from primary paper-policy validation. An ablation
index is not promoted to a primary canonical index.

## Link-usefulness analysis accompanying A/B/C

This analysis supplements the representation ablations; it does not add a
HOP-off condition. All nodes, links, and retrieval controls remain unchanged.
It measures whether activated links supply useful destinations, separately
from the final selected evidence. No results are available yet.

For each query, let D be the distinct directly retrieved passages before graph
expansion, including supplementary direct candidates if present. Let H be the
distinct destinations reached through HOP; exclude NEXT-only destinations.
Let U = H minus D. Record NEXT provenance separately when a destination is
reached by both edge types. Let F(X) denote matched gold facts on MultiHop-RAG
or gold (article title, sentence index) pairs on HotpotQA. For passage-level
coverage, map contained sentences to their original identities. This diagnostic
is distinct from scoring the explicitly predicted supporting-fact set.

Report the following per condition and benchmark:

- **HOP destination relevance:** query-macro mean of the fraction of passages
  in H that match at least one gold evidence item. Report the number of queries
  with nonempty H; empty H has no destination-precision denominator.
- **Added gold coverage:** query-macro mean of
  |F(D union H) minus F(D)| / |G| over every evidence-bearing query, including
  zero-gain queries. This describes expansion before ranking, not equal-budget
  retrieval or a causal HOP contribution.
- **Retained added coverage:** the same numerator restricted to HOP-added
  passages in the final selected set. Report overlap with NEXT so credit is
  not assigned exclusively to HOP when both routes reach the same passage.
- **Expansion size:** mean distinct HOP destinations per query, including zeros,
  alongside added coverage to expose the effect of candidate breadth.

| Dataset | Condition | HOP destination relevance (%) | Added gold coverage (%p) | Retained added coverage (%p) | Mean HOP destinations |
|---|---|---|---|---|---|
| MultiHop-RAG | A | — | — | — | — |
| MultiHop-RAG | B | — | — | — | — |
| MultiHop-RAG | C | — | — | — | — |
| HotpotQA fullwiki | A | — | — | — | — |
| HotpotQA fullwiki | B | — | — | — | — |
| HotpotQA fullwiki | C | — | — | — | — |

Benchmark usefulness does not establish semantic correctness of every stored
link. Separately sample 100 unique stored edges per condition using a fixed
random seed, or all edges if fewer exist. A and B share the same sample because
their graph is identical. Two annotators, blinded to benchmark retrieval scores,
judge whether the destination supplies information sought by the source Q+
(for question links) and whether the two passages form a meaningful evidence
transition (for both graph types). Allow unclear cases; report counts, agreement,
and adjudicated labels. Inspect every matched question pair retained by a sampled
passage edge. This index-wide sample complements query-conditioned utility and
addresses threshold-free matching without claiming a link-quality effect from
aggregate Hits or MAP alone. Gold evidence is used only for analysis.

## Connection-timing ablation: stored lookup versus online matching

This experiment tests whether resolving destinations during indexing
reduces query-time work. It is separate from A/B/C: both timing conditions use
A's question representations and common retrieval policy, including disabled
initial rewriting; neither uses evidence-conditioned re-search. Neither condition regenerates questions or embeddings.
The runtime supports both arms through `scripts/prehop_connection_timing.py`
and `scripts/prehop_ablation.py`. Completed paper results are not yet available.

| Condition | Destination resolution | Query-time operation |
|---|---|---|
| T-precomputed | Resolve every outgoing question before evaluation | Read stored destinations for the activated starting passages |
| T-online | Resolve activated outgoing questions during the query | Match the same stored Q+ vectors against the same Q− index |

### Matched conditions

Use one frozen corpus, passage segmentation, Q−/Q+ set, embeddings, and ANN
index snapshot. Construct T-precomputed's links with the same resolver that
T-online invokes. Match ANN candidate budgets, source-file exclusion, similarity
rules, owner mapping, duplicate merging, and tie handling. Do not replace ANN
with exact search in only one arm. Historical stored links are unsuitable if
they cannot be reproduced under this matching configuration.

For a pair of runs, hold query IDs, direct search, starting passages, activation,
NEXT traversal, one-step expansion, candidate scoring, final LLM selection,
answer generation, and generation settings fixed. T-online must not read the
stored HOP destinations. It must return the same destination metadata needed by
subsequent processing. Time destination lookup or matching through completion
of that comparable output; include all work required to obtain it.

### Measurements

1. **Paired connection-stage replay.** Save starting passages from the common
   direct search and supply the same activations to both arms. Measure elapsed
   connection-stage time, destination-set agreement per start, and the proportion
   of starts with identical destinations. Include zero-link starts and report
   the number of matches requested. This isolates connection handling from LLM
   variation without using benchmark gold evidence to select starting passages.
2. **End-to-end runs.** Evaluate the same full query set with each arm and report
   mean query latency and the benchmark's primary retrieval metrics. Also report
   the paired latency difference with a query-bootstrap interval. A faster
   connection stage need not imply a large end-to-end gain. If destinations
   differ, report the agreement and retrieval differences; do not present the
   entire elapsed-time difference as a pure computation-placement effect.
3. **Preparation cost.** Report the precomputed link build separately from
   shared question generation and embedding costs. Precomputation moves work
   earlier; it does not make that work free. For a fixed workload, total-cost
   interpretation includes the added preparation time and the number of queries.

Keep serving concurrency and resource allocation matched and alternate arm
order across repeated paired measurements. Prevent contention from other
benchmark jobs from being mistaken for an arm effect. Do not interrupt those
jobs to run this experiment. Record repetition counts and variability.

The primary comparison disables cross-query destination memoization in
T-online; repeated requests within a query use the same deduplication policy
in both arms. Ordinary database/ANN warm-up is matched. If online memoization
is evaluated, report it as a separate cached condition, with first-use and
reused-destination timings and cache-hit rate; do not blend it with uncached
online matching or assume a permanently cold database.

| Dataset | Condition | Connection-stage mean (ms) | End-to-end mean (s) | MAP@10 | AllFacts@10 (%) | Supporting Fact F1 (%) |
|---|---|---|---|---|---|---|
| MultiHop-RAG | T-precomputed | — | — | — | — | N/A |
| MultiHop-RAG | T-online | — | — | — | — | N/A |
| HotpotQA fullwiki | T-precomputed | — | — | N/A | N/A | — |
| HotpotQA fullwiki | T-online | — | — | N/A | N/A | — |

| Dataset | Identical destination sets (%) | Paired connection-time saving (ms), 95% CI | Paired query-time saving (s), 95% CI | Link preparation time (s) |
|---|---|---|---|---|
| MultiHop-RAG | — | — | — | — |
| HotpotQA fullwiki | — | — | — | — |

Time saving is online minus precomputed; negative values indicate no saving
for that measurement. `—` is unmeasured; `N/A` is a metric not used for that
dataset. Use all full-benchmark queries for timing and the original eligible
populations for quality metrics. The experiment supports claims about replacing
online destination matching under this fixed retrieval policy, not a universal
speedup over every GraphRAG implementation.

## Revised query procedure and existing evidence

### Analysis tools

`scripts/analyze_ablation_links.py --result RESULT --events EVENTS --output NEW_JSON`
reads content-hashed traversal payloads. Successful queries require exactly one
event, including queries with no starting passages. Failures remain counted and
are reported separately from conditional utility means; report those
denominators with the means. The tool does not infer HOP utility from aggregate
retrieval scores.

For paired connection-stage replay, first export the starting passages from a
completed reference A run using `scripts/export_ablation_activations.py` with
the same `--result`, `--events` and `--output` arguments. Failed reference queries
are listed in its report; a replay with exclusions must disclose them and does
not establish an all-query timing result. The replay never reruns retrieval to
choose different starts for either arm.

Build a separate destination store with
`scripts/prehop_connection_timing.py --mode build --index-stats STATS --store NEW_SQLITE --output NEW_JSON --execute`.
Then use `--mode replay`, the same store and index statistics, plus
`--activations ACTIVATIONS_JSON`. Set `--repetitions` and `--warmups` explicitly
for reported runs. Both arms use the same resolver and destination hydration;
online matching has no cross-query destination cache. Build reports separate
connection construction from graph verification time. End-to-end runs use
`scripts/prehop_ablation.py --connection-timing precomputed|online --timing-store STORE`
with profile `question_full` and the same frozen source index. These are
ablation-only modes, not additional primary query pipelines.

`scripts/export_ablation_edge_sample.py --namespace NAMESPACE --output NEW_DIRECTORY --execute`
creates the blinded edge sample and two blank annotation CSVs. After independent
labeling and adjudication, `scripts/analyze_ablation_annotations.py --sample SAMPLE_JSON --annotator-1 CSV1 --annotator-2 CSV2 --adjudicated FINAL_CSV --output NEW_JSON`
reports label counts, raw agreement, Cohen's kappa and adjudicated counts. It
requires all sampled IDs and explicit labels; it never generates human labels.

### Query policy

The revised method removes initial query rewriting and evidence-conditioned
question regeneration/re-search. All input lengths use the original query.
A/B/C and both connection-timing arms use one retrieval pass. There is no
query-rewrite prompt, input-length gate, or follow-up-search configuration. Their agreed controls are otherwise unchanged. Existing
Prehop benchmark scores include the earlier loop and must not populate revised
method or ablation results. New full-run results remain pending.
