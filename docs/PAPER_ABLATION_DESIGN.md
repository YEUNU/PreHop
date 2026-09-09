# Prehop Representation Ablations

This document defines three experimental conditions and two comparisons. It is
an experiment specification, not a record of measured performance. Manuscript
results must come from completed, compatible benchmark artifacts.

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
- Disable query rewriting and refinement. Question-specific rewrites must not
  enter a body-only condition through supplemental body searches.
- A retrieves up to 12 candidates per channel; B and C retrieve up to 12 from
  the body channel. Final evidence budgets match, but initial candidate counts
  and search work do not. A–B does not establish superiority at equal compute.

These settings differ from primary Prehop's Q+-owner activation, bridge scoring,
and query rewriting. Run A again under the shared controls; do not copy primary
benchmark scores into its result row. Primary defaults remain unchanged.

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
| MuSiQue | Answer EM and Answer F1 | Global paragraph Support P/R/F1 |

MultiHop-RAG QA Precision, Recall, F1, and Accuracy coincide under the decision
rule used by the repository. Global-corpus MuSiQue support evaluation is a task
adaptation, not the official question-local candidate protocol. Metric
implementations are in `utils/metrics.py`; result fields and denominators are
recorded by `cli/benchmark.py`.

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
