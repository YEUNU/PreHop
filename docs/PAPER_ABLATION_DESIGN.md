# Prehop ablation and connection experiments

This specification defines the controlled experiments and their interpretation.
It does not contain measured performance claims. Completed evidence belongs in
[RESULTS](RESULTS.md); machine-readable execution state belongs under
`data/results/`, not in a separate progress document.

## Questions and scope

| Experiment | Question | Primary measurement |
|---|---|---|
| Stored versus online matching | Does moving destination resolution to indexing reduce connection-stage work? | Paired connection-stage time saving |
| Co-evidence connectivity | Does the graph connect passages containing the evidence required by a question? | Directed evidence-document pair coverage |
| B–C | Does question-based destination selection improve retrieval over body-based selection? | MultiHop-RAG MAP@10; HotpotQA Supporting Fact F1 |
| A–B | What is the effect of adding question search channels on the fixed question graph? | Retrieval and QA differences, secondary |
| A–C | What is the joint effect of question search and question linking? | Retrieval and QA differences, secondary |

Co-evidence connectivity is an automatically scored structural diagnostic. Gold
annotations specify required evidence, not the direction of a reasoning chain.
Neither an edge between two gold passages nor an aggregate retrieval improvement
establishes the semantic correctness of every generated question or edge.

## Data and inference units

Use all 2,556 MultiHop-RAG query rows and the same 609-document corpus in every
arm. Queries without gold evidence remain in QA evaluation and failure counts;
they have no evidence-retrieval denominator.

HotpotQA uses the original HippoRAG v1 release: 9,221 passages and 1,000 query
rows. These rows contain 944 unique original question IDs. Preserve every
released row, attach a unique occurrence ID, and retain `original_query_id`.
This is a pooled supporting-and-distractor retrieval corpus, not official
fullwiki and not a per-question ten-passage task. See [HOTPOTQA](HOTPOTQA.md) for
the pinned source and preparation command.

Pair all contrasts by occurrence ID. Report release-row means and a separate
unique-original-question macro mean. Resample original-question clusters,
retaining all occurrences within each sampled cluster. For repeated timing,
first average the repetitions for each occurrence, then resample original
questions. Use 10,000 bootstrap draws and seed 42 for analysis only. Generation
remains unseeded; `seed_42` output directories are legacy run labels.

Intervals are pointwise descriptive intervals. The primary endpoints above
are chosen before observing these experiments. Other metrics and question-type
breakdowns are secondary; do not present many unadjusted intervals as an
omnibus significance test. These intervals capture query-population variation,
not variation across independently regenerated graphs or all LLM realizations.

## Representation arms

| Arm | Stored HOP destinations | Direct search |
|---|---|---|
| A: `question_full` | Existing frozen Q+ → Q− links | Body + Q− + Q+ |
| B: `question_body` | The same links as A | Body only |
| C: `body_body` | Body similarity, matched outgoing degree | Body only |

All arms use original-question single-pass retrieval, six-sentence passage
boundaries, the same body embeddings and generation model, NEXT expansion,
one-step graph depth, `HOP_SEED_POLICY=all`, body-only graph semantic scoring,
path decay 0.5, the same final selection policy, and a 12-passage output budget.
The exact configuration is owned by `core/prehop_ablation.py`.

A is a controlled reference, not the primary Prehop result. Primary Prehop's
Q+-owner activation and question-aware bridge scoring differ from these common
controls. Do not copy primary scores into A. A retrieves up to 12 candidates
per enabled channel; B/C use one channel. Consequently A–B changes candidate
breadth and representation scoring as well as access to question vectors. It
is a channel-configuration contrast, not an equal-compute comparison or a pure
embedding-quality intervention.

### Frozen direct search for B–C

`scripts/prepare_ablation_inputs.py` runs direct retrieval once and stores the
query vector, candidate passages, scores and original retrieval time. B and C
read independent copies of exactly this body-search input. A uses a separate
full-channel input prepared from the same source graph. Duplicate query texts
share one frozen retrieval realization; every release occurrence is evaluated.
No gold labels participate in preparation.

Frozen-prefix benchmark latency is explicitly
`frozen_prefix_downstream_only`: it includes loading and downstream processing,
not freshly performing the common retrieval prefix. It is not primary
end-to-end latency. Common retrieval preparation is recorded separately.
Natural end-to-end timing is measured by the timing arms below without this
input replay.

### Body-link preparation

B reuses the completed question graph without modifying it. C clones Document
and Chunk properties, stored body vectors, CONTAINS and NEXT into a distinct
Neo4j namespace. It does not regenerate questions or embeddings. C then queries
the body ANN index, excludes the source document, and selects distinct
similarity-ranked destinations using the outgoing degree of each B passage,
including degree zero. The ANN candidate request is source-document passage
count plus requested degree, capped by corpus size; ties use destination ID.

The clone artifact records body identity observations, requested/actual edge
counts, and every degree mismatch. These are experimental observations, not
execution gates. If degrees differ, report this and qualify the intended
matched-degree contrast. C's degree budget comes from B; the experiment tests
destination selection under a shared budget, not an independently optimized
body baseline. Its separate ANN index can also introduce approximate-search
variation; the frozen direct prefix removes this confound from starting search,
not from the intended link-construction intervention.

Clone-and-link wall time is an incremental preparation cost. Never compare it
with another model's cold total indexing cost. Keep the original source build
cost and reused embedding/generation work separate.

## Automatic co-evidence analysis

`scripts/analyze_evidence_connections.py` reads the frozen graph and benchmark
gold after construction. It never supplies gold to the linker or retriever.

Map evidence to all matching passages, grouped by gold document title. On
HotpotQA, a passage must contain a complete gold supporting sentence with its
original title/index identity. On MultiHop-RAG, use the existing official fact
matcher within gold document titles. This analysis is document-group level;
recovering every supporting sentence/fact remains a separate final-evidence
metric. Retain missing mappings as unreachable groups and report their count.
Do not silently exclude them or inject missing gold text into the corpus.

For questions with at least two gold document groups, report:

1. **Any inter-evidence HOP:** whether at least one HOP edge joins two distinct
   evidence groups. This is a weak structural measure, not complete recovery.
2. **Directed group-pair coverage:** fraction of ordered distinct gold-group
   pairs connected by at least one direct HOP. Directions are reported
   symmetrically; the dataset does not prescribe them as reasoning directions.
3. **Oracle-start one-hop group coverage:** average fraction of gold groups
   reachable from a gold-bearing passage through one HOP, including the source.
4. **Oracle-start all-groups rate:** fraction of those starting passages that
   reach every group. Average within question before aggregating questions.

The last two measures deliberately provide an oracle starting passage. Label
them as diagnostics, never actual deployed retrieval. Score HOP, NEXT and their
union separately. Break down bridge and comparison questions on HotpotQA;
comparison questions can require both documents without a directed dependency.

Use 20 seeded random destination realizations as an analysis-only null. Match
each source passage's outgoing degree, exclude its document, and prohibit
duplicate destinations. Do not run additional QA or expose this null as a
product retrieval pipeline. Report the observed connectivity and the null
summaries; density alone can otherwise create apparent co-evidence connectivity.

## Actual-query link usefulness

`scripts/analyze_ablation_links.py` uses explicit traversal traces. Let D be
directly retrieved passages, H distinct HOP destinations, N NEXT destinations,
and S final selected passages. Compute:

- Destination relevance: fraction of H matching any gold fact/sentence; report
  the number of queries with nonempty H.
- Added gold coverage: gold in H minus D that was absent from D, divided by all
  gold evidence units. Include zero-gain queries.
- Retained added coverage: the same gain restricted to S.
- HOP destination count and retained gain also reachable through NEXT.

These quantify observed use, not the causal effect of removing HOP. Gains before
selection can reflect more candidates; always report expansion size and final
fixed-budget results together. Failures remain in benchmark quality accounting;
trace-based usefulness is conditional on successful queries with trace evidence.

## Stored versus online connection timing

Both arms use the same source graph, stored Q+/Q− vectors and Q− ANN index.
`scripts/prehop_connection_timing.py --mode build` invokes the same resolver
used online and writes experiment-scoped `HOP_TIMING` relationships and a
`RAGTimingSnapshot` in Neo4j. A small JSON file identifies this experiment and
records counts/costs. It is not an edge store. Historical primary HOP_ANSWER
relationships are unchanged.

The precomputed arm reads these Neo4j destinations; the online arm recomputes
them from stored Q+ vectors and never reads HOP_TIMING destinations. Both hydrate
the same destination metadata from Neo4j. Include destination lookup/matching
and hydration in the timer. The small experiment pointer is setup outside the
timer. Neither arm generates questions or embeddings; online has no cross-query
destination memoization. Match owner mapping, source exclusion, candidate
budgets, deduplication and tie handling through the shared resolver.

### Paired connection replay

Replay exactly the full-channel starting passages recorded during common
retrieval. Include zero-link starts. Warm up each arm once per query and use
five paired repetitions, alternating arm order. Report mean/p95 connection time,
query-macro paired savings and cluster-bootstrap intervals, per-start destination
identity rate, destination Jaccard and requested question-match count. An empty
pair of destination sets agrees, but report zero-link counts separately so
agreement is not dominated by empty outputs.

Destination disagreement is recorded, not an execution failure. If outputs
change, the elapsed difference is not a pure computation-placement effect.
ANN may be nondeterministic even with the same configuration.

### Natural end-to-end supplement

Run both arms on the complete query population with normal direct retrieval,
not frozen-prefix replay. Use serving concurrency one for these measurements
and two block repetitions in precomputed/online then online/precomputed order.
Report each repetition and query-level latency/quality differences. Check
observed starting-passage and destination agreement from traces as analysis.
Residual temporal drift, ANN variation and unseeded generation remain possible;
connection replay is the primary timing experiment.

The campaign runs timing measurements without other campaign targets. It does
not claim that no unrelated external user can access the same server. Keep the
same queue limit and model settings. Ordinary quality/index work uses at most
two active targets. Timing waits without interrupting an existing target.

Report one-time link-build seconds and edge counts separately from shared
question/embedding costs. A time-based break-even query count is build seconds
divided by positive mean per-query connection saving. If the saving is zero or
negative, no finite break-even is established. This ratio assumes unchanged
corpus and workload; it is not a monetary cost claim.

## Execution and artifacts

The reusable task graph is generated by:

```bash
python scripts/plan_link_experiments.py --campaign RUN_ID \
  --multihoprag-index-stats /absolute/path/to/completed_prehop_index.json
nohup python -u scripts/link_experiment_campaign.py run \
  data/results/RUN_ID/plan.json > data/results/RUN_ID/controller.log 2>&1 &
```

Use the configured shared inference environment. Credentials are never part of
the public plan. The task graph includes A/B/C, C body cloning, common retrieval,
connectivity/null analysis, actual-query utility, paired comparisons, matched
Neo4j link construction, replay, and the end-to-end timing supplement for both
datasets. It also continues the primary HotpotQA index/benchmark matrix.

`--adopt` accepts an explicit JSON list identifying already running work and its
completion artifact. It does not restart an adopted model. Dependencies advance
on recorded process completion; actual failures and blocked dependents remain
visible. No paper approval, source-change, hash-match or schema gate is added.

Benchmark artifacts live under `data/results/ablations/<run-id>/<profile>/`.
Shared preparations, comparisons, task state and logs live under
`data/results/<campaign>/`. Each task has its own output and graph namespace.
Results are unmeasured until the corresponding output finishes; a generated
plan or a successful service probe is not a paper result.
