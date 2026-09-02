# Prehop: Offline Question Links for Multi-Hop Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Prehop is a GraphRAG retrieval system that constructs inspectable,
question-level chunk links during indexing. At query time, it refines the
question by evidence role, expands stored neighbors, selects twelve candidate
paragraphs with the configured generation model, and synthesizes the answer.

Each document has one role:

| Document | Role |
|---|---|
| `README.md` | Public overview, setup, and command guide |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Normative implementation, evaluation, and component-control specification |
| [RESULTS](docs/RESULTS.md) | Canonical complete-result and artifact register |
| [CHANGELOG](docs/CHANGELOG.md) | Chronological engineering record |
| `CLAUDE.md` | Maintainer and experiment-operation policy |

The gitignored `docs/prehop_paper.md` is the research manuscript, and the
gitignored `SUBMISSION_TARGET.md` contains private submission logistics.

---

## What this repository is

Core indexing-time design, currently evaluated on MultiHop-RAG and MuSiQue:

1. Every chunk receives separate hypothetical questions for facts it answers
   ($Q^-$) and information it still requires ($Q^+$).
2. Each Q+ retrieves the closest cross-document Q-. The Q-'s owner chunk is
   the HOP target candidate, so its body follows from graph ownership without
   a second body search. When Q- is disabled for ablation, Q+ retrieves a body
   candidate directly.

Chunking is fixed-size (page-scoped sentence windows) — see
[ARCHITECTURE](docs/ARCHITECTURE.md#shared-input-contract) for details.

For questions of at most 32 words, the query path first produces bounded Q⁻
and Q⁺ retrieval views; longer questions retain their original explicit
constraints. Retrieved evidence can produce additional non-duplicate role
views. Refinement stops when no new role view or selected chunk appears. The
body channel always uses the original question. Retrieval searches Q⁻/body
direct evidence and Q⁺ dependency seeds in parallel, applies reciprocal-rank
fusion of representation and semantic orders, and performs deterministic
batched 1-hop traversal over bidirectional `NEXT` and outgoing `HOP_ANSWER`
edges. One complete-list call ranks the resulting candidate union, and the
first 12 paragraphs feed one synthesis call. Rank evidence propagated over a
graph edge is attenuated by reciprocal path length, so indirect evidence
receives less weight than evidence from a directly retrieved owner.
Q+ retrieval retains the exact matched question-node IDs when results collapse
to owner chunks. The default activates every stored HOP provenance item on a
matched Q+ owner; reciprocal filtering and exact-ID activation remain
ablations.
Within each vector or full-text modality, backend scores establish only that
modality's deterministic rank with a stable node-ID tie break; raw scores are
never mixed across modalities.

---

## Evaluation

Evaluation is dataset-specific. MultiHop-RAG reports official-compatible
Hits@k, MRR@10, and MAP@10 plus separate fact-coverage and null-refusal
diagnostics. MuSiQue reports answer EM/F1 and supporting-paragraph metrics.
The in-repo Naive RAG is a controlled vector-search baseline. It uses the same
six-sentence chunks, top-k 12, and final answer prompt as Prehop, but does not
index question representations or construct and traverse graph edges. This
isolates Prehop's retrieval architecture instead of adding a chunk-size or
evidence-budget difference to the comparison.
The optional LLM judge is disabled by default and is not a primary metric.
Only complete prepared-split runs are eligible for submission results.

Prehop, Naive RAG, HopRAG, BrowseNet, and PropRAG attach an explicit answer
boundary before evaluation. MS GraphRAG requests the equivalent
`Final Answer:` contract from its official search API. The evaluator scores
the complete marked span; an unmarked response remains the complete prediction
and is never replaced by a fixed-length suffix.

### Result admission

The evaluator supports six independent strategies: Prehop, Naive RAG, HopRAG,
MS GraphRAG, BrowseNet, and PropRAG. Each target uses the full prepared
MultiHop-RAG or MuSiQue split. Generation and answer synthesis use
`gemma-4-31b-it`; semantic retrieval embeddings use the configured
`qwen3-embedding-8b` endpoint. MultiHop-RAG and MuSiQue remain in separate
tables because their metrics and denominators differ. The artifact-admission
and publication checks are defined in [RESULTS](docs/RESULTS.md).

---

## Repository layout

```
prehop/
├── main.py                          # single CLI entry point (index/benchmark/maintenance)
├── cli/
│   ├── index.py                     # indexing runner
│   └── benchmark.py                 # benchmark runner (single + multi-seed)
├── core/
│   ├── config.py                    # RAGConfig — validated env-driven settings
│   ├── neo4j_service.py             # async Neo4j driver lifecycle
│   └── vllm_client.py               # external generation/embedding clients
├── models/
│   ├── prehop/                     # the paper's system
│   │   ├── graphrag.py              # GraphRAG facade; run_workflow() is the query entry point
│   │   ├── indexing/                 # chunking (fixed-size), knowledge_mapping (Q-/Q+), hop_edges, graph_writer
│   │   └── retrieval/                # hybrid (RRF), cosine ordering, deterministic traversal
│   ├── naive/                       # baseline (shared fixed-window chunks + vector search)
│   ├── hoprag/                      # baseline (runtime hop traversal via official HopRAG)
│   ├── ms_graphrag/                 # baseline (community-report retrieval via graphrag package)
│   ├── browsenet/                   # pinned BrowseNet reference adapter
│   └── proprag/                     # pinned PropRAG reference adapter
├── utils/
│   ├── abstain.py                   # honest-abstain detection + shared 3-way answer_label
│   ├── metrics.py                   # deferred Batch judge + retrieval metrics
│   ├── batch_judge.py               # OpenAI Batch submit/poll/reconcile support
│   ├── similarity.py                # cosine similarity for final candidate ordering
│   ├── prompts/                     # indexing, shared synthesis, and judge prompts
│   └── io.py / formatters.py / parsers.py / reporting.py
├── data/                            # prepared datasets and generated local indices
├── scripts/
│   ├── datasets/                    # dataset download, normalization, and sampling
│   └── *.py                         # experiment measurement and evaluation utilities
├── tests/                           # chunking / retrieval / live-integration
├── run_servers.sh                   # validate/start Neo4j + generation/embedding endpoints
├── run_index.sh / run_benchmark.sh  # low-level, dataset-agnostic
├── run_multihoprag.sh               # per-dataset entry: index|benchmark|all
├── run_dataset.sh                   # per-dataset entry for MuSiQue
├── pyproject.toml                   # canonical dependency list (uv-managed)
└── README.md
```

---

## Installation

```bash
# Python 3.12+ (pinned in .python-version). The env is managed with uv.
uv sync --locked

# Neo4j version used by the paper runs:
docker run -d --name prehop-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> neo4j:5.26.21-community

# Configure env vars
cp .env.example .env
# Required: NEO4J_PASSWORD and the external generation/embedding endpoint settings
```

`pyproject.toml` and `uv.lock` are the dependency contract, including the
HopRAG spaCy model. The run scripts
auto-discover `.venv/bin/python` (override with `PYTHON_BIN`), so you do not
need to activate the environment.

`run_servers.sh` validates the configured external generation and embedding
endpoints. It never launches local model processes.

The current cold-run configuration uses the following model identities. The
same names are copied to `RAG_GENERATION_REVISION` and
`RAG_EMBEDDING_REVISION` so index and benchmark artifacts retain them directly.

| Role | Served model | Required setting |
|---|---|---|
| Generation and synthesis | `gemma-4-31b-it` | `VLLM_SERVED_MODEL_NAME` |
| Embeddings | `qwen3-embedding-8b`, 4,096 dimensions | `VLLM_SERVED_EMBED_MODEL_NAME` |

Result tables retain the model identity stored in their cited artifacts. A
runtime configuration change does not relabel an earlier result.

Prehop and MS GraphRAG generation calls use temperature 0. HopRAG retains its
upstream indexing temperature 0.1 and retrieval-time node judgement. External
server hardware and launch options are not part of the reported method
configuration.

### BrowseNet and PropRAG runtimes

BrowseNet and PropRAG run in isolated Python 3.10 environments because their
official dependencies conflict with the main project environment. Install the
pinned official revisions once:

```bash
./scripts/setup_official_baselines.sh
```

The setup keeps official source, model dependencies, and downloaded weights
under the ignored `data/official_baselines/` directory. The repository stores
only the adapters and exact upstream commit identifiers. Generation and
semantic embeddings use the configured LiteLLM endpoints; BrowseNet's GLiNER
and ColBERT components remain local because they are algorithm-specific rather
than general generation or embedding endpoints. Run either strategy through
the same dataset entrypoint used by the other baselines:

```bash
./run_dataset.sh musique all --model browsenet --queries full
./run_dataset.sh musique all --model proprag --queries full
```

MuSiQue uses BrowseNet's native question-decomposition template. BrowseNet has
no MultiHop-RAG template, so that dataset uses its official HotpotQA template;
the upstream graph retrieval procedure and search budgets remain unchanged.

---

## Quick start

```bash
# 0) Prepare a dataset (downloads + builds corpus + queries)
.venv/bin/python scripts/datasets/prepare_multihoprag.py

# 1) Start Neo4j and validate external generation/embedding endpoints
./run_servers.sh all

# 2) Build the index
./run_index.sh --model prehop \
  --dataset data/multihoprag_corpus --corpus-tag multihoprag

# 3) Benchmark
./run_benchmark.sh --model prehop \
  --queries data/multihoprag_queries.json --corpus-tag multihoprag

# 4) Stop services
./stop_servers.sh all
```

Result JSON is written to `data/results/<timestamp>/prehop/<corpus_tag>/*.json`.
It contains per-query deterministic answer and evidence metrics, category
breakdowns, eligibility metadata, and aggregate metrics. Optional judge fields
appear only when judging is enabled. Missing gold units use `-1`; evaluated
misses are zero; runtime-error rows are excluded from aggregates. A result is
reportable only after the complete prepared split finishes without failed rows.

Command logs are isolated by run, dataset, and strategy under
`logs/{index|benchmark}/<run-id>/<corpus-tag>/<strategy>.log`. MS GraphRAG's
required pipeline report is kept below the same index scope in `internal/`;
it no longer writes a shared `logs/indexing-engine.log`.

Prehop is marked complete only after the live graph passes deterministic
index-quality checks: complete embeddings and ownership, valid and role-distinct
Q−/Q+, exact document-order `NEXT`, cross-document HOP constraints, bounded
out-degree, consistent Q+→Q-→owner provenance,
and online search indexes. Coverage and graph density are recorded as
descriptive statistics rather than tuned acceptance thresholds. Retrieval
quality remains a separate held-out benchmark question.

### Per-dataset entrypoints

`run_multihoprag.sh` and `run_dataset.sh` wrap the steps above with each
dataset's corpus, queries, and tags so you don't pass them by hand:

```bash
./run_servers.sh all            # services first

# MultiHop-RAG
.venv/bin/python scripts/datasets/prepare_multihoprag.py  # downloads corpus + full queries
./run_multihoprag.sh index --model prehop      # one strategy at a time
./run_multihoprag.sh benchmark --model prehop --queries full

# MuSiQue: preparation defaults to all 2,417 answerable dev rows.
.venv/bin/python scripts/datasets/prepare_musique.py
./run_dataset.sh musique all --model prehop
```

See `CLAUDE.md` "Data and tags" for corpus/query file details per dataset.

### Independent paper runs

Prepare both datasets once, then run each dataset/strategy pair independently.
The preparation scripts write a content-bound `corpus_manifest.json`; retain
the printed fingerprint with every reported result.

```bash
.venv/bin/python scripts/datasets/prepare_multihoprag.py
.venv/bin/python scripts/datasets/prepare_musique.py

# One cold index and full benchmark per invocation. Repeat for every strategy.
./scripts/run_paper_target.sh multihoprag prehop mhr-prehop-cold-01
./scripts/run_paper_target.sh multihoprag naive mhr-naive-cold-01
./scripts/run_paper_target.sh multihoprag hoprag mhr-hoprag-cold-01
./scripts/run_paper_target.sh multihoprag ms_graphrag mhr-ms-cold-01
./scripts/run_paper_target.sh multihoprag browsenet mhr-browsenet-cold-01
./scripts/run_paper_target.sh multihoprag proprag mhr-proprag-cold-01

./scripts/run_paper_target.sh musique prehop musique-prehop-cold-01
./scripts/run_paper_target.sh musique naive musique-naive-cold-01
./scripts/run_paper_target.sh musique hoprag musique-hoprag-cold-01
./scripts/run_paper_target.sh musique ms_graphrag musique-ms-cold-01
./scripts/run_paper_target.sh musique browsenet musique-browsenet-cold-01
./scripts/run_paper_target.sh musique proprag musique-proprag-cold-01
```

For the dataset files used by this revision, preparation must print the
following identities. A different fingerprint is a different prepared corpus
and must not be mixed with these runs.

| Dataset | Sources | Full queries | Corpus fingerprint |
|---|---:|---:|---|
| MultiHop-RAG | 609 | 2,556 | `87781bfec56d944e9e57c3f0e96dc28ba473d837bf4a48d17fdc5b8690a4a0b8` |
| MuSiQue answerable dev | 21,099 | 2,417 | `7560a2113c736776b7d4970ec02e3a8c8a2c04bf495f1b5ee4bf67718c323735` |

The wrapper requires a clean tracked worktree, disables Prehop chunk and
embedding caches, assigns run-specific HopRAG/MS GraphRAG output roots, clears
Neo4j, fixes benchmark concurrency at 4, disables the optional judge, and uses
the full prepared query file. It refuses an existing run ID instead of reusing
artifacts. An interrupted or failed index is incomplete and must be rebuilt
under a new run ID after resolving the cause. A
deterministic benchmark interrupted after valid checkpoints can resume under
the same run ID only when the runner verifies identical queries, models,
configuration, and index identity. Retained and resumed code provenance remain
separate in the result. Run artifacts record phase timings, graph integrity,
provenance coverage, index-storage size, and failures. Index-storage
size means the persisted index used during retrieval: a logical-payload
estimate for the Neo4j-backed Prehop, Naive RAG, and HopRAG indexes, and the
physical local retrieval-artifact size for MS GraphRAG. Inputs, caches, logs,
debug output, and temporary files are excluded. Detailed measurement fields,
storage-method limitations, and adapter-specific timing boundaries are defined
in [ARCHITECTURE](docs/ARCHITECTURE.md#run-measurements).

Use `--check` as the fourth argument to validate the worktree, revision fields,
prepared manifest, run ID, and isolated output paths without starting a run.

LLM judging is disabled by default. When it is explicitly enabled, OpenAI Batch
is the default transport. An interrupted submitted batch can be resumed without
re-running retrieval:

```bash
.venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<run-id>
```

An interrupted deterministic benchmark uses its original run ID and timestamp:

```bash
RAG_RUN_ID=<original-run-id> \
RAG_BENCHMARK_TIMESTAMP=<original-run-id> \
RAG_BENCHMARK_RESUME=true \
./run_benchmark.sh --model <strategy> --queries <queries.json> --corpus-tag <tag>
```

Batch payloads must contain valid explicit `score` and `groundedness` fields;
`hallucination` is derived locally from groundedness.
Partial or malformed output keeps its reconciliation manifest and is never
converted into a paper metric. Query-level `paired_bootstrap.py` produces
uncertainty intervals for paired strategy differences over the same evaluated
questions. A query-only ablation should also pass
`--expected-ablation-difference <metadata-key>`; the comparison then fails if
any unlisted ablation setting changed. It also requires identical active-index
snapshot, model identities and seed, code provenance, benchmark concurrency,
and judge state. Comparisons between intentionally different index tags require
the explicit `--allow-index-variant` flag and still require identical dataset,
corpus fingerprint, evaluation scope, and query IDs.

The predeclared relative-improvement requirement is checked from completed
artifacts rather than copied into a hand-edited table:

```bash
.venv/bin/python scripts/performance_gate.py \
  --prehop <prehop-summary.json> \
  --baselines <naive-summary.json> <hoprag-summary.json> <ms-summary.json> \
              <browsenet-summary.json> <proprag-summary.json> \
  --margin 0.10 --output data/results/<run-id>/official_metric_gate.json
```

For each dataset-specific official metric, the gate independently selects the
strongest supplied non-Prehop baseline and requires a 10% relative gain. Only
complete full-split artifacts are paper-eligible.

### Complete-split presentation diagnostics

These utilities require the complete MuSiQue split for reported presentation
results and preserve the distinction between structural diagnostics and human
annotation:

```bash
# Capture every MuSiQue candidate pool during a complete benchmark, then
# replay only the ordering call in the deterministic fused order and a fixed shuffle.
RAG_CANDIDATE_ORDER_TRACE_PATH=data/results/diagnostics/frozen_candidate_pools_2417.jsonl \
  ./run_benchmark.sh --model prehop --queries data/musique_queries.json \
  --corpus-tag musique
.venv/bin/python -m scripts.replay_frozen_candidate_order \
  --trace data/results/diagnostics/frozen_candidate_pools_2417.jsonl \
  --queries data/musique_queries.json --benchmark <complete-result.json> \
  --orders search hash_shuffle \
  --out data/results/diagnostics/frozen_candidate_order_replay_2417.json \
  --expected-traces 2417 --resume

# Re-evaluate deterministic rank variants on all frozen candidate pools.
.venv/bin/python -m scripts.analyze_full_frozen_rank_variants \
  --trace data/results/diagnostics/frozen_candidate_pools_2417.jsonl \
  --queries data/musique_queries.json --benchmark <complete-result.json> \
  --out data/results/diagnostics/frozen_rank_variants_2417.json \
  --expected-queries 2417

# Summarize non-overlapping stage timers from that fixed-concurrency full run.
.venv/bin/python -m scripts.analyze_full_stage_profile \
  --artifact <complete-result.json> \
  --out data/results/diagnostics/full_stage_profile_2417.json \
  --expected-queries 2417 --declared-concurrency 4

# After a same-query full graph-on/off pair, test effects separately where
# stored HOP edges do and do not connect gold paragraphs.
.venv/bin/python -m scripts.analyze_graph_shortcut_effect \
  --graph-on <graph-on-result.json> --graph-off <graph-off-result.json> \
  --coverage data/results/diagnostics/gold_hop_coverage.json \
  --out data/results/diagnostics/graph_shortcut_effect.json \
  --expected-queries 2417
```

The fixed-candidate order replay reports selection stability and
supporting-paragraph metrics. The rank-variant analysis reports
supporting-paragraph metrics on the same candidate pools. See
[ARCHITECTURE](docs/ARCHITECTURE.md#diagnostic-controls-and-timing) for the
measurement boundaries.
The stored-connection subgroup analysis compares graph expansion effects by
whether gold paragraphs share a stored connection.

---

## Ablation toggles

Method ablations are driven by environment toggles read in `core/config.py`:

| Variable | Default | Effect when set to `false` |
|---|---|---|
| `RAG_ABLATION_Q_PLUS` | `true` | Q⁺ dependency-seed retrieval disabled (also disables offline HOP-edge construction) |
| `RAG_ABLATION_Q_MINUS` | `true` | Q⁻ direct-evidence retrieval disabled |

Query-only ablations do not require rebuilding the index:

| Variable | Default | Alternatives |
|---|---|---|
| `RAG_HYPO_CHANNEL_VARIANT` | `full` | `body_only`, `qminus_only`, `qplus_only`, `single_combined` |
| `RAG_GRAPH_HOP_DEPTH` | `1` | `0` disables graph expansion |
| `RAG_GRAPH_PATH_DECAY` | `0.5` | `0` and `1` are declared propagation sensitivities |
| `RAG_GRAPH_EDGE_VARIANT` | `full` | `hop_only` or `next_only` isolates traversal-edge contributions |
| `RAG_HOP_EDGE_FILTER` | `none` | `reciprocal_offline` uses materialized reverse-Q+ agreement; `reciprocal` recomputes it online |
| `RAG_QPLUS_HOP_ACTIVATION` | `owner` | `exact` restricts activation to the matched Q+ IDs as an ablation |
| `RAG_CONTINUATION_EDGES_ENABLED` | `false` | `true` activates exact matched-Q− continuation edges on a `linked_v2` index |
| `RAG_QUERY_REWRITE_VARIANT` | `role_aligned_evidence_iterative` | `none` disables rewriting; `role_aligned` performs only the initial bounded rewrite |
| `RAG_QUERY_REWRITE_MAX_WORDS` | `32` | Questions above the limit skip rewriting; `0` rewrites every question |
| `RAG_QUERY_REFINEMENT_MAX_ROUNDS` | `0` | `0` uses evidence stability; a positive value is an operational call cap |
| `RAG_SOURCE_SELECTION_VARIANT` | `role_body_list_ranking` | `global` uses the fixed fused order directly; `round_robin` diversifies sources |
| `RAG_CANDIDATE_ORDER_INPUT_ORDER` | `search` | `reverse` and `hash_shuffle` are position-dependence diagnostics over a frozen pool |
| `RAG_FINAL_RANK_VARIANT` | `fused` | `semantic_only` and `representation_only` isolate final deterministic signals |
| `RAG_HOP_SEMANTIC_VARIANT` | `body_bridge_min` | `body_only` and `bridge_only` isolate HOP semantic evidence |

The submission component controls are the same-index MuSiQue graph on/off,
refinement removal, candidate-selection policy comparison, fixed-candidate
order replay, rank variants, and fixed-concurrency stage profile. Their exact
stages, fixed conditions, and outputs are specified in
[ARCHITECTURE](docs/ARCHITECTURE.md#component-evaluation-contract).

Index-changing ablations use distinct corpus tags. Query-only ablations reuse
the same immutable index and are recorded in result metadata.
`RAG_QUESTION_SCHEMA=grounded_v1` and
`RAG_PRECOMPUTE_RECIPROCAL_HOPS=true` are index-time options; the former stores
source-verifiable structured Q−/Q+, and the latter enables the
`reciprocal_offline` query filter.
`RAG_QUESTION_SCHEMA=linked_v2` is an experimental, separately indexed schema.
It adds complete grounded Q− answer anchors and exact cross-document
continuation links. Repeated answers share one anchor node, so common entities
do not create a source-question × target-mention edge product. The links can be
enabled or disabled at query time on the same snapshot. Continuation edges are
outside the submitted configuration and final component claims.

The final query path uses Q−/Q+ question representations, all stored outgoing
connections from a matched Q+ paragraph, evidence-conditioned role rewriting
for questions of at most 32 words, and one complete-list candidate-selection
call. Top-k remains 12.

---

## Key hyperparameters

Full list in the paper appendix; the most important:

| Parameter | Value | Where |
|---|---|---|
| `CHUNK_SENTENCES` | 6 | page-scoped Prehop and controlled-Naive window |
| Retrieval depth | Prehop 12; Naive 12 | common evidence budget |
| Input capacity | generation 262,144; embedding 32,768 tokens | configured endpoint limits |
| Questions per direction | 3 | fixed output-schema bound for Q− and Q+ |
| HOP targets | at most one candidate per Q+ | nearest cross-document Q− owner |
| Query representations | role-normalized set union | original body query plus bounded Q−/Q+ views for questions of at most 32 words |
| Query rewrite | evidence-conditioned role refinement, at most 32 input words | stops on no new role question or selected chunk; optional operational round cap |
| Query-time rank fusion | equal reciprocal ranks | representation order + body/bridge semantic order; no fitted weight |
| Graph rank propagation | reciprocal path length | a one-edge target inherits half of its source rank evidence |
| Offline HOP selection | nearest cross-document Q+→Q− owner resolution | reciprocal provenance is retained only for ablation |
| Q+ activation | owner | exact matched-Q+ activation is an ablation |
| Embedding dim | `NEO4J_VECTOR_DIMENSIONS` | must match the configured embedding model's output dimension |

Offline candidate construction uses rank-based Q+→Q− matching. Query-time
candidate ordering retains each
representation's reciprocal ranks, combines the resulting representation
order with the semantic order using equal reciprocal ranks, then asks the
configured generation model to select paragraph numbers from the complete
candidate pool before final answer synthesis.

---

## License

MIT — see [LICENSE](LICENSE).
