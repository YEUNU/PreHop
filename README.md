# Prehop: Offline Question Links for Multi-Hop Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Prehop is a GraphRAG retrieval system that constructs inspectable,
question-level chunk links during indexing. At query time, it refines the
question by evidence role, expands stored neighbors, selects up to twelve candidate
paragraphs with the configured generation model, and synthesizes the answer.

Use the guide that matches your task:

| Document | Role |
|---|---|
| `README.md` | Public overview, setup, and command guide |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Normative implementation, evaluation, and component-control specification |
| [RESULTS](docs/RESULTS.md) | Canonical complete-result and artifact register |
| [RUNTIME REQUIREMENTS](docs/RUNTIME_REQUIREMENTS.md) | Pinned external setup and fail-closed preflight |
| [CHANGELOG](docs/CHANGELOG.md) | Chronological engineering record |
| [Throughput execution](docs/THROUGHPUT_EXECUTION.md) | Queue profiles, launch procedures, and cost definitions |
| [Maintainer guide](CLAUDE.md) | Contribution, experiment, and documentation policy |

The gitignored `docs/prehop_paper.md` is the research manuscript, and the
gitignored `SUBMISSION_TARGET.md` contains private submission logistics.

---

## What this repository is

Prehop builds the graph offline and uses it to retrieve evidence at query time.
The implementation and branch details are in [ARCHITECTURE](docs/ARCHITECTURE.md).

Core indexing-time design, currently evaluated on MultiHop-RAG and MuSiQue:

1. Every chunk receives separate hypothetical questions for facts it answers
   ($Q^-$) and information it still requires ($Q^+$).
2. Each Q+ retrieves the closest cross-document Q-. The Q-'s owner chunk is
   the HOP target candidate, so its body follows from graph ownership without
   a second body search. When Q- is disabled for ablation, Q+ retrieves a body
   candidate directly.

Chunking is fixed-size (page-scoped sentence windows) — see
[ARCHITECTURE](docs/ARCHITECTURE.md#shared-input-contract) for details.

For questions of at most 32 words, Prehop creates Q−/Q+ retrieval views and
refines them using retrieved evidence. It combines body and question searches,
expands stored neighbors by one hop, and selects up to 12 paragraphs for answer
synthesis. Longer questions keep their original wording. See the
[query path](docs/ARCHITECTURE.md#prehop-query-path-and-branches) for ranking,
refinement stopping rules, and graph controls.

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
Only admitted complete prepared-split runs are eligible for submission
results.

Adapters preserve their declared native retrieval and answer paths. Pinned
upstream source is immutable; controlled deviations have separate semantic
identities. See the [adapter boundary](docs/ARCHITECTURE.md#upstream-immutability-boundary).

### Result admission

The primary paper matrix contains eight independent strategies: Prehop, Naive
RAG, MS GraphRAG, LightRAG, HippoRAG2, GFM-RAG, LinearRAG, and
Youtu-GraphRAG. BrowseNet, HopRAG, and PropRAG remain supported legacy
adapters but are not primary matrix targets. Each target uses the full prepared
MultiHop-RAG or MuSiQue split. Generation and answer synthesis use
`gemma-4-31b-it` where the upstream method permits an OpenAI-compatible
endpoint. Remote embeddings use the configured `qwen3-embedding-0.6b`;
LinearRAG retains its official pinned MPNet, and the declared controlled Youtu
no-agent adapter retains pinned MiniLM. GFM-RAG retains its
checkpoint-defined local components. Local method components and adapter
variants are recorded separately from controlled remote-backbone
configurations. MultiHop-RAG and MuSiQue remain in separate
tables because their metrics and denominators differ. A canary is not target
completion, and completion is not admission. The artifact-admission
and publication checks are defined in [RESULTS](docs/RESULTS.md).

---

Prehop uses strict structured output for indexing questions and query control.
See the [schema contracts](docs/ARCHITECTURE.md#structured-generation-contracts)
and [runtime validation](docs/RUNTIME_REQUIREMENTS.md#live-gates) before running
the full paper workflow.

## Repository layout

```
prehop/
├── main.py                          # single CLI entry point (index/benchmark/maintenance)
├── cli/
│   ├── index.py                     # indexing runner
│   └── benchmark.py                 # benchmark runner (single + multi-seed)
├── core/
│   ├── config.py                    # RAGConfig — validated env-driven settings
│   ├── strategy_registry.py         # primary/legacy strategy source of truth
│   ├── inference_transport.py       # typed single-gateway contract
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
│   ├── proprag/                     # pinned PropRAG reference adapter
│   └── external_research/            # thin Light/Hippo/GFM/Linear/Youtu drivers
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
├── run_servers.sh                   # validate/start Neo4j + single inference gateway
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
# Required: NEO4J_PASSWORD and the single LiteLLM gateway settings
```

`pyproject.toml` and `uv.lock` are the dependency contract, including the
HopRAG spaCy model. The run scripts
auto-discover `.venv/bin/python` (override with `PYTHON_BIN`), so you do not
need to activate the environment.

`run_servers.sh` validates the configured LiteLLM gateway and its registered
generation and embedding models. It never launches local model processes.

The current cold-run configuration uses the following model identities. The
paper target derives controlled-gateway revision provenance from these exact
registered names; local ancillary revisions come from the strategy registry.

| Role | Served model | Required setting |
|---|---|---|
| Generation and synthesis | `gemma-4-31b-it` | `RAG_GENERATION_MODEL` |
| Remote embeddings | `qwen3-embedding-0.6b`, 1,024 dimensions | `RAG_EMBEDDING_MODEL` |

Result tables retain the model identity stored in their cited artifacts. A
runtime configuration change does not relabel an earlier result.

Remote generation and embedding requests use the same
`RAG_INFERENCE_BASE_URL` and `RAG_INFERENCE_API_KEY`; a missing gateway,
different generation/embedding bases, or a direct vendor fallback fails
closed in paper mode. Shell entrypoints reject legacy `VLLM_*`, ambient
provider, and direct-vendor variables. Isolated child processes receive any
upstream compatibility names only from the validated typed transport. External server
hardware and launch options are recorded as the execution resource budget,
separately from semantic method settings.

### Isolated official baseline runtimes

Externally maintained methods run in isolated environments because their
official dependencies conflict with the main project environment. Install the
pinned official revisions once:

```bash
./scripts/setup_official_baselines.sh --primary
```

The setup keeps official source, model dependencies, and downloaded weights
under the ignored `data/official_baselines/` directory. The repository stores
only thin adapters and exact upstream commit identifiers. LinearRAG's GPL code
never enters this MIT tree, and Youtu's upstream academic/research-use terms
still apply. LinearRAG's primary official-faithful mode uses a pinned local
MPNet snapshot and `en_core_web_trf` in its Python 3.9 environment;
Youtu-GraphRAG's controlled no-agent adapter uses a pinned local MiniLM
snapshot and `en_core_web_lg`;
GFM-RAG resolves its approved checkpoint, configuration, and ColBERT entity
linker from pinned local snapshots; Youtu-GraphRAG also requires a
dataset-specific approved content-addressed schema in its pinned checkout.
MultiHop-RAG explicitly uses the upstream `hotpot` dataset policy; MuSiQue uses
the upstream `musique` policy. The native retrieval/filter budget is 20 and
the upstream no-chunk document path is retained for both aliases.
BrowseNet's legacy ColBERT checkpoint belongs under
`data/official_baselines/browsenet/artifacts/colbertv2.0`, outside its source
checkout. Existing dirty legacy checkouts and source-local checkpoints are
preserved; setup does not clean or migrate them. Youtu post-validation separately records staged-input
coverage, native extraction success, and source reachability observable in the
native graph. Its public no-agent API returns ordered evidence; sentinel,
malformed, empty, and swallowed-error results fail. Native
deduplication can leave repeated-source provenance unreachable; that
observational limitation is reported and does not rewrite native retrieval.
After preparing the dataset as described below, check the target with the
selected main environment (`PYTHON_BIN` or `UV_PROJECT_ENVIRONMENT`). These
checks do not print credentials or synchronize the environment:

```bash
./scripts/run_paper_target.sh multihoprag linear_rag linear-preflight-01 --check
./scripts/run_paper_target.sh multihoprag youtu_graphrag youtu-preflight-01 --check
./scripts/run_paper_target.sh multihoprag gfm_rag gfm-preflight-01 --check
```

The machine-readable requirements, including local model revisions, approved
checkout-relative schema paths, and schema SHA-256 values, are in
`configs/paper_runtime_requirements.json`. The setup and preflight fail when an
external checkout has the wrong revision or a dirty tracked or untracked
working tree. They do not install dependencies at import time.

---

## Quick start

```bash
# 0) Prepare a dataset (downloads + builds corpus + queries)
.venv/bin/python scripts/datasets/prepare_multihoprag.py

# 1) Start Neo4j and validate the single external inference gateway
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

# One cold index and full benchmark per invocation.
./scripts/run_paper_target.sh multihoprag prehop mhr-prehop-cold-01
./scripts/run_paper_target.sh multihoprag naive mhr-naive-cold-01
./scripts/run_paper_target.sh multihoprag ms_graphrag mhr-ms-cold-01
./scripts/run_paper_target.sh multihoprag lightrag mhr-light-cold-01

# Print the authoritative primary order used by the runners.
python3 core/strategy_registry.py --primary-lines
```

To preflight the complete 16-target matrix in registry order:

```bash
./scripts/run_paper_matrix.sh submission-01 --check
```

A real matrix is locked behind the content-bound live-gate ledger. After an
independent static GO, use `scripts/run_paper_live_gates.sh` in the order
defined in [RUNTIME REQUIREMENTS](docs/RUNTIME_REQUIREMENTS.md), then run
`./scripts/run_paper_matrix.sh submission-01`. The matrix refuses a missing,
incomplete, changed, or stale ledger.

Each target is independent. The matrix records a failure, continues with the
remaining targets, reports every failed target, and exits nonzero if any target
failed. It never launches a fallback model implicitly.

The target wrapper allocates a fresh run namespace, disables shared index
caches, checks runtime and input identities, and records failures. It never
clears the whole Neo4j database. A complete compatible index can be reused only
through the verified index-link workflow; checkpoint recovery preserves the
original run identity.

Corpus identities, query denominators and admission checks belong in
[RESULTS](docs/RESULTS.md). Indexing and query throughput use separate measured
wall-time boundaries defined in [Throughput execution](docs/THROUGHPUT_EXECUTION.md).
For gate order, recovery and background ownership, use
[Runtime requirements](docs/RUNTIME_REQUIREMENTS.md#durable-campaign-ownership).

### Component analysis

The component analyses use the complete MuSiQue split and verified index or
candidate-pool identities. They cover graph expansion, query refinement,
candidate selection, fixed-candidate ordering, rank variants and stage timing.
Use the [component contract](docs/ARCHITECTURE.md#component-evaluation-contract)
for required controls and the [diagnostic tools](docs/ARCHITECTURE.md#diagnostic-controls-and-timing)
for implementation details. Their outcomes remain separate from the primary
full-system comparison.

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

## Background execution

### Full paper campaign

From the prepared execution worktree, launch a validated campaign plan with
`"$PYTHON_BIN" scripts/paper_campaign.py launch <exact-plan.json>` after setting
`PYTHON_BIN` to the selected environment's Python executable. The default
backend uses `nohup` and a new process session; it does not require systemd or
user linger. `--backend systemd` retains the optional legacy service launcher.

The launch receipt identifies the supervisor, private environment file and
independent monitor. Beside the plan, `status.json` and stage logs retain
progress and failures. `monitor-*-observations.jsonl` records a read-only check
every three hours; `monitor-*-terminal.json` records completion, failure or a
missing supervisor. Terminal checks wait for owned cleanup or process exit.
The monitor verifies existing admissions and reports unavailable ETA explicitly.
These files are persistent observations, not automatic chat notifications.

After the complete-corpus one-query matrix and the separate fresh full-target
gate pass, the final matrix reuses verified indexes with fresh result files and
separate native query workspaces. Each result's `index_link.json` preserves the
source configuration, data, index identity and original indexing cost.

### Throughput profiles and amortized paper costs

The optional owned inference queue bounds generation and embedding requests
across native worker processes while targets run sequentially. Explicit JSON
profiles bind operational settings to gate and admission evidence. New artifacts
record indexing s/source-document and query s/query as wall-time normalization,
separate from request latency. See [the execution protocol](docs/THROUGHPUT_EXECUTION.md)
for the pilot tool, foreground wrapper, persistent campaigns, and measurement
limitations. `index-throughput-tested.json` records the tested indexing settings;
`throughput-pilot.json` is an unvalidated tuning example. Neither establishes
a global throughput optimum or validated query concurrency.
[Adapter producer parallelism](docs/ARCHITECTURE.md#adapter-producer-parallelism) documents native
producer controls, bounded Prehop chunk lookahead and synchronized Youtu schema
updates. Pinned upstream source files are not modified.

For indexing only, follow the [detached index batch procedure](docs/THROUGHPUT_EXECUTION.md#index-only-batch).
It queues all 16 targets and records indexing outcomes without running full
query benchmarks or starting the paper campaign monitor.

### Inspect Prehop traces

Prehop saves stage inputs/outputs and LLM request/response bodies by default,
including failed and retried responses. Index stats and benchmark rows contain
`prehop_trace.events_path`; document intermediates remain under `data/debug`.

```bash
# Show failed attempts and their full payloads.
.venv/bin/python scripts/inspect_prehop_trace.py /path/to/events.jsonl --errors --payloads

# Follow one source document or benchmark query.
.venv/bin/python scripts/inspect_prehop_trace.py /path/to/events.jsonl --source source.txt --payloads
.venv/bin/python scripts/inspect_prehop_trace.py /path/to/events.jsonl --query-id query-id --payloads
```

The inspector verifies event order and payload hashes. See
[Prehop tracing](docs/ARCHITECTURE.md#prehop-tracing) for scope, retention and
measurement boundaries. Trace payloads contain source and model text and remain
local; they are not publication-ready exports.
For repository exclusions and custom trace paths, see
[Local files and Git](docs/RUNTIME_REQUIREMENTS.md#local-files-and-git).

## License

MIT — see [LICENSE](LICENSE).
