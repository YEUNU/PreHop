# Prehop: Offline Question Links for Deterministic Multi-Hop Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Benchmark numbers are intentionally omitted from this overview. They are
> published only from complete, integrity-checked full-corpus runs.

Reference implementation of a GraphRAG framework whose core claim is
indexing-time HOP construction: inspectable question-level evidence links are
constructed once, offline, into chunk-to-chunk edges, so the query path expands
the graph deterministically with no per-hop LLM reasoning. The query path is
a thin role-based hybrid retrieve over a graph built once offline — no agent
loop, no reflection, no refinement.

---

## What this repository is

Core indexing-time design, currently evaluated on MultiHop-RAG and MuSiQue:

1. Every chunk receives separate hypothetical questions for facts it answers ($Q^-$) and information it still requires ($Q^+$).
2. Each Q+ retrieves the closest cross-document Q-. The Q-'s owner chunk is
   the evidence target, so its body follows from graph ownership without a
   second body search. When Q- is disabled for ablation, Q+ retrieves a body
   target directly.

Chunking is fixed-size (page-scoped sentence windows) — see `CLAUDE.md` "Architecture notes" for details.

The query path is deliberately thin: one parallel role-based retrieve over
Q⁻/body direct evidence and Q⁺ dependency seeds, parameter-free reciprocal-rank
fusion of representation and semantic orders, deterministic batched 1-hop
traversal over bidirectional `NEXT` and outgoing `HOP_ANSWER` edges, and a
single LLM synthesis call. Rank evidence propagated over a graph edge is
attenuated by reciprocal path length, so indirect evidence does not enter the
representation order as strongly as a directly retrieved owner.

---

## Results

The repository fixes dataset-specific evaluation rather than using one pooled
retrieval score. Gold-evidence retrieval is primary for the method claim, and
deterministic normalized answer EM/F1 provides downstream validation, with
alias-aware matching for MuSiQue. MultiHop-RAG reports official-compatible
any-hit/MRR/MAP fields separately from custom `evidence_fact_recall@k` and a
null-refusal slice. MuSiQue reports `paragraph_support_*` using the official
SupportMetric formula over stable global paragraph identities; this is not an
official query-local-`idx` submission (titles remain diagnostic-only). The
LLM judge is an optional exploratory diagnostic: semantic correctness and
context groundedness are separate fields, and neither supports a primary
claim without qualified-human validation. It is disabled by default; set
`RAG_JUDGE_ENABLED=true` only for internal error analysis or a separately
validated supplemental study. See [the local paper
specification](docs/prehop_paper.md), [the architecture and evaluation
contract](docs/ARCHITECTURE.md), and `CLAUDE.md` for the complete research
protocol. The finalized paper specification is intentionally kept local and
untracked until a deliberate submission export.

The judge model must differ from both the run's generation model and
`VLLM_SERVED_MODEL_NAME`. `RAG_JUDGE_ALLOW_SELF=true` exists only for explicit
non-paper debugging and is recorded in result metadata. Paired bootstrap uses
stable query IDs and rejects incomplete, scope-mismatched, or corpus/index
fingerprint-incompatible artifacts; legacy/exploratory overrides are opt-in.

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
│   ├── naive/                       # baseline (shared fixed-window chunking + vector search)
│   ├── hoprag/                      # baseline (runtime hop traversal via official HopRAG)
│   └── ms_graphrag/                 # baseline (community-report retrieval via graphrag package)
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
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .   # clients, Neo4j, and baseline libraries
.venv/bin/python -m spacy download en_core_web_sm   # required by the hoprag baseline

# Neo4j 5.x — Docker is simplest:
docker run -d --name prehop-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> neo4j:5

# Configure env vars
cp .env.example .env
# Required: NEO4J_PASSWORD and the external generation/embedding endpoint settings
```

`pyproject.toml` is the canonical dependency list. The run scripts auto-discover `.venv/bin/python` (override with `PYTHON_BIN`), so you do not need to activate the venv.

`run_servers.sh` validates the configured external generation and embedding
endpoints. It never launches local model processes. See `CLAUDE.md` "Model /
inference infra".

---

## Quick start

```bash
# 0) Prepare a dataset (downloads + builds corpus + queries)
python3 scripts/datasets/prepare_multihoprag.py

# 1) Start Neo4j and validate external generation/embedding endpoints
./run_servers.sh all

# 2) Build the index
./run_index.sh --model prehop --dataset data/multihoprag_corpus --corpus-tag multihoprag

# 3) Benchmark
./run_benchmark.sh --model prehop --queries data/multihoprag_sample200_queries.json --corpus-tag multihoprag

# 4) Stop services
./stop_servers.sh all
```

Result JSON is written to `data/results/<timestamp>/prehop/<corpus_tag>/*.json`
and includes per-query deterministic answer EM/F1, official-compatible and
custom evidence metrics,
category breakdowns, the shared 3-way answer label (Correct / Incorrect /
Refusal), supplemental LLM-judge fields, and aggregate metrics. Missing gold
units are excluded with `-1`; evaluated misses are zero. Aggregates print an
eligible-row denominator and exclude runtime-error rows. Sample manifests are
explicitly exploratory; only complete official splits with a rebuilt index can
be described as official-compatible runs. MuSiQue corpus metadata headers are
removed before indexing and are used only to audit source identity. Incomplete or failed runs are not
paper-eligible.

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
python3 scripts/datasets/prepare_multihoprag.py  # downloads corpus + full queries
./run_multihoprag.sh index --model prehop      # one strategy at a time
./run_multihoprag.sh benchmark --model prehop --queries full

# MuSiQue: preparation defaults to all 2,417 answerable dev rows.
# Create exploratory samples only with make_sample.py.
python3 scripts/datasets/prepare_musique.py
python3 scripts/datasets/make_sample.py --dataset musique --per-type 67
./run_dataset.sh musique all --model prehop
```

See `CLAUDE.md` "Multi-hop dataset suite" for corpus/query file details per dataset.

### Independent paper runs

Run each dataset/strategy pair independently. This keeps failures, resource
measurements, logs, and corpus state attributable to one experimental target
and avoids matrix-level retry or resume policy. Clear Neo4j only before the
first target when multiple corpus tags must coexist.

```bash
RAG_RUN_ID=mhr-prehop-cold ./run_multihoprag.sh index \
  --model prehop --clear-graph
RAG_RUN_ID=musique-prehop-cold ./run_dataset.sh musique index \
  --model prehop

# Repeat explicitly for controlled and official baselines.
RAG_RUN_ID=mhr-naive-cold ./run_multihoprag.sh index --model naive
RAG_RUN_ID=mhr-hoprag-cold ./run_multihoprag.sh index --model hoprag
RAG_RUN_ID=mhr-ms-graphrag-cold ./run_multihoprag.sh index --model ms_graphrag
```

Use a new explicit run ID for every cold paper run. An interrupted or failed
run is incomplete; start a replacement run after resolving the cause rather
than combining partial state. Prehop's runtime stats split document
generation/embedding, final graph
flush, HOP construction, and structural-audit time. Question coverage,
direction coverage, provenance completeness, graph size, prompt-length
violations, and NEXT/HOP topology checks are read from the stored result rather
than inferred from counters. For Prehop, full-query gold evidence titles are
also resolved against the indexed corpus and used to report gold-document-pair
and query-level HOP connectivity. This is an intermediate semantic-validity
measure, not a substitute for retrieval/answer accuracy.
Official adapters retain their aggregate pipeline timing plus the workflow or
stage timings exposed upstream; boundaries that upstream code does not expose
are not fabricated. MS GraphRAG relationship drops caused by missing extracted
entities are recorded as integrity warnings.

LLM judging is disabled by default. When it is explicitly enabled, OpenAI Batch
is the default transport. An interrupted submitted batch can be resumed without
re-running retrieval:

```bash
.venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<run-id>
```

Batch payloads must contain valid explicit `score` and `groundedness` fields;
`hallucination` is derived locally from groundedness.
Partial or malformed output keeps its reconciliation manifest and is never
converted into a paper metric. Query-level `paired_bootstrap.py` produces
uncertainty intervals for paired strategy differences over the same evaluated
questions.

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
| `RAG_HYPO_CHANNEL_VARIANT` | `full` | `qminus_only`, `qplus_only`, `single_combined` |
| `RAG_GRAPH_HOP_DEPTH` | `1` | `0` disables graph expansion |
| `RAG_GRAPH_EDGE_VARIANT` | `full` | `hop_only` or `next_only` isolates traversal-edge contributions |
| `RAG_QUERY_REWRITE_VARIANT` | `none` | `role_aligned` generates Q−/Q+ retrieval views at query time |
| `RAG_SOURCE_SELECTION_VARIANT` | `global` | `round_robin` enables source diversification as an ablation |

The primary direction ablation is `{full, Q⁻-only, Q⁺-only}`. Source
text, including pipe-delimited text, is indexed as-is; there is no
table-to-text generation branch.

Index-changing ablations use distinct corpus tags. Query-only ablations reuse
the same immutable index and are recorded in result metadata.

---

## Key hyperparameters

Full list in the paper appendix; the most important:

| Parameter | Value | Where |
|---|---|---|
| `CHUNK_SENTENCES` | 6 | fixed-size chunking window (sentences per chunk) |
| Questions per direction | 3 | fixed output-schema bound for Q− and Q+ |
| HOP targets | at most one candidate per Q+ | nearest cross-document Q− owner |
| Query representations | set union | $Q^-$ / body direct evidence and $Q^+$ dependency seeds, each searched once |
| Query-time rank fusion | equal reciprocal ranks | representation order + body/bridge semantic order; no fitted weight |
| Graph rank propagation | reciprocal path length | a one-edge target inherits half of its source rank evidence |
| Offline HOP selection | Q+→Q− owner resolution | the matched Q− deterministically identifies its evidence chunk |
| Embedding dim | `NEO4J_VECTOR_DIMENSIONS` | must match the configured embedding model's actual output dim (see `CLAUDE.md` "Model / inference infra") |

Offline candidate construction is rank-based and has no learned reranker,
fixed cosine threshold, domain gate, semantic judge, or tuned acceptance
score. Query-time candidate ordering is also threshold-free: it retains each
representation's reciprocal ranks, combines the resulting representation
order with the semantic order using equal reciprocal ranks, and makes no LLM
call before final answer synthesis.

---

## What is intentionally **not** in this repository

The system does not include a reflective agent loop. A five-stage Perception/Planning/Execution/Reflection/Refinement loop was explored on top of the same indexing pipeline; it measured net-negative on this system itself and at best baseline-dependent across the four GraphRAG systems tried. The paper reports the retrieval-only configuration and the code here mirrors that decision.

---

## License

MIT — see [LICENSE](LICENSE).
