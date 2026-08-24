# Prehop / HypoHop: Question-Level Offline HOP Construction for Deterministic Multi-Hop Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Benchmark numbers are intentionally omitted from this overview. They are
> published only from a complete, integrity-checked full-corpus matrix.

Reference implementation of a GraphRAG framework whose core claim is
indexing-time HOP construction: inspectable question-level evidence links are
rank-fused once, offline, into chunk-to-chunk edges, so the query path expands
the graph deterministically with no per-hop LLM reasoning. The query path is
a thin two-stage hybrid retrieve over a graph built once offline — no agent
loop, no reflection, no refinement.

---

## What this repository is

Core indexing-time design, currently evaluated on MultiHop-RAG, 2WikiMultiHopQA, and MuSiQue:

1. **Predictive Knowledge Mapping** — every chunk receives dual hypothetical-query annotations ($Q^-$ for self-contained facts, $Q^+$ for outgoing dependencies), indexed separately. This is the structural precondition for HOP edges below, not a standalone feature.
2. **Rank-Fused HOP Edges Pre-Built Offline** — every Q+ independently searches cross-document Q-, body, and Q+ representations. Q+/Q- or Q+/body is required for a traversable edge; Q+/Q+ can only support its rank. There is no learned cross-encoder or cosine threshold.

Chunking is fixed-size (page-scoped sentence windows) — see `CLAUDE.md` "Architecture notes" for details.

The query path is deliberately thin: two-stage hybrid retrieve (Q⁻/body, then
Q⁺ expansion), external-embedding cosine top-k ordering, deterministic 1-hop
traversal over bidirectional `NEXT` and outgoing `HOP_ANSWER` edges, and a
single LLM synthesis call.

---

## Results

The repository fixes dataset-specific evaluation rather than using one pooled
retrieval score. Deterministic normalized answer EM/F1 is primary, with
alias-aware matching for MuSiQue. MultiHop-RAG uses non-null fact recall and a
separate null-refusal slice; 2WikiMultiHopQA uses supporting-title P/R/F1;
MuSiQue uses supporting-paragraph/title P/R/F1 because its gold evidence is
paragraph-level. The LLM judge is supplemental: semantic correctness and
context groundedness are separate fields, and it is never the sole answer
metric. See [the local paper specification](docs/prehop_paper.md) and
`CLAUDE.md` for the complete protocol.

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
├── data/                            # datasets and generated local indices (gitignored)
│   ├── prepare_multihoprag.py       # download/build the MultiHop-RAG corpus + queries
│   ├── prepare_2wikimultihopqa.py   # build the official 2WikiMultiHopQA dev corpus + queries
│   ├── prepare_musique.py           # download/build the MuSiQue (answerable dev) corpus + queries
│   └── make_sample.py               # stratified query samples for the supported datasets
├── scripts/                         # analysis, judge reconciliation, and measured matrix runs
├── tests/                           # chunking / retrieval / live-integration
├── run_servers.sh                   # validate/start Neo4j + generation/embedding endpoints
├── run_index.sh / run_benchmark.sh  # low-level, dataset-agnostic
├── run_multihoprag.sh               # per-dataset entry: index|benchmark|all
├── run_dataset.sh                   # per-dataset entry for 2wikimultihopqa|musique
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
python3 data/prepare_multihoprag.py

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
and includes per-query deterministic answer EM/F1, evidence metrics,
category breakdowns, the shared 3-way answer label (Correct / Incorrect /
Refusal), supplemental LLM-judge fields, and aggregate metrics. Missing gold
units are excluded with `-1`; evaluated misses are zero. Incomplete or failed
runs are not paper-eligible.

### Per-dataset entrypoints

`run_multihoprag.sh` and `run_dataset.sh` wrap the steps above with each
dataset's corpus, queries, and tags so you don't pass them by hand:

```bash
./run_servers.sh all            # services first

# MultiHop-RAG
python3 data/prepare_multihoprag.py            # downloads corpus + full queries
./run_multihoprag.sh all                       # index all 4 + benchmark (sample200)
./run_multihoprag.sh benchmark --queries full  # or the full 2556-query set

# 2WikiMultiHopQA / MuSiQue
python3 data/prepare_2wikimultihopqa.py && python3 data/make_sample.py --dataset 2wikimultihopqa --per-type 50
python3 data/prepare_musique.py && python3 data/make_sample.py --dataset musique --per-type 67
./run_dataset.sh 2wikimultihopqa all
./run_dataset.sh musique all
```

If `data/2wikimultihop_raw/dev.json` is absent, the 2Wiki preparation script
downloads the official archive and extracts only the development split needed
for this closed-corpus experiment.

See `CLAUDE.md` "Multi-hop dataset suite" for corpus/query file details per dataset.

### Full indexing matrix and paper measurements

After starting generation and embedding services, the following command clears
Neo4j once and runs the matrix in four strategy barriers. Each barrier runs the
three datasets concurrently, then advances to the next strategy:

```bash
VLLM_MAX_NUM_SEQS=120 VLLM_GENERATION_MAX_NUM_SEQS=120 VLLM_EMBED_MAX_NUM_SEQS=120 \
.venv/bin/python scripts/run_index_matrix.py \
  --datasets multihoprag 2wikimultihopqa musique \
  --strategies ms_graphrag hoprag naive prehop \
  --clear-graph --max-parallel 3 --max-generation-parallel 3 \
  --save-prehop-intermediate
```

Results are isolated under `artifacts/indexing/<run-id>/`: per-target logs,
GNU-time CPU/RSS measurements, host/vLLM pressure samples, integrity statistics,
`summary.csv`, and a paste-ready `paper_table.md`. Sustained host-memory or
vLLM queue pressure automatically reduces the parallel width for remaining
targets; a resource/rate-limit failure also halves that target's internal
worker count on retry. `logical_payload_bytes_estimate` is a cross-strategy reproducible
payload estimate; it is not Neo4j's physical store-file size.
The runner applies one per-target generation budget to `MAX_CONCURRENT_LLM_CALLS`,
`RAG_MS_CONCURRENT_REQUESTS`, and HopRAG's
`RAG_HOP_DOC_WORKERS × RAG_HOP_MAX_THREADS`. With three generation targets and
`max_num_seqs=120`, the aggregate generation upper bound is at most 120.
Naive flattens 32 source documents into each embedding/write batch, which is
important for one-chunk corpora. Matrix runs also preserve per-target phase
timing and interrupted attempt fragments in `attempt_journal.jsonl`. Resume an
interrupted run in the same folder with the original `--run-id`:

```bash
.venv/bin/python scripts/run_index_matrix.py \
  --run-id <run-id> --resume --target-attempts 3
```

Use `scripts/merge_index_matrix_runs.py` only when combining independent
historical run folders. `--target-attempts 2` means two total pipeline
attempts (one retry); `RAG_HOP_INTERNAL_RETRIES=2` means two total HopRAG
official-call attempts (one internal retry). The default watch line is emitted
hourly and includes the current strategy, completed/pending targets, and an
ETA that includes estimated future barriers; on a cold run it uses
strategy-specific durations from prior matrix result folders when available.
If no prior sample exists, the snapshot reports the unknown target count
instead of fabricating a duration. Use `--watch-interval 60` for a minute-level
watch. SIGINT/SIGTERM create resumable attempt fragments; SIGKILL cannot be
captured. Resource failures halve the affected target's adapter limits and
global LLM semaphore on retry.
Prehop's runtime stats also split document generation/embedding, final graph
flush, HOP construction, and structural-audit time. Question coverage,
direction coverage, provenance completeness, graph size, prompt-length
violations, and NEXT/HOP topology checks are read from the stored result rather
than inferred from counters. For Prehop, full-query gold evidence titles are
also resolved against the indexed corpus and used to report gold-document-pair
and query-level HOP connectivity. This is an intermediate semantic-validity
measure, not a substitute for retrieval/answer accuracy.
Capacity topology is explicit. `RAG_INFERENCE_CAPACITY_MODE=separate` is used
when generation and embedding model names are routed by one API gateway to
different accelerator servers; each receives its own `max_num_seqs=120`
budget. `shared` combines their worst-case pressure under one budget, while
`auto` conservatively infers sharing from URL equality. The calculation uses
`--max-generation-parallel` for generation rather than dividing that budget by
embedding-only matrix slots. Both queues are sampled. Official adapters retain
the aggregate `official_pipeline_seconds` plus adapter-observed workflow/stage
timings; timings are not fabricated for boundaries the upstream package does
not expose. Naive has aggregate pipeline timing only. MS GraphRAG relationship
drops caused by missing extracted entities are recorded as integrity warnings.

OpenAI Batch judging is the default evaluation path. An interrupted submitted
batch can be resumed without re-running retrieval:

```bash
.venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<run-id>
```

Batch payloads must contain valid explicit `score` and `hallucination` fields;
partial or malformed output keeps its reconciliation manifest and is never
converted into a paper metric. Post-hoc `kfold_analysis.py` and
`paired_bootstrap.py` produce uncertainty and paired-difference artifacts.

---

## Ablation toggles

Indexing-time ablations are driven by environment toggles read in `core/config.py`:

| Variable | Default | Effect when set to `false` |
|---|---|---|
| `RAG_ABLATION_Q_PLUS` | `true` | Stage 2 Q⁺ expansion disabled (also disables offline HOP-edge construction) |
| `RAG_ABLATION_Q_MINUS` | `true` | Stage 1 Q⁻ channel disabled |

`{full, Q⁻-only, Q⁺-only}` is the paper's reported ablation matrix. Source
text, including pipe-delimited text, is indexed as-is; there is no
table-to-text generation branch.

Each ablation lives under its own corpus tag so indexed graphs never collide.

---

## Key hyperparameters

Full list in the paper appendix; the most important:

| Parameter | Value | Where |
|---|---|---|
| `CHUNK_SENTENCES` | 6 | fixed-size chunking window (sentences per chunk) |
| `MIN_CHUNK_SENTENCES` | 2 | trailing short window merges into the previous chunk below this |
| `L_hop` | 5 | max outgoing HOP edges per source chunk |
| `K_hop` | 15 | retained candidates per question and target channel |
| ANN floor | 50 | minimum pool; raised per source by its own representations + `K_hop` |
| same-need weight | 0.5 | Q+→Q+ RRF support; never sufficient to create `HOP_ANSWER` |
| Stage 1 weights | 0.7 / 0.3 | $Q^-$ / body |
| Stage 2 weights | 0.6 / 0.4 | $Q^+$ / $Q^-$ support |
| RRF `k` | 60 | $w_v=1.3$, $w_t=1.0$ |
| Embedding dim | `NEO4J_VECTOR_DIMENSIONS` | must match the configured embedding model's actual output dim (see `CLAUDE.md` "Model / inference infra") |

Offline HOP construction is rank-based and has no learned reranker, fixed
cosine threshold, domain gate, or heuristic gate. Query-time candidate
ordering is likewise threshold-free.

---

## What is intentionally **not** in this repository

The system does not include a reflective agent loop. A five-stage Perception/Planning/Execution/Reflection/Refinement loop was explored on top of the same indexing pipeline; it measured net-negative on this system itself and at best baseline-dependent across the four GraphRAG systems tried. The paper reports the retrieval-only configuration and the code here mirrors that decision.

---

## License

MIT — see [LICENSE](LICENSE).
