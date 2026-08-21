# Prehop / HypoHop: Indexing-Time HOP-Edge Pre-Scoring for Deterministic Multi-Hop Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **This README is mid-rewrite and partly stale** — see `CLAUDE.md` for the
> current, accurate state of the pipeline and run instructions. **No results
> table is published yet**; the system has not been benchmarked end-to-end.

Reference implementation of a GraphRAG framework whose core claim is
indexing-time HOP-edge pre-scoring: chunk-to-chunk semantic edges are
computed once, offline, from embedding similarity, so the query path expands
the graph deterministically with no per-hop LLM reasoning. The query path is
a thin two-stage hybrid retrieve over a graph built once offline — no agent
loop, no reflection, no refinement.

---

## What this repository is

Core indexing-time design, currently evaluated on MultiHop-RAG, HotpotQA, and MuSiQue:

1. **Predictive Knowledge Mapping** — every chunk receives dual hypothetical-query annotations ($Q^-$ for self-contained facts, $Q^+$ for outgoing dependencies), indexed separately. This is the structural precondition for HOP edges below, not a standalone feature.
2. **Rank-Based HOP Edges Pre-Built Offline** — chunk-to-chunk semantic edges are computed once, offline, from Q+/candidate embedding cosine similarity (no cross-encoder model); the query path never expands the graph with an LLM call.

Chunking is fixed-size (page-scoped sentence windows) — see `CLAUDE.md` "Architecture notes" for details.

The query path is deliberately thin: two-stage hybrid retrieve (Q⁻/body, then Q⁺ expansion), embedding-similarity rerank with top-up, deterministic 1-hop traversal over the pre-built NEXT/HOP edges, and a single LLM synthesis call with inline citations.

---

## Results

Not yet published — the system has not been benchmarked end-to-end since
the current core-only architecture landed. See `CLAUDE.md` "Multi-hop
dataset suite" for how to run one.

---

## Repository layout

```
prehop/
├── main.py                          # single CLI entry point (--mode index|benchmark|ocr)
├── cli/
│   ├── index.py                     # indexing runner
│   └── benchmark.py                 # benchmark runner (single + multi-seed)
├── core/
│   ├── config.py                    # RAGConfig — env-driven thresholds
│   ├── neo4j_service.py             # async Neo4j driver lifecycle
│   ├── vllm_client.py               # vLLM + OpenAI routing
│   └── schemas.py
├── models/
│   ├── prehop/                     # the paper's system
│   │   ├── graphrag.py              # GraphRAG facade; run_workflow() is the query entry point
│   │   ├── indexing/                # §3.1 — ocr, chunking, knowledge_mapping (Q-/Q+), hop_edges, graph_writer
│   │   ├── retrieval/               # §3.2 — hybrid (RRF), rerank, traversal, retrieve, rewrite, text_utils
│   │   └── schemas.py / state.py / trace.py
│   ├── naive/                       # baseline (sentence chunking + vector search)
│   ├── hoprag/                      # baseline (runtime hop traversal via official HopRAG)
│   └── ms_graphrag/                 # baseline (community-report retrieval via graphrag package)
├── utils/
│   ├── abstain.py                   # honest-abstain detection + shared 3-way answer_label
│   ├── metrics.py                   # combined judge + hallucination call
│   ├── similarity.py                # cosine_similarity (reranking + HOP-edge scoring)
│   ├── prompts/                     # indexing + retrieval + judge prompts
│   └── io.py / formatters.py / parsers.py / reporting.py / tool_definitions.py
├── data/
│   ├── prepare_multihoprag.py       # download/build the MultiHop-RAG corpus + queries
│   ├── prepare_hotpotqa.py          # download/build the HotpotQA (distractor) corpus + queries
│   ├── prepare_musique.py           # download/build the MuSiQue (answerable dev) corpus + queries
│   └── make_sample.py               # stratified n≈200 query sample for hotpotqa/musique/multihoprag
├── scripts/                         # lib.sh (resolve_python, wait_for_server), port-probe, env-check
├── tests/                           # chunking / retrieval / live-integration
├── run_servers.sh                   # start Neo4j + vLLM (gen / embed / rerank)
├── run_index.sh / run_benchmark.sh  # low-level, dataset-agnostic
├── run_multihoprag.sh               # per-dataset entry: index|benchmark|all
├── run_dataset.sh                   # per-dataset entry for hotpotqa|musique: index|benchmark|all
├── pyproject.toml                   # canonical dependency list (uv-managed)
└── README.md
```

---

## Installation

```bash
# Python 3.12+ (pinned in .python-version). The env is managed with uv.
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e .   # vllm, torch, transformers, neo4j, flashinfer, ...
.venv/bin/python -m spacy download en_core_web_sm   # required by the hoprag baseline

# Neo4j 5.x — Docker is simplest:
docker run -d --name prehop-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> neo4j:5

# Configure env vars
cp .env.example .env
# Required: NEO4J_PASSWORD, OPENAI_API_KEY (for the LLM judge)
```

`pyproject.toml` is the canonical dependency list. The run scripts auto-discover `.venv/bin/python` (override with `PYTHON_BIN`), so you do not need to activate the venv.

vLLM servers are launched by `run_servers.sh` (generation, embeddings; reranker only if you're running the HopRAG/MS-GraphRAG baselines — see `CLAUDE.md` "Reranking"; ports listed in `scripts/probe_ports.py`). Generation may instead route through a remote OpenAI-compatible proxy — see `CLAUDE.md` "Model / inference infra".

---

## Quick start

```bash
# 0) Prepare a dataset (downloads + builds corpus + queries)
python3 data/prepare_multihoprag.py

# 1) Start services (Neo4j + vLLM gen / embed)
#    Default GPU placement targets a 2-GPU box. Single GPU? Put all on GPU 0:
#    GEN_GPU=0 EMBED_GPU=0 RERANK_GPU=0 ./run_servers.sh all
./run_servers.sh all

# 2) Build the index
./run_index.sh --model prehop --dataset data/multihoprag_corpus --corpus-tag multihoprag

# 3) Benchmark
./run_benchmark.sh --model prehop --queries data/multihoprag_sample200_queries.json --corpus-tag multihoprag

# 4) Stop services
./stop_servers.sh all
```

Result JSON is written to `data/results/<timestamp>/prehop/<corpus_tag>/*.json` and includes per-query details, category breakdowns, the shared 3-way answer label (Correct / Incorrect / Refusal), and aggregate metrics.

### Per-dataset entrypoints

`run_multihoprag.sh` and `run_dataset.sh` wrap the steps above with each
dataset's corpus, queries, and tags so you don't pass them by hand:

```bash
./run_servers.sh all            # services first

# MultiHop-RAG
python3 data/prepare_multihoprag.py            # downloads corpus + full queries
./run_multihoprag.sh all                       # index all 4 + benchmark (sample100)
./run_multihoprag.sh benchmark --queries full  # or the full 2556-query set

# HotpotQA / MuSiQue
python3 data/prepare_hotpotqa.py && python3 data/make_sample.py --dataset hotpotqa --per-type 100
python3 data/prepare_musique.py && python3 data/make_sample.py --dataset musique --per-type 67
./run_dataset.sh hotpotqa all
./run_dataset.sh musique all
```

See `CLAUDE.md` "Multi-hop dataset suite" for corpus/query file details per dataset.

---

## Ablation toggles

Indexing-time ablations are driven by environment toggles read in `core/config.py`:

| Variable | Default | Effect when set to `false` |
|---|---|---|
| `RAG_ABLATION_TABLE` | `true` | Markdown tables left as raw pipe-delimited text instead of converted to prose |
| `RAG_ABLATION_Q_PLUS` | `true` | Stage 2 Q⁺ expansion disabled (also disables offline HOP-edge construction) |
| `RAG_ABLATION_Q_MINUS` | `true` | Stage 1 Q⁻ channel disabled |

Each ablation lives under its own corpus tag so indexed graphs never collide.

---

## Key hyperparameters

Full list in the paper appendix; the most important:

| Parameter | Value | Where |
|---|---|---|
| `CHUNK_SENTENCES` | 6 | fixed-size chunking window (sentences per chunk) |
| `MIN_CHUNK_SENTENCES` | 2 | trailing short window merges into the previous chunk below this |
| `τ_hop` (`HOP_THRESHOLD`) | 0.82 | HOP-edge cosine-similarity gate (offline construction + runtime traversal) |
| `τ_r` (`RERANKER_THRESHOLD`) | 0.4 | query-time embedding-rerank cosine-similarity gate |
| `L_hop` | 5 | max outgoing HOP edges per source chunk |
| `K_hop` | 15 | HOP candidate pool per source chunk |
| Stage 1 weights | 0.7 / 0.3 | $Q^-$ / body |
| Stage 2 weights | 0.6 / 0.4 | $Q^+$ / $Q^-$ support |
| RRF `k` | 60 | $w_v=1.3$, $w_t=1.0$ |
| Embedding dim | 1024 | Qwen3-Embedding-0.6B (local) — verify if switched to a remote embedding model |

`τ_hop`/`τ_r` were calibrated for the old cross-encoder reranker's classifier
scores and now gate raw cosine similarity instead — likely need empirical
re-tuning (see `CLAUDE.md` "Reranking").

---

## What is intentionally **not** in this repository

The system does not include a reflective agent loop. A five-stage Perception/Planning/Execution/Reflection/Refinement loop was explored on top of the same indexing pipeline; it measured net-negative on this system itself and at best baseline-dependent across the four GraphRAG systems tried. The paper reports the retrieval-only configuration and the code here mirrors that decision.

---

## License

MIT — see [LICENSE](LICENSE).
