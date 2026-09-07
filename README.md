# Prehop: Offline Question Links for Multi-Hop Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Prehop builds question-guided links between corpus chunks during indexing and
uses those stored links for multi-hop evidence retrieval. It generates Q−
(answered here) and Q+ (needed elsewhere) representations, connects Q+ to a
cross-document Q− owner, expands those links at query time, and selects evidence
for answer synthesis.

The primary comparison set is Prehop, Naive RAG, MS GraphRAG, LightRAG,
HippoRAG2, GFM-RAG, LinearRAG, and Youtu-GraphRAG. BrowseNet, HopRAG, and
PropRAG are callable legacy/reserve adapters; MS GraphRAG is a primary method,
not a legacy adapter.

## Installation

Requirements: Python 3.12, `uv`, Docker, Neo4j 5.26.21, and an
OpenAI-compatible LiteLLM gateway that serves the configured generation and
embedding models.

```bash
uv sync --locked

docker run -d --name prehop-neo4j -p 7474:7474 -p 7687:7687 \
  -e 'NEO4J_AUTH=neo4j/<your_password>' neo4j:5.26.21-community

cp .env.example .env
```

Set `NEO4J_PASSWORD`, `RAG_INFERENCE_BASE_URL`, `RAG_INFERENCE_API_KEY`,
`RAG_GENERATION_MODEL`, and `RAG_EMBEDDING_MODEL` in `.env`. The current remote
contract uses `gemma-4-31b-it` and `qwen3-embedding-4b` (2,560 dimensions).
`run_servers.sh` starts or validates Neo4j and validates the gateway; it does
not start model servers.

## Quick start

```bash
.venv/bin/python scripts/datasets/prepare_multihoprag.py
./run_servers.sh all

./run_index.sh --model prehop \
  --dataset data/multihoprag_corpus --corpus-tag multihoprag

./run_benchmark.sh --model prehop \
  --queries data/multihoprag_queries.json --corpus-tag multihoprag

./stop_servers.sh all
```

Dataset wrappers provide the representative full flows:

```bash
./run_multihoprag.sh index --model prehop
./run_multihoprag.sh benchmark --model prehop --queries full

.venv/bin/python scripts/datasets/prepare_musique.py
./run_dataset.sh musique all --model prehop
```

Generated indexes, logs, traces, and results stay under ignored local data and
log directories. Do not treat a smoke run or an indexing completion as a full
benchmark result.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): implementation modules and behavior.
- [Runtime requirements](docs/RUNTIME_REQUIREMENTS.md): external runtimes,
  transport, and validation.
- [Throughput execution](docs/THROUGHPUT_EXECUTION.md): current and historical
  profiles, launch procedures, and cost definitions.
- [Results](docs/RESULTS.md): result status and artifact requirements.
- [Maintainer policy](CLAUDE.md): repository maintenance rules.
- [Changelog](docs/CHANGELOG.md): chronological engineering changes.

## Verification

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

## License

MIT. External methods retain their upstream licenses and run in isolated
environments where required.
