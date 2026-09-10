# Prehop: Role-Activated Retrieval over Offline Question Links

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Prehop builds question-guided links between corpus chunks during indexing and
uses those stored links for multi-hop evidence retrieval. It generates Q−
(answered here) and Q+ (needed elsewhere) representations, connects Q+ to a
Q− owner in another source file, activates links through Q+ matches, and selects evidence
for answer synthesis. The incoming/outgoing question formulation follows
[HopRAG](https://arxiv.org/html/2502.12442v2#S3.S2); Prehop uses its own
original-query role search, stored-link expansion, and evidence selection.
The implementation uses the original question at every input length and runs
retrieval once. Initial query rewriting and evidence-conditioned re-search have
been removed. The completed MultiHop-RAG run covers all 2,556 questions; results and source
artifacts are recorded in [the result register](docs/RESULTS.md).

The primary comparison set is Prehop, Naive RAG, HopRAG, MS GraphRAG,
LightRAG, GFM-RAG, and LinearRAG. Strategy identities and pinned
revisions come from `core/strategy_registry.py`. HopRAG uses a separately
prepared runtime; see [runtime requirements](docs/RUNTIME_REQUIREMENTS.md#hoprag-runtime).

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
```

The second benchmark uses the original HippoRAG HotpotQA release: 9,221 passages
and 1,000 query rows (944 unique original questions). Preparation, duplicate-row
handling, and official scoring rules are described in
[HOTPOTQA](docs/HOTPOTQA.md). This is a reduced retrieval corpus; completed
results remain outstanding. Controlled A/B/C and Neo4j stored-versus-online
experiments are specified in [PAPER_ABLATION_DESIGN](docs/PAPER_ABLATION_DESIGN.md).

Generated indexes, logs, traces, and results stay under ignored local data and
log directories. Do not treat a smoke run or an indexing completion as a full
benchmark result.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): implementation modules and behavior.
- [Runtime requirements](docs/RUNTIME_REQUIREMENTS.md): external runtimes,
  transport, and native output recording.
- [Execution and measurement](docs/THROUGHPUT_EXECUTION.md): profiles, launch
  procedures, recovery, and cost definitions.
- [Results](docs/RESULTS.md): result status and artifact requirements.
- [Maintainer policy](CLAUDE.md): repository maintenance rules.
- [Representation ablations](docs/PAPER_ABLATION_DESIGN.md): comparisons, controls, and execution commands.

## Verification

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

## License

MIT. External methods retain their upstream licenses and run in isolated
environments where required.

Completed benchmark outputs are linked in [RESULTS](docs/RESULTS.md#evidence-locations).
Final completion records execution status without a separate paper-policy gate.
