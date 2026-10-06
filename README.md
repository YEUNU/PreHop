# Prehop: Multi-Hop Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Prehop is a graph-based retriever for multi-hop retrieval-augmented generation
(RAG). During indexing it generates questions for every passage and stores
links between passages; at query time it retrieves initial passages, expands
the stored links once, reranks the combined pool with an LLM and generates an
answer from the returned passages. This repository provides the implementation and experiment code for
**Reached but Not Retained: Candidate Admission under Reranker Input Limits in Graph-Based Multi-Hop RAG**.
The paper separates evidence reached by graph expansion from evidence retained
under an LLM reranker input ceiling, using Prehop as a controlled test bed.

[Method and implementation](docs/METHOD.md) ·
[Reproducing the experiments](docs/REPRODUCING.md) ·
[Runtime setup](docs/SETUP.md)

## How Prehop works

![Prehop architecture: offline indexing with generated questions and stored links, then one-step expansion, LLM reranking and answer generation](docs/figures/prehop_architecture.png)

Each passage receives two question sets from an LLM. **Q−** questions are
answerable from the passage itself; **Q+** questions ask for information that
lies beyond it. Index-time matching of a passage's Q+ to another source's Q−
creates a directed **HOP** link, and adjacent passages of one source are joined
by **NEXT** links. The question-role formulation follows
[HopRAG](https://arxiv.org/html/2502.12442v2#S3.S2).

![A Q+ question of passage A matches a Q− question of passage B and creates the HOP link A to B](docs/figures/question_link_example.png)

At query time the original question searches the body, Q− and Q+ indexes; the
fused hits form the initial passages. Prehop follows outgoing HOP links and
both NEXT directions once from every initial passage, keeps the initial
candidates in the pool, orders the pool by a fused score and passes it to one
LLM reranking request that returns up to 12 passages for answer generation.
Stored links propose evidence; the query decides its relevance only through
scoring and reranking. The [method guide](docs/METHOD.md) gives the exact
fusion, scoring and reranking rules.

## Installation

Run the commands below from the repository root on Linux with Bash, Git, curl
and `uv` installed. `uv` can provision Python 3.12. Prepare Neo4j 5.26.21 and an
OpenAI-compatible LiteLLM gateway serving generation and embedding models.
Those services may run on another machine.

```bash
uv sync --locked --python 3.12 --extra dev
test -f .env || cp .env.example .env
```

Edit `.env` to match your infrastructure:

| Setting | Value to supply |
| --- | --- |
| `NEO4J_URI` | Bolt URI, including your host and port |
| `NEO4J_USER`, `NEO4J_PASSWORD` | Database credentials |
| `NEO4J_DATABASE` | Optional database name; defaults to `neo4j` |
| `RAG_INFERENCE_BASE_URL` | Gateway API base URL, including `/v1` when applicable |
| `RAG_INFERENCE_API_KEY` | Gateway credential |
| `RAG_GENERATION_MODEL` | Registered generation model; paper alias is in `.env.example` |
| `RAG_EMBEDDING_MODEL` | Registered embedding model; paper alias is in `.env.example` |
| `NEO4J_VECTOR_DIMENSIONS` | Actual embedding dimensions; paper value is in `.env.example` |

The gateway must support chat completions with JSON-schema output and embeddings.
The embedding model must return the configured number of dimensions. Changing
models or method settings defines a different experiment.

Select the environment and request concurrency, then check the connections:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export RAG_EXECUTION_PROFILE="$PWD/configs/execution_profiles/direct-8.json"
./run_servers.sh all
```

`run_servers.sh` checks the configured Neo4j connection and both gateway model
names. It can start a default local Neo4j installation if needed; remote hosts
and custom ports must already be running. It does not start model processes.
Shell launchers and the smoke runner load `.env`, preserving exported overrides.
Keep the exports above in the shell used for the remaining commands.
Settings resolve in this order: registry defaults, `.env`, exported values,
then the selected execution profile for its throughput fields. Choose throughput
in the profile; it overrides matching concurrency variables. Python and shell
entrypoints use the same resolver. See the
[configuration contract](docs/SETUP.md#configuration-ownership-and-precedence)
for ownership and precedence.

## Quick start

With the services ready, run a small real experiment using the included
two-document fixture. No benchmark download is needed:

```bash
export DEMO_RUN="demo-$(date +%Y%m%d-%H%M%S)-$$"
"$PYTHON_BIN" scripts/paper_cold_canary.py "$DEMO_RUN" prehop multihoprag --attempt a1
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
path = Path("data/results") / os.environ["DEMO_RUN"] / "cold_v2/a1/multihoprag/prehop/query.json"
result = json.loads(path.read_text())
print("Question:", result["query"])
print("Answer:", result["answer"])
print("Retrieved passages:", len(result["documents"]))
print("Saved result:", path)
PY
```

A successful execution prints `cold_canary_passed` and the generated
answer. The fixture's expected city is **Larkhaven**; the recorded answer lets
you inspect generation quality. The command checks execution, not exact answer
equality. Each invocation above creates a fresh namespace and preserves existing
graphs. See [smoke outputs](docs/REPRODUCING.md#completion-and-continuation) for metadata.

## Run the benchmarks and inspect scores

Prepare both datasets once in a fresh checkout. Do not rerun preparation while
an experiment uses those files:

```bash
"$PYTHON_BIN" scripts/datasets/prepare_multihoprag.py
"$PYTHON_BIN" scripts/datasets/prepare_hotpotqa_hipporag.py --download
```

Run Prehop on both complete question populations and export their scores:

```bash
export BENCH_RUN="prehop-$(date +%Y%m%d-%H%M%S)-$$"
bash scripts/run_paper_target.sh multihoprag prehop "$BENCH_RUN-multihoprag"
bash scripts/run_paper_target.sh hotpotqa prehop "$BENCH_RUN-hotpotqa"

"$PYTHON_BIN" scripts/export_official_results.py \
  "data/results/$BENCH_RUN-multihoprag/prehop/multihoprag/seed_42/prehop_multihoprag.json" \
  "data/results/$BENCH_RUN-hotpotqa/prehop/hotpotqa/seed_42/prehop_hotpotqa.json" \
  --output-dir "data/results/$BENCH_RUN-tables"
cat "data/results/$BENCH_RUN-tables/multihoprag/comparison.csv"
cat "data/results/$BENCH_RUN-tables/hotpotqa/comparison.csv"
```

These runs build the full corpus indexes and make model calls for 2,556 and
1,000 questions. They take substantially longer than the smoke test. To resume,
repeat the target command with its original run ID; keep the prepared corpus,
settings and index unchanged. Choose a new ID for a new experiment.

Each result JSON contains answers, ordered `retrieved_sources` and per-question
scores. Its neighboring `.official.json` contains metric summaries; exports
also produce CSV tables. `failed_rows` records terminal query failures, whose
quality scores remain zero. Completion alone does not imply every query succeeded.

The dense baseline (`naive`) uses the same environment, fixed passage windows,
embedding model, Neo4j vector index and gateway. Run and export both
dataset/system pairs for the comparison:

```bash
export COMPARISON_RUN="comparison-$(date +%Y%m%d-%H%M%S)-$$"
results=()
for dataset in multihoprag hotpotqa; do
  for method in prehop naive; do
    run_id="$COMPARISON_RUN-$dataset-$method"
    bash scripts/run_paper_target.sh "$dataset" "$method" "$run_id" || break 2
    results+=("data/results/$run_id/$method/$dataset/seed_42/${method}_${dataset}.json")
  done
done
if [ "${#results[@]}" -eq 4 ]; then
  "$PYTHON_BIN" scripts/export_official_results.py "${results[@]}" \
    --output-dir "data/results/$COMPARISON_RUN-tables"
fi
```

These commands evaluate each system's benchmark answer pipeline. The paper's
comparison with a shared answer generator needs the separate
[saved-evidence answer replay](docs/REPRODUCING.md#compare-answer-generation-over-saved-evidence). The
[experiment guide](docs/REPRODUCING.md) covers that distinction, the HOP/NEXT
expansion controls and timing measurements. New runs produce their own results;
identical settings do not guarantee identical generated answers or times.

## Experimental results

Prehop supports controlled comparisons of candidate acquisition and LLM
reranking. The values below describe the reported configurations; new runs can
produce different outputs even at temperature zero. Differences in percentage-valued
metrics use percentage points (pp).

| Comparison | MultiHop-RAG result | Evaluation scope |
| --- | --- | --- |
| One-step expansion versus fixed initial passages | +0.0195 MAP@10; +4.89 pp in QA | Same initial passages and LLM reranking |
| Body-only direct retrieval versus the one-step graph | +5.67 pp in QA | Same per-query reranker input ceiling; separate initial passages |
| Four-step reachable versus rank-fusion admitted recall | 94.60% versus 82.55% annotated recall | Reachable evidence versus evidence admitted under the original input ceiling |
| Four-step similarity-only admission versus rank fusion | +4.29 pp admitted recall; +3.02 pp returned recall | Same neighborhood and input ceiling; QA difference is unresolved |

![Candidate coverage as multi-channel direct retrieval retains additional passages](docs/figures/direct_prefix_coverage.png)

![Annotated recall before and after admission at increasing expansion depths](docs/figures/v2_depth_admission.png)

Appendix K reports the fixed-initial-passage ablation, evidence partition and
direct-prefix coverage. Intermediate prefixes measure coverage only; only the
full direct pool is matched to the one-step input ceiling.

The fixed-initial, three-policy and four-step comparisons use separate
recorded generations. They answer different questions and are not one combined
leaderboard. Direct retrieval's returned context can contain more annotated evidence
than the entire one-step graph pool, locating part of the recall deficit before
LLM reranking. Body-only direct retrieval retains the MultiHop-RAG advantage
under the same input ceiling; its QA difference from four-step similarity-only admission
is unresolved.

Dataset conditions matter. Initial evidence recall is 66.98% on MultiHop-RAG
and 94.83% on the reduced HotpotQA corpus. Relative to the initial mean recall
deficit, one-step expansion recovers 29.50% and 45.11%, respectively. The smaller
absolute HotpotQA gain does not imply less recovery relative to initially
missing evidence. These are descriptive retrieval ratios, not QA metrics or a
causal comparison of datasets.

See the [experiment inventory](docs/REPRODUCING.md#paper-experiment-inventory)
for inputs, paired intervals, repeated-generation comparisons and limitations.
Figure exports are included in `docs/figures/`; exact reconstruction of archived
results requires the saved inputs listed in the reproduction guide.

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py`, `cli/` | Indexing, benchmark and shared answer-generation entry points |
| `models/prehop/` | Passage chunking, question generation, link construction, hybrid retrieval, one-step expansion, scoring and LLM reranking |
| `models/naive/` | The dense title/body baseline on the same index infrastructure |
| `core/` | Configuration, inference transport, structured outputs, evaluation, checkpoints and run identity |
| `scripts/` | Dataset preparation, experiment runners, controlled analyses, statistics and exports |
| `utils/` | Metrics, HotpotQA support projection, prompts and provenance |
| `configs/` | Execution profiles, the smoke fixture and prompt-development question IDs |
| `docs/` | The three guides and the figure exports used on this page |
| `tests/` | Unit tests with mocked services; integration tests are marked |

## Development

Run the unit tests from the repository root using the prepared environment:

```bash
.venv/bin/python -m pytest -q -m "not integration"
```

Unit tests use fixtures and mocked Neo4j/inference boundaries. Some tests need
optional saved experiment artifacts and are skipped when those are absent.
Integration tests require configured services and are explicitly marked.
Tests check implementation behavior; they do not reproduce paper scores.

Keep retrieval and indexing changes in `models/`, shared execution and
configuration in `core/`, and CLI orchestration in `cli/` and `scripts/`.
The [module map](docs/METHOD.md#code-ownership) describes those responsibilities.
Method, prompt or metric changes define new experimental conditions; preserve
the original results and use a new run ID for comparison.

## Data and reproducibility

The source checkout contains code, tests, configuration examples, dependency
specifications and static figures. Downloaded corpora, indexes, checkpoints,
traces and generated results are stored locally under `data/` and excluded from
Git. Reuse explicit result paths when exporting scores; the exporter records
source hashes and preserves the inputs.

New benchmark runs are supported directly by this checkout. Exact replay of
archived experiments additionally requires the graphs, candidate pools and
recorded outputs listed in [Reproducing experiments](docs/REPRODUCING.md).

## License and attribution

Repository-owned code is released under the [MIT License](LICENSE).
Datasets retain their respective licenses. HotpotQA source attribution and
its reduced-corpus evaluation setting are documented in the
[dataset guide](docs/REPRODUCING.md#hotpotqa-source-and-population).
