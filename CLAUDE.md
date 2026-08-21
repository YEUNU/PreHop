# CLAUDE.md

Operational guide for running **Prehop** (paper name on resubmission:
**HypoHop**) locally and for its benchmark work. Architecture/results live in
`README.md`; this file captures the run-environment knowledge that isn't
obvious from the code.

## Project identity

Prehop targets **NAACL 2027** via the ARR October 2026 cycle (submission
deadline 2026-10-12, NAACL commitment 2026-12-20~23), with a deliberately
narrow contribution scope after an earlier submission under a different
name/scope was rejected (criticism: too many bundled components obscuring
the real contribution, a directional-weighting claim that wasn't
statistically significant, and low absolute performance).

**Contribution scope** (core claims, kept narrow on purpose):
1. Offline HOP-edge pre-scoring enabling deterministic query-time 1-hop
   expansion with no per-hop LLM reasoning — the latency win vs HopRAG-style
   per-hop LLM reasoning is the headline claim. Edges are scored from the
   Neo4j vector index's own ANN cosine-similarity score (see "Reranking"
   below) — no cross-encoder model is used anywhere in this scoring.
2. Q-/Q+ generative chunk split — framed as the structural precondition for
   HOP edges to exist at all ("no Q+ → no HOP-edge material"), not a feature
   that needs its own significance proof. This is the architecture's core
   identity and is never on the chopping block.
3. Q-/Q+ combined-channel (non-directional) retrieval quality gain.
   Directional Q-/Q+ weighting is explicitly OUT of the headline claim (not
   statistically significant in the earlier submission).
4. There is no OCR, adaptive chunking, or rolling summary in this codebase
   (not just de-scoped — not present) — see "Architecture notes" below.

## Architecture notes

- **No OCR.** Corpora are plain text from the start — there is no PDF/image
  ingestion path anywhere in this repo. `--mode` only supports
  `index|benchmark|benchmark_all`.
- **Chunking is fixed-size.** `models/prehop/indexing/chunking.py` windows
  each page into `RAGConfig.CHUNK_SENTENCES`-sentence chunks (env
  `RAG_CHUNK_SENTENCES`, default 6); a trailing window shorter than
  `RAGConfig.MIN_CHUNK_SENTENCES` merges into the previous chunk. No
  embedding-similarity-based splitting, no page-grouping, no rolling context
  fed into Q-/Q+ generation. The table-to-text fallback (`RAG_ABLATION_TABLE`)
  is unrelated to this and works independently.
- **No cross-encoder reranker anywhere in this system's own pipeline.** See
  "Reranking" below.
- **Benchmark suite is MultiHop-RAG + HotpotQA + MuSiQue** (see "Multi-hop
  dataset suite" below) — no FinanceBench support. The 3-way `answer_label`
  taxonomy (Correct/Incorrect/Refusal) is shared/dataset-agnostic infra, used
  by every dataset. `RAGConfig.DOMAIN` defaults to `"news"`;
  `RAGConfig.COMPANY_ANCHORING` defaults off; `--domain financial` remains a
  valid manual override should a company-anchored dataset get added later,
  but no current dataset sets it.

## Reranking

**There is no reranker/cross-encoder model in this system's own pipeline.**
An earlier iteration used a dedicated reranker model (`Qwen3-Reranker-0.6B`,
served separately) for both query-time retrieval reranking and, critically,
for scoring candidate HOP edges at indexing time. That model isn't part of
the current inference setup, so both uses were replaced with **embedding
cosine-similarity**:

- **HOP-edge construction** (`models/prehop/indexing/hop_edges.py`): edges
  are scored directly from the score Neo4j's `db.index.vector.queryNodes`
  ANN query already returns — no extra model call. Gated on
  `RAGConfig.HOP_THRESHOLD` (env `RAG_HOP_THRESHOLD`, default 0.82).
- **Query-time reranking** (`models/prehop/retrieval/rerank.py`,
  `traversal.py`): a shared `_embedding_rerank_scores(query, texts)` helper
  embeds the query and candidates and scores by cosine similarity. Gated on
  `RAGConfig.RERANKER_THRESHOLD` (env `RERANKER_THRESHOLD`, default 0.4).

Both thresholds were originally calibrated for cross-encoder classifier
scores (roughly a 0-1 probability) and now gate raw bi-encoder cosine
similarity instead — **they likely need empirical re-tuning** once a real
benchmark run is available; don't assume the old defaults are still correct.

**Baselines still need a reranker.** `models/hoprag/hoprag_adapter.py` and
`models/ms_graphrag/ms_adapter.py` implement other papers' methods and still
call `core/vllm_client.py`'s `rerank()` (unchanged) — running those two
baselines for comparison requires a live reranker endpoint
(`VLLM_RERANK_URL`), which is a separate infra decision from Prehop's own
model setup below.

## Model / inference infra

Generation and embeddings both route through a **LiteLLM proxy**
(OpenAI-compatible, internal network) instead of directly-served local vLLM
models — set `VLLM_URL` / `VLLM_EMBED_URL` / `VLLM_API_KEY` /
`VLLM_SERVED_MODEL_NAME` / `VLLM_SERVED_EMBED_MODEL_NAME` in `.env` to point
at it. Verified working: `gemma-4-31b-it` (chat) and `qwen3-embedding-4b`
(embeddings, **dim=2560**). If the served model set changes, re-verify with a
real request and update `NEO4J_VECTOR_DIMENSIONS` in `.env` to match the new
embedding dimension — this is set at Neo4j vector-index creation time, so a
dimension change needs a fresh index. There is no reranker model on the
proxy at all (see "Reranking" above). Do not commit real API keys/tokens to
this repo — they belong in the gitignored `.env` only.

## Pipeline at a glance

```
python main.py --mode {index|benchmark|benchmark_all} \
  --strategy {prehop|naive|hoprag|ms_graphrag} \
  --dataset <corpus_dir> --queries_file <queries.json> \
  --corpus-tag <tag> --model generation-model
```

- `benchmark_all` loops strategies `["naive","prehop","hoprag","ms_graphrag"]`.
- prehop / naive / hoprag store their graphs in **Neo4j** (label-prefixed by
  corpus tag). ms_graphrag writes **parquet** under
  `data/ms_graphrag_output/<tag>/` (not Neo4j).
- **Never pass `--clear-graph` on a resume/re-run** — it wipes ALL Neo4j data,
  including the other baselines' indexes.

## Setup (uv-managed, reproducible for a fresh clone)

The Python env is managed with **uv**. `pyproject.toml` is the canonical
dependency list (there is no `requirements.txt`).

```bash
uv venv --python 3.12 .venv               # .python-version pins 3.12
VIRTUAL_ENV=.venv uv pip install -e .      # installs vllm/torch/.../flashinfer
cp .env.example .env                        # set NEO4J_PASSWORD, OPENAI_API_KEY, LiteLLM proxy creds
.venv/bin/python -m spacy download en_core_web_sm   # required by the hoprag baseline
```

- This venv has **no `pip`** (uv-managed). Check installs with
  `VIRTUAL_ENV=.venv uv pip show <pkg>` or `.venv/bin/python -c "import pkg"` —
  `.venv/bin/pip` does not exist and gives false negatives.
- The flat layout needs `[tool.setuptools] py-modules = []` (already set) or the
  editable build aborts with "Multiple top-level packages".
- Dev/test tooling (`pytest`, `pytest-asyncio`) is an optional extra:
  `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"`.
- The run scripts do **not** require activating the venv: a shared
  `resolve_python` helper (`scripts/lib.sh`) finds `.venv/bin/python`
  automatically (override with `PYTHON_BIN`, falls back to system python). So a
  fresh clone runs `./run_*.sh` directly after `uv venv && uv pip install -e .`.

## Multi-hop dataset suite

Headline benchmark suite: **MultiHop-RAG + HotpotQA + MuSiQue** — all three
share one query schema (`evidence_docs` + `evidence_facts` + `category` +
a `dataset` marker) and one evaluator
(`utils.metrics.evaluate_multihoprag_response`: fact-level MRR@10/MAP@10/
Hits@10/4 + LLM judge), so adding a dataset means writing a `prepare_*.py`
that emits the same schema, not new evaluator code. Dataset is detected via
the per-query `dataset` marker in `cli/benchmark.py`, NOT the filename.
`RAGConfig.DOMAIN` (financial|news) is auto-set to "news" for all three via
`main.py`'s `_preset_rag_domain()` and `core/config.py`'s
`_DOMAIN_BY_DATASET` — this selects the news-framed prompt variants
(HOPRAG_PROMPT, QUERY_REWRITE_PROMPT, SEARCH_CONTINUATION_PROMPT,
`answer_role()`) and turns off company-anchoring (`COMPANY_ANCHORING`
defaults to `DOMAIN=="financial"`).

```bash
# Prep (one-time; downloads + builds corpus + full query pool)
python data/prepare_multihoprag.py
python data/prepare_hotpotqa.py             # HF parquet, distractor config, --limit 2000 default
python data/prepare_musique.py              # HF JSONL, answerable dev split, --limit 2000 default

# Stratified n≈200 sample for benchmark figures (graph baselines are slow —
# hoprag ~160s/query — so figures run on a balanced sample, not the full set)
python data/make_multihoprag_sample.py --per-type 50 --seed 42   # legacy dedicated script
python data/make_sample.py --dataset hotpotqa --per-type 100 --seed 42   # 2 types -> n=200
python data/make_sample.py --dataset musique  --per-type 67  --seed 42   # 3 hop-counts -> n=201

# Run (index all 4 strategies + benchmark on the sample)
./run_multihoprag.sh all
./run_dataset.sh hotpotqa all
./run_dataset.sh musique all
```

- **MultiHop-RAG**: yixuantt/MultiHop-RAG (609 articles, 2556 queries;
  question_type comparison/inference/temporal/null). **GitHub LFS gotcha**:
  must use `media.githubusercontent.com/media/...` (raw endpoint returns LFS
  pointer text). Titles carry HTML entities (`&#039;`) → `html.unescape`
  applied. Corpus dir `data/multihoprag_corpus/`.
- **HotpotQA**: distractor config from `hotpotqa/hotpot_qa` on HuggingFace
  (parquet export, parsed with `pyarrow` — no `datasets` library dependency).
  distractor (not fullwiki) because each question ships ~10 candidate
  paragraphs (2 gold + distractors), which is exactly the "union of
  per-question contexts into one corpus" shape the other two datasets
  already use. `type` field (bridge/comparison) is the category.
  Corpus dir `data/hotpotqa_corpus/`.
- **MuSiQue**: answerable dev split from `dgslibisey/MuSiQue` on HuggingFace
  (JSONL). Gold evidence is paragraph-level (`is_supporting` flag), not
  sentence-level like HotpotQA — `evidence_facts` is the full paragraph text.
  Category is the hop count parsed from the id prefix (`2hop`/`3hop`/`4hop`).
  `question_decomposition` (per-hop sub-question/sub-answer) is downloaded
  but not currently used by the benchmark. Corpus dir `data/musique_corpus/`.
- Both new prep scripts cache their raw download (`data/hotpotqa_distractor_validation.parquet`,
  `data/musique_ans_v1.0_dev.jsonl`) and re-parse locally on `--limit` change
  — delete the cached file to force a re-download.
- K-fold post-hoc analysis (any of the three): `scripts/kfold_analysis.py
  --run-dir <run> --k 5` (partitions one run's per-query details into folds
  → mean/std/CI per metric).
- Judge via OpenAI Batch API (opt-in, 50% cheaper): set `RAG_JUDGE_BATCH=true`
  with an OpenAI `EVAL_MODEL`. Async by default (`RAG_JUDGE_BATCH_ASYNC=true`)
  — each strategy submits its batch and continues without blocking; a
  reconcile pass after all strategies polls every batch in parallel and
  patches scores. Manual re-poll of an interrupted run:
  `python scripts/reconcile_batch_judge.py --run-dir <run>`.

## Local single-GPU run environment (RTX 5000 Ada, 32 GB)

`run_servers.sh` defaults to a 2-GPU layout but the per-service GPU is now
env-configurable — on a single GPU run
`GEN_GPU=0 EMBED_GPU=0 RERANK_GPU=0 ./run_servers.sh all` (watch total
`--gpu-memory-utilization` when co-locating). The manual launch below is
equivalent and what was validated on this box. Attention backend:
`flashinfer-python` IS installed (transitive dep of
vllm, 0.6.8.post1), but vLLM defaults attention to FLASH_ATTN/FlashAttention 2
when `--attention-backend` is unset (FLASHINFER is listed as available; sampling
already uses FlashInfer). To use FlashInfer for attention too, pass
`--attention-backend FLASHINFER` (requires a server restart to take effect).
FLASH_ATTN works fine as-is. torch 2.11.0+cu130, system CUDA 13.2 — compatible.
Validated layout (generation now typically comes from the LiteLLM proxy
instead — see "Model / inference infra" above; embed/rerank below still
apply when running locally):

| Port  | Model                      | served-name      | util | max-len | role           |
|-------|----------------------------|------------------|------|---------|----------------|
| 28000 | Qwen/Qwen3-4B-Instruct-2507| generation-model | 0.45 | 16384   | generation     |
| 18082 | Qwen/Qwen3-Embedding-0.6B  | embedding-model  | 0.15 | 8192    | embeddings     |
| 18083 | Qwen/Qwen3-Reranker-0.6B   | reranker-model   | 0.20 | 4096    | reranking (baselines only — see "Reranking" above) |

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name generation-model --host 0.0.0.0 --port 28000 \
  --gpu-memory-utilization 0.45 --max-model-len 16384 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code > logs/vllm_gen.log 2>&1 &
# embed (18082, util 0.15) and rerank (18083, util 0.20) similarly.
```

Notes:
- gen + embed + rerank ≈ 0.80 util fits in 32 GB (≈ 26.8 GiB observed). Start
  servers sequentially (KV-cache contention can kill a second small model that
  comes up simultaneously).
- **Indexing needs only gen + embed** (Prehop's own pipeline never calls a
  reranker anymore — see "Reranking"). A rerank server is only needed to run
  the HopRAG/MS-GraphRAG baselines.
- venv: flat layout needs `[tool.setuptools] py-modules = []` in pyproject for
  the editable build to pass.

## Neo4j

- `bash run_servers.sh neo4j` → docker `prehop-neo4j` (neo4j:5-community, bolt
  7687, http 7474, creds `neo4j/1q2w3e4r`).
- **Data persists across reboots**: bind-mounted host dir `neo4j_data/` → /data.
  After a reboot just `docker start prehop-neo4j` — prehop/naive/hoprag data
  is intact. Verify before re-indexing.
- Query from CLI:
  `docker exec prehop-neo4j cypher-shell -u neo4j -p 1q2w3e4r --format plain "<cypher>"`
- This Neo4j container also holds data from earlier, unrelated indexing runs
  under other corpus tags — keep corpus-tags namespaced (e.g. a
  dataset-name prefix) so reruns never collide with existing `PR_<tag>_*`
  labels already in the database.

## Indexing failure tracking

A single chunk/file/HOP-edge/summarize error no longer silently degrades or
(usually) takes down the whole run — per-file errors are isolated
(`cli/index.py`'s `process_file`), so other files keep indexing. Whenever
`run_indexing` finishes with any failures, the full list (not just the
10-item log preview) is written to
`data/index_failures/<strategy>_<corpus_tag>_<timestamp>.json` —
`{strategy, corpus_tag, dataset_path, total_files, succeeded, failed,
failures: [{item, stage, error}]}`, `stage` one of `read|index|task|
hop_edges|summarize`. Check this file after a partial run instead of
grepping scrollback.

## Reboot / interruption recovery

1. `docker start prehop-neo4j`; verify labels survived:
   `CALL db.labels()` + per-label counts.
2. Relaunch vLLM embed (+ rerank only if running a baseline); generation
   comes from the LiteLLM proxy (see "Model / inference infra") unless
   falling back to a local gen server.
3. Re-run any unfinished index. **prehop/naive/hoprag** are durable in Neo4j —
   if a strategy's node counts look complete, skip it. **ms_graphrag** resumes
   from its own cache (`data/ms_graphrag_output/<tag>/_cache/extract_graph/`):
   re-run the identical index command and it replays cached chunks instantly,
   then continues from the break. No `--clear-graph`.

## Neo4j data layout & integrity checks

Corpus-tag prefixes labels so datasets coexist: `PR_<tag>_*` (prehop),
`NA_<tag>_*` (naive), `HO_<tag>` (hoprag).

prehop schema (written by `models/prehop/indexing/graph_writer.py`):
- `PR_<tag>_Document {filename, corpus, title, summary, updated_at}`
- `PR_<tag>_Chunk {id, text, chunk_summary, q_minus_text, q_plus_text,
  embedding, body_embedding, q_minus_embedding, q_plus_embedding}`
- Relationships: `NEXT` (chunk sequence; edges = chunks − docs),
  `HOP` (rank-based, see "Reranking" for the current scoring mechanism,
  MERGE-idempotent)
- Indexes: 3 vector (body/qminus/qplus) + 3 fulltext + range on id/filename

By-design partial coverage (NOT incomplete indexing):
- `q_minus_embedding` < chunks: empty Q⁻ falls back to body embedding for the
  primary `embedding` (graph_writer.py).
- `q_plus_embedding` < chunks: Q⁺ passes `_is_high_quality_q_plus` gating; the
  surviving count equals the "wrote N HOP edges over M Q+ chunks" log line.

Useful integrity probes:
```cypher
// no duplicate documents
MATCH (d:PR_<tag>_Document) RETURN count(d), count(DISTINCT d.title);
// hoprag edges are unique (start,end,question) — multi-question, not dupes
MATCH (a:HO_<tag>)-[r:HO_<tag>_p2a]->(b:HO_<tag>)
RETURN count(r), count(DISTINCT [elementId(a),elementId(b),r.question]);
// prehop property coverage
MATCH (c:PR_<tag>_Chunk) RETURN count(c), count(c.embedding),
  count(c.q_minus_embedding), count(c.q_plus_embedding);
```

## MultiHop-RAG domain-tuning mechanisms

Four knobs shape how the news/multi-hop domain is handled. All four numbers
cited below were measured before the core-only rewrite (chunking/reranking
have since changed) and need re-validation on a fresh benchmark run — treat
them as rationale for why each knob exists, not current-state results.

- **Company-anchoring gate** (`core/config.py` `COMPANY_ANCHORING`, default
  `DOMAIN=="financial"`, env `RAG_COMPANY_ANCHORING`). When OFF (news):
  `_extract_company_keys` returns `set()` (neutralizes the strict company
  filter in `rerank.py`/`traversal.py` + mismatch penalty + company boost), and
  `hop_edges.py` drops the same-company HOP filter. This was THE multihoprag
  fix: HOP edges 638 → ~5k, Judge 0.59 → 0.70, MRR 0.37 → 0.49.
- **`DEFAULT_TOP_K`** is domain-aware: **news=12, financial=8** (env
  `RAG_DEFAULT_TOP_K` overrides). top_k 8→12 lifted Hits@10 +3.5pp, DocRecall
  +5.9pp, Judge +5pp with negligible hallucination.
- **News metadata in synthesis context** (`text_utils._build_context_from_nodes`,
  gated `DOMAIN=="news"`). Injects `Source: … , Published: …` per context
  block using `published_at`/`pub_source` stored on Chunk + Document (parsed
  in `cli/index.py`, written by `graph_writer.py`). Fixed temporal-question
  over-abstention (temporal Judge 0.52→0.68).
- **Comparison query decomposition** (`_NEWS_QUERY_REWRITE_PROMPT`, gated via
  `DOMAIN=="news"` prompt select). Multi-subject queries ("A vs B") decompose
  into one single-subject variant per entity, reusing the existing
  `query_variants`/hybrid-RRF machinery. Fixed comparison-question evidence
  coverage (comparison Judge 0.54→0.66).

**Re-index note:** company-anchoring touches HOP-edge construction, so a
HOP-only rebuild suffices for that knob alone (chunks/Q±/embeddings
unchanged). For a fully reproducible from-scratch reindex, scoped-delete only
`PR_<tag>_{Chunk,Document}` (NEVER `--clear-graph`) and set
`RAG_CHUNK_CACHE=off` — otherwise the on-disk chunk cache
(`data/index_cache/v4/<tag>/`, see `chunking.py`) replays prior Q±/summaries.
top_k change needs NO reindex (benchmark only). News metadata also needs no
full reindex: backfill `published_at`/`pub_source` onto existing chunks+docs
by matching `c.source = <txt filename>` against the corpus txt headers (a
fresh index populates them automatically via `cli/index.py`).
