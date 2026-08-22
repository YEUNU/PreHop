# CLAUDE.md

Operational guide for running **Prehop** (paper name on resubmission:
**HypoHop**) locally and for its benchmark work. Architecture/results live in
`README.md`; this file captures the run-environment knowledge that isn't
obvious from the code. For the history behind any decision below — what was
tried, what it replaced, why — see `docs/CHANGELOG.md`.

## Documentation structure

- `CLAUDE.md` (this file): current-state operational reference only. Never
  write "used to be X", "earlier iteration...", "originally...", before/after
  numbers, or the rationale behind a past decision here — that belongs in
  `docs/CHANGELOG.md`. When something changes, update the fact in place;
  don't leave the old fact plus a note describing the change.
- `docs/CHANGELOG.md`: reverse-chronological record of what was tried,
  changed, and why — one entry per decision/fix, newest first. Rationale,
  before/after numbers, and rejected approaches all live here.
- `docs/ARCHITECTURE.md`: per-module reference for prehop's own pipeline
  (indexing/retrieval) — what each file owns, key functions, pipeline
  diagrams. Update when a module's responsibility actually changes (e.g. the
  refactor that moved RRF math into `text_utils.py`), not on every commit.
- `README.md`: repo-level overview (what this is, install, quick start,
  ablation toggles, results once published) — outside this split, not
  governed by the rule below.
- `SUBMISSION_TARGET.md`: **gitignored, local-only** — target venue, cycle,
  deadlines, and rejection history. This repo is public; none of that
  belongs in any tracked file. If a decision's rationale needs a specific
  venue/date to make sense, phrase `CHANGELOG.md`'s entry generically
  ("an earlier submission under a different scope was rejected...") and
  point to `SUBMISSION_TARGET.md` for the specifics instead of naming them
  inline.

**Rule**: when you fix a bug or make a decision, the durable *fact* goes in
`CLAUDE.md` (or `ARCHITECTURE.md` if it's about module structure), and the
*story* goes in `CHANGELOG.md`. Never duplicate narrative into `CLAUDE.md`.
Never put a venue name, submission cycle, deadline, or rejection detail in
any tracked file — that always goes in `SUBMISSION_TARGET.md`.

## Project identity

Contribution scope is deliberately narrow — see `docs/CHANGELOG.md`
"Original scope narrowing" for why. (Target venue/deadline details live in
the local-only, gitignored `SUBMISSION_TARGET.md` — not in this file.)

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
   Directional Q-/Q+ weighting is explicitly OUT of the headline claim.
4. There is no OCR, adaptive chunking, or rolling summary in this codebase
   (not just de-scoped — not present) — see "Architecture notes" below.

## Architecture notes

- **No OCR.** Corpora are plain text from the start — there is no PDF/image
  ingestion path anywhere in this repo. `--mode` only supports
  `index|benchmark|benchmark_all|hop_rebuild`.
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
Edges and query-time candidates are both scored via **embedding
cosine-similarity**:

- **HOP-edge construction** (`models/prehop/indexing/hop_edges.py`): edges
  are scored directly from the score Neo4j's `db.index.vector.queryNodes`
  ANN query already returns — no extra model call. Gated on
  `RAGConfig.HOP_THRESHOLD` (env `RAG_HOP_THRESHOLD`, default 0.82).
- **Query-time reranking** (`models/prehop/retrieval/rerank.py`,
  used by both `retrieve.py` and `traversal.py`): a shared
  `_embedding_rerank_scores(query, texts)` helper embeds the query and
  candidates and scores by cosine similarity, via `rerank.py`'s
  `_rerank_and_select`. Gated on `RAGConfig.RERANKER_THRESHOLD` (env
  `RERANKER_THRESHOLD`, default 0.4). The query is passed through
  `_simplified_rerank_query` first — strips verbose/role-framing/
  output-format preludes before scoring, since long role-played queries
  otherwise collapse rerank scores.
- Q⁺ quality gate (`quality_gates.py::_is_high_quality_q_plus`): a Q⁺
  question is retained as a HOP anchor only when it carries at least 2 of 4
  signals (entity token, period token, metric token, source anchor).

Both `HOP_THRESHOLD`/`RERANKER_THRESHOLD` gate raw bi-encoder cosine
similarity and likely need empirical re-tuning once a real benchmark run is
available — don't assume the defaults are correct.

**Baselines still need a reranker.** `models/hoprag/hoprag_adapter.py` and
`models/ms_graphrag/ms_adapter.py` implement other papers' methods and still
call `core/vllm_client.py`'s `rerank()` — running those two baselines for
comparison requires a live reranker endpoint (`VLLM_RERANK_URL`), a separate
infra decision from Prehop's own model setup below.

## Model / inference infra

Generation and embeddings both route through a **LiteLLM proxy**
(OpenAI-compatible, internal network) — set `VLLM_URL` / `VLLM_EMBED_URL` /
`VLLM_API_KEY` / `VLLM_SERVED_MODEL_NAME` / `VLLM_SERVED_EMBED_MODEL_NAME` in
`.env` to point at it. Current models: `gemma-4-31b-it` (chat) and
`qwen3-embedding-4b` (embeddings, **dim=2560**). If the served model set
changes, re-verify with a real request and update `NEO4J_VECTOR_DIMENSIONS`
in `.env` to match the new embedding dimension — this is set at Neo4j
vector-index creation time, so a dimension change needs a fresh index. There
is no reranker model on the proxy at all (see "Reranking" above). Do not
commit real API keys/tokens to this repo — they belong in the gitignored
`.env` only.

## Pipeline at a glance

```
python main.py --mode {index|benchmark|benchmark_all|hop_rebuild} \
  --strategy {prehop|naive|hoprag|ms_graphrag} \
  --dataset <corpus_dir> --queries_file <queries.json> \
  --corpus-tag <tag> --model generation-model
```

- `benchmark_all` loops strategies `["naive","prehop","hoprag","ms_graphrag"]`.
- `hop_rebuild` (`prehop` only): deletes and rebuilds just the HOP edges for
  `--corpus-tag` under the current `RAG_HOP_THRESHOLD` — chunks/Q-/Q+/
  embeddings untouched, far cheaper than a full reindex. See "τ_hop/τ_r
  sensitivity sweeps" below.
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
`main.py`'s `_preset_rag_domain()` (checked against the `dataset`/
`queries_file` marker before heavy imports) — this selects the news-framed
prompt variants
(HOPRAG_PROMPT, QUERY_REWRITE_PROMPT, SEARCH_CONTINUATION_PROMPT,
`answer_role()`) and turns off company-anchoring (`COMPANY_ANCHORING`
defaults to `DOMAIN=="financial"`).

```bash
# Prep (one-time; downloads + builds corpus + full query pool)
python data/prepare_multihoprag.py
python data/prepare_hotpotqa.py --limit 0    # HF parquet, distractor config; --limit 0 = full 7,405 questions
python data/prepare_musique.py --limit 0     # HF JSONL, answerable dev split; --limit 0 = full 2,417 questions

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
  question_type comparison/inference/temporal/null). Corpus build pulls from
  `media.githubusercontent.com/media/...` — the raw GitHub endpoint returns
  an LFS pointer file instead of content for this dataset. Titles carry HTML
  entities (`&#039;`) → `html.unescape` applied. Corpus dir
  `data/multihoprag_corpus/`.
- **HotpotQA**: distractor config from `hotpotqa/hotpot_qa` on HuggingFace
  (parquet export, parsed with `pyarrow` — no `datasets` library dependency).
  Distractor (not fullwiki): each question ships ~10 candidate paragraphs (2
  gold + distractors) — the same "union of per-question contexts into one
  corpus" shape the other two datasets use. `type` field (bridge/comparison)
  is the category. Corpus dir `data/hotpotqa_corpus/`.
- **MuSiQue**: answerable dev split from `dgslibisey/MuSiQue` on HuggingFace
  (JSONL). Gold evidence is paragraph-level (`is_supporting` flag), not
  sentence-level like HotpotQA — `evidence_facts` is the full paragraph text.
  Category is the hop count parsed from the id prefix (`2hop`/`3hop`/`4hop`).
  `question_decomposition` (per-hop sub-question/sub-answer) is downloaded
  but not currently used by the benchmark. Corpus dir `data/musique_corpus/`.
- Both HotpotQA/MuSiQue prep scripts cache their raw download
  (`data/hotpotqa_distractor_validation.parquet`,
  `data/musique_ans_v1.0_dev.jsonl`) and re-parse locally on `--limit` change
  — delete the cached file to force a re-download. `--limit 0` (or any
  value `<= 0`) means the true full dataset.
- K-fold post-hoc analysis (any of the three): `scripts/kfold_analysis.py
  --run-dir <run> --k 5` (partitions one run's per-query details into folds
  → mean/std/CI per metric).
- Query-paired significance test (any of the three, dataset auto-detected
  from each result file's `dataset`/`corpus_tag` fields):
  `scripts/paired_bootstrap.py --prehop <prehop_run.json> --baselines
  <naive.json> <hoprag.json> <ms_graphrag.json> --out-dir <dir>` — pairs
  per-query scores by query string and bootstraps the mean diff to a 95% CI;
  a CI excluding 0 is a statistically separated win/loss. Per-fold CIs from
  a single run overlap across strategies and can't establish this on their
  own, which is why this script exists separately from kfold_analysis.py.
- τ_hop/τ_r sensitivity sweep: `scripts/threshold_sweep.py --param
  {hop|rerank} --values <comma-separated> --corpus-tag <tag> --queries_file
  <queries.json> --model <model> --out-dir <dir>`. `--param hop` rebuilds
  just the HOP edges per value (`main.py --mode hop_rebuild`, cheap — see
  "Pipeline at a glance") then benchmarks; `--param rerank` needs no rebuild,
  just repeated benchmark passes with `RERANKER_THRESHOLD` set. Each sweep
  point runs main.py as a fresh subprocess (RAGConfig reads its threshold
  env vars once at class-definition time, so mutating os.environ mid-process
  wouldn't take effect) with a deterministic `RAG_BENCHMARK_TIMESTAMP` so
  the result file is found without scraping stdout. Outputs
  `<out-dir>/{hop,rerank}_sweep.{json,csv}`.
- Judge via OpenAI Batch API (opt-in, 50% cheaper): set `RAG_JUDGE_BATCH=true`
  with an OpenAI `EVAL_MODEL`. Async by default (`RAG_JUDGE_BATCH_ASYNC=true`)
  — each strategy submits its batch and continues without blocking; a
  reconcile pass after all strategies polls every batch in parallel and
  patches scores. Manual re-poll of an interrupted run:
  `python scripts/reconcile_batch_judge.py --run-dir <run>`.

## Local single-GPU run environment (RTX 5000 Ada, 32 GB)

`run_servers.sh` defaults to a 2-GPU layout but the per-service GPU is
env-configurable — on a single GPU run
`GEN_GPU=0 EMBED_GPU=0 RERANK_GPU=0 ./run_servers.sh all` (watch total
`--gpu-memory-utilization` when co-locating). The manual launch below is
equivalent. Attention backend: `flashinfer-python` is installed (transitive
dep of vllm, 0.6.8.post1), but vLLM defaults attention to
FLASH_ATTN/FlashAttention 2 when `--attention-backend` is unset (FLASHINFER
is listed as available; sampling already uses FlashInfer). To use FlashInfer
for attention too, pass `--attention-backend FLASHINFER` (requires a server
restart to take effect). FLASH_ATTN works fine as-is. torch 2.11.0+cu130,
system CUDA 13.2 — compatible.

Current setup: generation + embeddings route through the LiteLLM proxy by
default (`gemma-4-31b-it` / `qwen3-embedding-4b`, dim=2560 — see "Model /
inference infra" above); the **reranker is the one component that's
genuinely local**, since the proxy has no reranker model:

| Port  | Model                      | served-name      | util | max-len | role           |
|-------|----------------------------|------------------|------|---------|----------------|
| 18083 | Qwen/Qwen3-Reranker-0.6B   | reranker-model   | 0.20 | 4096    | reranking (baselines only — see "Reranking" above) |

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/vllm serve Qwen/Qwen3-Reranker-0.6B \
  --served-model-name reranker-model --host 0.0.0.0 --port 18083 \
  --gpu-memory-utilization 0.20 --max-model-len 4096 \
  --trust-remote-code > logs/vllm_reranker.log 2>&1 &
```

Falling back to local gen/embed instead of the LiteLLM proxy is still
supported (`run_servers.sh gen`/`embed`) but isn't the current default path —
no specific model is pinned here since whatever you point
`VLLM_URL`/`VLLM_EMBED_URL` at just needs to speak the OpenAI-compatible
`/v1/chat/completions`+`/v1/embeddings` API; match `NEO4J_VECTOR_DIMENSIONS`
to whatever embedding model you actually run.

Notes:
- **Indexing needs gen + embed** (from the LiteLLM proxy by default), **not
  the reranker** — Prehop's own pipeline never calls a reranker (see
  "Reranking"). A local rerank server is only needed to run the
  HopRAG/MS-GraphRAG baselines.
- If falling back to local gen/embed alongside the local reranker, start
  servers sequentially and watch combined `--gpu-memory-utilization` — KV-
  cache contention can kill a second small model that comes up
  simultaneously.
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
- This Neo4j container also holds data from unrelated indexing runs under
  other corpus tags — keep corpus-tags namespaced (e.g. a dataset-name
  prefix) so reruns never collide with existing `PR_<tag>_*` labels already
  in the database.

## Indexing failure tracking

A single chunk/file/HOP-edge error does not silently degrade or take down
the whole run — per-file errors are isolated (`cli/index.py`'s
`process_file`), so other files keep indexing. Whenever `run_indexing`
finishes with any failures, the full list (not just the 10-item log preview)
is written to `data/index_failures/<strategy>_<corpus_tag>_<timestamp>.json`
— `{strategy, corpus_tag, dataset_path, total_files, succeeded, failed,
failures: [{item, stage, error}]}`, `stage` one of `read|index|task|
hop_edges`. Check this file after a partial run instead of grepping
scrollback.

## Graph statistics (paper dataset/graph tables)

`run_indexing` (`prehop` strategy only — this structure doesn't exist for
`naive`) queries the live graph right after indexing finishes and writes
`data/index_stats/<strategy>_<corpus_tag>_<timestamp>.json` — total
documents/chunks, `avg_chunks_per_doc`, `q_minus_coverage`/`q_plus_coverage`
(fraction of chunks with a non-empty Q⁻/surviving Q⁺), `total_hop_edges`,
and `avg_hop_out_degree_per_eligible_chunk`/`avg_hop_out_degree_per_chunk`
(HOP-edge density among Q⁺-surviving chunks vs. across the whole chunk
population). Queried directly from Neo4j (not counters threaded through
indexing), so it's always consistent with what actually landed in the
graph — same source of truth as the "Neo4j data layout" integrity probes
below. Best-effort write (a collection failure doesn't fail the indexing
run); feeds the paper's per-dataset graph-statistics table.

## Query-time latency breakdown (paper headline-latency claim)

`prehop`'s query path (`models/prehop/graphrag.py`'s `run_workflow`) times
its own retrieve / traversal / synthesis stages (`retrieve_ms`,
`traversal_ms` — 0 when `RAG_GRAPH_HOP_DEPTH=0` — `synthesis_ms`), attaches
them to `interaction_trace`, and `cli/benchmark.py` lifts them onto each
result row's top level. The existing generic per-run averaging
(`_recompute_aggregates`) then produces `avg_retrieve_ms`/
`avg_traversal_ms`/`avg_synthesis_ms` automatically, overall and per
category — no separate aggregation code needed. Other strategies' traces
don't carry these keys, so their summaries simply omit these fields rather
than reporting a misleading 0. The three stages sum to `avg_latency` (in ms)
to within rounding.

## Reboot / interruption recovery

1. `docker start prehop-neo4j`; verify labels survived:
   `CALL db.labels()` + per-label counts.
2. Generation and embeddings come from the LiteLLM proxy by default (see
   "Model / inference infra") — nothing to relaunch for those. Relaunch a
   local vLLM server only if falling back to local gen/embed, or if running
   a baseline that needs the local reranker.
3. Re-run any unfinished index. **prehop/naive/hoprag** are durable in Neo4j —
   if a strategy's node counts look complete, skip it. **ms_graphrag** resumes
   from its own cache (`data/ms_graphrag_output/<tag>/_cache/extract_graph/`):
   re-run the identical index command and it replays cached chunks instantly,
   then continues from the break. No `--clear-graph`.

## Neo4j data layout & integrity checks

Corpus-tag prefixes labels so datasets coexist: `PR_<tag>_*` (prehop),
`NA_<tag>_*` (naive), `HO_<tag>` (hoprag).

prehop schema (written by `models/prehop/indexing/graph_writer.py`):
- `PR_<tag>_Document {filename, title, updated_at}`
- `PR_<tag>_Chunk {id, text, chunk_summary, q_minus_text, q_plus_text,
  embedding, q_minus_embedding, q_plus_embedding}`
- Relationships: `NEXT` (chunk sequence; edges = chunks − docs),
  `HOP` (rank-based, see "Reranking" for the current scoring mechanism,
  MERGE-idempotent)
- Indexes: 3 vector (body/qminus/qplus) + 3 fulltext + range on id/filename

By-design partial coverage (NOT incomplete indexing):
- `q_minus_embedding` < chunks: empty Q⁻ is legitimate (some chunks yield no
  self-contained question). The primary `embedding` is always the body
  embedding, never substituted with Q⁻.
- `q_plus_embedding` < chunks: Q⁺ passes `_is_high_quality_q_plus` gating
  (see "Reranking"); the surviving count equals the "wrote N HOP edges over
  M Q+ chunks" log line.

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

Two knobs shape how the news/multi-hop domain is handled:

- **Company-anchoring gate** (`core/config.py` `COMPANY_ANCHORING`, default
  `DOMAIN=="financial"`, env `RAG_COMPANY_ANCHORING`). When OFF (news):
  `_extract_company_keys` returns `set()` (neutralizes the strict company
  filter in `rerank.py`/`traversal.py` + mismatch penalty + company boost), and
  `hop_edges.py` drops the same-company HOP filter — without this, cross-
  document evidence in a non-company-anchored corpus gets spuriously pruned.
- **`DEFAULT_TOP_K`** is domain-aware: **news=12, financial=8** (env
  `RAG_DEFAULT_TOP_K` overrides).

`_build_context_from_nodes` always uses the single `[[title, Page N, Chunk
N]]` format; `QUERY_REWRITE_PROMPT` always does plain paraphrase rewriting —
no domain-specific synthesis formatting or query-decomposition branches
exist in the current code.

**Re-index note:** company-anchoring touches HOP-edge construction, so a
HOP-only rebuild suffices for that knob alone (chunks/Q±/embeddings
unchanged). For a fully reproducible from-scratch reindex, scoped-delete only
`PR_<tag>_{Chunk,Document}` (NEVER `--clear-graph`) and set
`RAG_CHUNK_CACHE=off` — otherwise the on-disk chunk cache
(`data/index_cache/v0/<tag>/`, see `chunking.py`) replays prior Q±/summaries.
top_k change needs NO reindex (benchmark only).
