# Prehop repository guide

The current system evaluates Prehop, Naive RAG, official HopRAG, and official
MS GraphRAG on MultiHop-RAG and MuSiQue. The detailed module and
branch map is in `docs/ARCHITECTURE.md`.

## Documentation responsibilities

Keep repository facts in one authoritative place and link to them elsewhere:

- `CLAUDE.md` is the operational policy for maintainers and coding agents:
  supported state, invariants, experiment hygiene, and documentation rules.
- `README.md` is the user-facing overview: the method at a glance, repository
  layout, setup, commands, and links to deeper contracts. It does not carry
  development-result tables.
- `docs/ARCHITECTURE.md` is the normative implementation map: module ownership,
  indexing/query data flow, algorithmic behavior, and evaluation contracts.
- `docs/CHANGELOG.md` is the chronological engineering record. It records what
  changed and may summarize explicitly labelled exploratory validation, but it
  is not the current architecture specification or a paper-results source.
- `docs/prehop_paper.md` is the local, gitignored AI-research manuscript and
  confirmatory evaluation specification. It contains the fixed method, claim
  scope, final reportable results, their analysis, and limitations.
  Development history, rejected variants, intermediate checkpoints, and
  negative exploratory results belong in `docs/CHANGELOG.md`, not in the
  manuscript.

When behavior changes, update the implementation contract in
`docs/ARCHITECTURE.md`, the concise user-facing description in `README.md` if
externally relevant, and the chronological entry in `docs/CHANGELOG.md`.
Update `docs/prehop_paper.md` only for method, protocol, analysis, or claim
changes. Update this file when the supported state, workflow rules, or these
document responsibilities change. Do not copy full sections between files.

### Documentation style

- Preserve each document's audience and tone: `README.md` is concise and
  user-facing; `CLAUDE.md` is an operational policy; `ARCHITECTURE.md` is a
  neutral implementation specification; `CHANGELOG.md` is a factual history;
  and `prehop_paper.md` uses cautious, evidence-bound research prose.
- Do not invent or capitalize labels for ordinary concepts merely to organize
  the writing. Avoid AI-like management phrases, repeated slogans, ornamental
  names, and unexplained shorthand when a direct description is clearer.
- Use proper nouns only when they identify an actual dataset, model, system,
  external project, published method, code symbol, configuration value, or
  established research term required for precision or reproducibility.
- Define necessary project shorthand once and keep it scoped to the document
  that needs it. Historical experiment labels such as P1/P2 belong in
  `CHANGELOG.md`; current documentation and paper prose describe the actual
  configuration instead.
- Prefer concrete statements about code, settings, artifacts, and measured
  results. Do not turn hypotheses, development-sample outcomes, or structural
  checks into named claims or qualitative judgments of quality.

## Supported state

- Python 3.12, dependencies managed by `uv` from `pyproject.toml`.
- Neo4j is the only local service. Generation and embeddings are mandatory
  external OpenAI-compatible endpoints configured by `VLLM_URL` and
  `VLLM_EMBED_URL`; there is no localhost/model-launch fallback.
- The embedding dimension in `NEO4J_VECTOR_DIMENSIONS` must equal the endpoint's
  actual vector length. Startup probes validate the configured model ids and
  dimensions.
- `EMBEDDING_MAX_NUM_SEQS=512` records the embedding endpoint's batch capacity.
  The client uses one batch of up to 512 texts, independently from the
  generation endpoint's `VLLM_MAX_NUM_SEQS` limit.
- Paper runs execute one dataset/strategy target at a time with an explicit
  run ID. `VLLM_MAX_NUM_SEQS` records endpoint capacity; each adapter's worker
  controls must remain within that capacity.
- The configuration selected for confirmatory evaluation uses legacy Q−/Q+,
  owner activation, materialized reciprocal provenance, depth-one full
  NEXT/HOP traversal, global selection, and role-aligned rewriting for
  questions of at most 32 words. Exact activation, rewrite-all, no-rewrite,
  and online reciprocal filtering are explicit ablations. The complete
  algorithm belongs in `ARCHITECTURE.md`.
- Query-time retrieval makes at most one bounded role-rewrite call and uses no
  learned reranker, runtime HOP construction, iterative generation, or
  heuristic domain gate. A single final synthesis call is made only when
  context is non-empty.
- Query-time filtering must not mutate or replace the offline HOP graph. Fresh
  indexes must materialize reciprocal provenance before the offline filter is
  enabled.
- This configuration was selected on a development sample and cannot support
  an effectiveness claim until the pre-declared confirmatory evaluation. Full
  benchmark tables remain useful for compatibility, but confirmatory paired
  claims exclude the fixed MultiHop-RAG sample-200 and MuSiQue sample-201 IDs.
  No method or parameter may change after that disjoint remainder is inspected.
- Official baselines keep their upstream behavior. In particular, official
  HopRAG `bfs_node` uses its published LLM node judgement during retrieval;
  this is documented baseline behavior and is routed to the external endpoint.
- Benchmark LLM judging is disabled by default. When explicitly enabled for
  exploratory error analysis, it uses OpenAI Batch by default and never
  silently falls back to synchronous paid calls. Without qualified-human
  validation, judge output is excluded from paper rankings and quantitative
  result tables.
- Deterministic dataset metrics remain authoritative. Negative sentinels and
  runtime-error rows are excluded from aggregates with eligible counts
  recorded; samples are exploratory rather than full benchmark runs.

## Data and tags

| Tag | Corpus directory | Full query file |
|---|---|---|
| `multihoprag` | `data/multihoprag_corpus` | `data/multihoprag_queries.json` |
| `musique` | `data/musique_corpus` | `data/musique_queries.json` |

Corpus files use `Title: ...`, optional `--- Page N ---` markers, then raw
text. Prehop and Naive share the parser and page-scoped fixed sentence-window
chunker. Raw pipe text remains raw.

The committed sample files are immutable development query-ID sets. After a
full dataset preparation changes evaluation annotations, run
`scripts/datasets/refresh_sample_records.py` to replace records by ID without
resampling. Do not run `make_sample.py` over an established development path.

## Common commands

```bash
# Validate/start Neo4j and validate both external endpoints
./run_servers.sh all

# One target
./run_index.sh --model prehop --dataset data/multihoprag_corpus --corpus-tag multihoprag

# One benchmark
./run_benchmark.sh --model prehop \
  --queries data/multihoprag_sample200_queries.json \
  --corpus-tag multihoprag

# Resume a checkpointed deterministic benchmark under its original identity
RAG_RUN_ID=<original-run-id> RAG_BENCHMARK_TIMESTAMP=<original-run-id> \
RAG_BENCHMARK_RESUME=true ./run_benchmark.sh \
  --model <strategy> --queries <queries.json> --corpus-tag <tag>

# Remove all graph data and application schema
.venv/bin/python main.py --mode clear_graph

# Rebuild only Prehop HOP/provenance edges after changing HOP construction
.venv/bin/python main.py --mode hop_rebuild --strategy prehop --corpus-tag multihoprag

# Run each dataset and strategy independently. Use a new run id for each cold run.
./scripts/run_paper_target.sh <multihoprag|musique> \
  <prehop|naive|hoprag|ms_graphrag> <run-id>

# Resume an already submitted OpenAI Batch judge after interruption
.venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<run-id>

# Optional supplemental LLM-judge analysis (off by default)
RAG_JUDGE_ENABLED=true EVAL_MODEL=<independent-judge-model> \
  .venv/bin/python main.py benchmark ...

# Non-paper debugging only; recorded as judge_independent=false
RAG_JUDGE_ALLOW_SELF=true RAG_JUDGE_ENABLED=true EVAL_MODEL=<generation-model> \
  .venv/bin/python main.py benchmark ...

# Verification
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

`run_servers.sh` never starts a model. If either inference endpoint or served
model id is missing/unreachable, indexing and benchmarking fail before work.

## Independent indexing runs

`run_index.sh`, `run_multihoprag.sh`, and `run_dataset.sh` accept exactly one
strategy per invocation. Dataset/strategy targets are never scheduled as an
implicit matrix. This makes the run ID, log, resource use, failure state, and
graph mutation attributable to one target.

Naive batches 32 source documents per embedding/write transaction. The paper
environment allows 64 active Prehop files; each file generates one chunk at a
time, and clients on the same endpoint/event loop share the 120-request
generation limit. Official HopRAG and MS GraphRAG retain their
adapter-specific worker limits.

A measured cold run must:

1. stop any prior indexing process;
2. clear Neo4j nodes, constraints, and non-lookup indexes;
3. remove prior `data/index_cache`, HopRAG outputs/caches, MS GraphRAG outputs,
   debug, failure, and stats artifacts for the selected targets;
4. set `RAG_CHUNK_CACHE=off` and disable baseline cache reuse;
5. use a new `RAG_RUN_ID`;
6. run endpoint/model/dimension preflight before launching the target.

The supported paper wrapper satisfies these rules without deleting another
run's baseline cache: it uses run-specific HopRAG and MS GraphRAG output roots
and disables the shared Prehop chunk and embedding caches. It also requires
explicit generation and embedding model identifiers in `.env`.

Each target writes an isolated stdout/stderr log. A target with any document,
workflow, graph-finalization, or integrity failure is failed, never silently
classified as complete.

Indexing is never combined across interrupted runs. Deterministic benchmark
resume is allowed only through the runner's strict identity gate; do not merge
or hand-edit result rows. Supplemental-judge runs use their separate batch
reconciliation path. The default ten-query checkpoint interval reduces report
rewrites and is recorded in each new benchmark artifact.

For Prehop, completion also requires the live index-quality gate in
`cli/index.py`. The gate checks representation ownership and embeddings,
question sanitation and role separation, exact NEXT topology, HOP direction,
Q+→Q−→owner provenance, bounded out-degree, and online
indexes. Coverage and density are diagnostics; benchmark retrieval metrics are
the evidence of effectiveness.

## Prehop inspection during indexing

For Prehop targets, inspect real intermediate files and the live graph rather
than relying only on progress logs:

- sample `data/debug/<run-id>/prehop/<corpus>/<source>/final_chunks.json` when
  `--save-intermediate` is enabled;
- verify raw chunk text, title/page/sent_id ordering, answerable Q-, outward Q+,
  generated questions and absence of fabricated or converted text;
- query total Documents/Chunks and Q-/Q+ coverage;
- after the final pass, inspect HOP question/owner provenance, per-source
  out-degree, cross-source property, and representative Q+→Q−→owner
  source/target text;
- compare source file count to indexed Document count and fail on any mismatch.

Debug output is namespaced by run, strategy, corpus, and source. Index logs and
paper measurement artifacts use separate directories, so simultaneous targets
cannot overwrite one another.

## Comparison policy

Prehop and Naive use the same six-sentence chunks, top-k 12, and final
synthesis prompt. Call this the controlled Naive RAG baseline, not the
canonical or standard Naive configuration: Naive RAG has no single required
chunk unit. The controlled comparison changes the retrieval architecture while
holding the evidence unit and budget fixed. A one-source-one-vector run is a
separately labelled chunking sensitivity analysis and cannot replace the
controlled baseline or a Prehop component ablation.
HopRAG retains the upstream official end-to-end top-k 20; MS GraphRAG retains
its official context budget. These unequal official settings must be stated in
the paper. A controlled equal-budget retrieval experiment is reported
separately from official-baseline results.

## Paper specification rules

`docs/prehop_paper.md` is a local, gitignored confirmatory evaluation
specification. It records the fixed method, dataset-specific metrics, and
reporting decisions; development numbers are not paper results. Only the fixed
methodology and final results may appear in its main text. Do
not narrate the development sequence or retain rejected/intermediate
experiments merely to justify the final design; preserve those records in
`docs/CHANGELOG.md`.
When editing it:

- Separate confirmed implementation facts, measured results, and hypotheses.
  Never invent a result, citation, author detail, dataset count, or statistical
  conclusion. A value without a verified artifact is omitted from the paper.
- Preserve established AI-research terminology when it improves precision,
  including RAG, GraphRAG, LLM, approximate-nearest-neighbor search, RRF, and
  LLM-as-a-judge. Define an acronym or specialized term at first use, then
  avoid unnecessary restatement; simplify only implementation-specific names
  that do not help a reader reproduce or understand the method.
- Record the exact run ID, git revision, environment/model settings, dataset
  split, seed, judge status, and artifact path for every reported number. A
  result may be reported in the paper only when the target completed without
  integrity or measurement failures. Optional judge output must be complete only when it is
  reported as a separately labelled supplemental analysis.
- Keep official-baseline behavior and Prehop ablations clearly distinct. Do
  not silently equalize top-k/context budgets or alter upstream HopRAG/MS
  GraphRAG behavior; label any controlled comparison separately.
- Report indexing and query-time costs separately. State that Prehop uses
  offline Q-/Q+/HOP construction, has no query-time retrieval LLM call, and
  makes one final synthesis call only when context is non-empty.
- For every table or figure, retain its metric definition, denominator,
  aggregation rule, uncertainty estimate, and source artifact. Do not publish
  self-judged or partial Batch-judge numbers as final results.
- Generate selection-free paired intervals from full artifacts with
  `scripts/paired_bootstrap.py --exclude-queries <fixed-sample.json>` and keep
  the recorded excluded-ID digest. A complete-split aggregate and a disjoint
  confirmatory interval answer different reporting needs and must be labelled.
- Keep the local specification ignored. A submission copy is a deliberate
  tracked export only after its venue/version is fixed and secrets, local
  endpoints, generated logs, and private submission notes are removed.

## Generated files and repository hygiene

Generated logs, caches, debug output, graphs, results, and index outputs
artifacts are not source files and must remain ignored. Do not commit virtual
environments, `__pycache__`, model weights, server logs, or partial indexes.
Root-level PDF, presentation, archive, and CSV handoff exports remain local;
add a release copy deliberately only after its contents and version are fixed.
Local PPT generation code, source files, outputs, and `_workspace/` drafts also
remain ignored because they are presentation-production material, not project
source.
Temporary scripts under `scripts/` use the `_tmp.py` suffix and remain ignored.
Obsolete scripts should be removed instead of kept as compatibility wrappers.
