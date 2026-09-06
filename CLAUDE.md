# Prehop repository guide

The primary paper matrix evaluates Prehop, Naive RAG, MS GraphRAG, LightRAG,
HippoRAG2, GFM-RAG, LinearRAG, and Youtu-GraphRAG on MultiHop-RAG and MuSiQue.
BrowseNet, HopRAG, and PropRAG remain supported legacy/reserve adapters but are
not primary targets. The detailed module and branch map is in
`docs/ARCHITECTURE.md`.

## Documentation responsibilities

Keep repository facts in one authoritative place and link to them elsewhere:

- `CLAUDE.md` is the operational policy for maintainers and coding agents:
  supported state, invariants, experiment hygiene, and documentation rules.
- `README.md` is the user-facing overview: the method at a glance, repository
  layout, setup, commands, and links to deeper contracts. It does not carry
  development-result tables.
- `docs/ARCHITECTURE.md` is the normative implementation map: module ownership,
  indexing/query data flow, algorithmic behavior, evaluation contracts, and
  component-control definitions.
- `docs/RESULTS.md` is the canonical final number-to-artifact register and keeps
  the two datasets and their denominators separate. It also defines the checks
  required before values are copied into the manuscript or presentation.
- `docs/RUNTIME_REQUIREMENTS.md` explains pinned external checkouts, local
  ancillary models, licenses, and the machine-readable preflight contract.
- `docs/CHANGELOG.md` is the chronological engineering record. It records what
  changed and may summarize explicitly labelled exploratory validation, but it
  is not the current architecture specification or a paper-results source.
- `docs/prehop_paper.md` is the local, gitignored AI-research manuscript and
  confirmatory evaluation specification. It contains the fixed method, claim
  scope, final reportable results, their analysis, and limitations.
  Development history, rejected variants, intermediate checkpoints, and
  negative exploratory results belong in `docs/CHANGELOG.md`, not in the
  manuscript.
- `SUBMISSION_TARGET.md` is the local, gitignored venue, deadline, and run
  logistics note. It is not a method or result source.
- `third_party/HopRAG/README.md` is vendored upstream documentation. Preserve
  it with the upstream code; it is not a Prehop documentation source.

When behavior changes, update the implementation contract in
`docs/ARCHITECTURE.md`, the concise user-facing description in `README.md` if
externally relevant, and the chronological entry in `docs/CHANGELOG.md`.
Update `docs/prehop_paper.md` only for method, protocol, analysis, or claim
changes. Update `docs/RESULTS.md` when a final artifact or accepted numerical
claim changes. Component-boundary changes belong in `docs/ARCHITECTURE.md`.
Update this file when the supported state, workflow rules, or these document
responsibilities change. Do not copy full sections between files.

Every result and component control must use the current recorded model
configuration. Do not carry results, artifact paths, or derived claims across
a model or vector-dimension change.

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

Paper gateway approval is the non-secret normalized URL SHA-256 in
`configs/paper_gateway.json`; changing the runtime URL alone cannot approve a
new endpoint. Keep credentials outside this file. Existing legacy checkouts
may be dirty and must never be described as clean without per-checkout
verification. Setup refuses tracked, untracked, and ignored changes before checkout and
does not migrate old checkpoints or packaging output. Build local upstream
packages only from exact-revision exports under strategy-local artifacts;
never run package installation against the immutable source checkout. Select
a fresh runtime home when an existing attempt is dirty.

- Python 3.12, dependencies managed by `uv` from `pyproject.toml`.
- `core/strategy_registry.py` is the single source of truth for primary order,
  legacy support, external repository/revision, worker, output root, license
  note, transport profile, and paper embedding identity. Python and shell
  entrypoints must consume the registry instead of duplicating lists.
- Neo4j is the only local service started by this repository. Remote
  generation and remote embeddings use one mandatory, fail-closed
  OpenAI-compatible LiteLLM gateway. Shell entrypoints accept only
  `RAG_INFERENCE_BASE_URL`, `RAG_INFERENCE_API_KEY`,
  `RAG_GENERATION_MODEL`, and `RAG_EMBEDDING_MODEL`; legacy `VLLM_*`, ambient
  provider, and direct-vendor variables fail closed. Compatibility names may
  be injected only into isolated child processes from the validated typed
  transport. Separate generation/embedding bases, public
  vendor fallbacks, and localhost/model-launch fallbacks are prohibited in
  paper mode.
- The embedding dimension in `NEO4J_VECTOR_DIMENSIONS` must equal the endpoint's
  actual vector length. Startup probes validate the configured model ids and
  dimensions.
- The current cold-run configuration uses `gemma-4-31b-it` for generation and
  `qwen3-embedding-0.6b` for 1,024-dimensional embeddings. The declared model
  revisions must match those served identities. Changing a model, revision,
  or vector dimension requires a new run ID and a cold index; never relabel an
  existing result with the new configuration.
- Paper remote embeddings use batches of at most 16 with effective concurrency
  1. Generation concurrency remains independently bounded. Strategy-specific
  operational overrides must be recorded separately from semantic settings.
- Paper runs execute one dataset/strategy target at a time with an explicit
  run ID. `RAG_GENERATION_MAX_NUM_SEQS` records endpoint capacity; each adapter's worker
  controls must remain within that capacity.
- The configuration selected for confirmatory evaluation uses legacy Q−/Q+,
  matched-paragraph connection activation, materialized reciprocal provenance,
  depth-one full NEXT/HOP traversal, question-role candidate selection, and
  role-aligned rewriting for questions of at most 32 words. Exact activation,
  rewrite-all, no-rewrite, integrated-rank selection, and online reciprocal
  filtering are explicit controls. The complete algorithm belongs in
  `ARCHITECTURE.md`.
- Query-time retrieval uses a bounded role rewrite, stored HOP links, one
  complete-list candidate-selection call to the configured generation model,
  and one final synthesis call when context is available.
- Query-time filtering must not mutate or replace the offline HOP graph. Fresh
  indexes must materialize reciprocal provenance before the offline filter is
  enabled.
- Submission effectiveness and component analyses use every query in the
  complete prepared split. Do not exclude former development-query IDs or
  substitute sample-run aggregates for the complete result.
- The primary external methods retain their method-defining upstream behavior.
  MS GraphRAG, LightRAG, and HippoRAG2 use the controlled remote backbone;
  GFM-RAG retains its validated checkpoint-defined local components;
  LinearRAG uses pinned local MPNet in the primary official-faithful mode; and
  Youtu-GraphRAG uses pinned local MiniLM and NER components. The pinned Youtu
  agent entrypoint does not return structured answer/evidence, so the primary
  registry declares a controlled no-agent adapter that calls its public native
  no-agent query API. It is not `official_faithful`. Controlled variants are
  separate semantic configurations, never silent fallbacks.
- Treat every pinned upstream checkout as immutable. Thin adapters may stage
  normalized input, configure the declared LiteLLM route, prepare run-local
  schemas, collect observational sidecars, and post-validate native outputs.
  They must not monkeypatch upstream retrieval, graph deduplication,
  serialization, or answer-path methods. A method-level change is a controlled
  deviation with its own semantic identity and is not official-faithful.
- BrowseNet's legacy ColBERT checkpoint belongs under the external runtime's
  `artifacts/` directory, never inside its source checkout. For Youtu, keep
  staged chunk coverage separate from native graph source reachability; record
  duplicate-source loss as a native limitation instead of modifying retrieval.
- BrowseNet, HopRAG, and PropRAG retain their official behavior as legacy
  adapters. Their earlier artifacts cannot populate a primary result cell.
- Benchmark LLM judging is disabled for paper targets. Paper mode prohibits the
  public OpenAI Batch path and any direct vendor fallback. A supplemental judge
  must use the same declared LiteLLM transport or fail closed, and remains
  excluded from primary rankings without qualified-human validation.
- Deterministic dataset metrics remain authoritative. Negative sentinels and
  runtime-error rows are excluded from aggregates with eligible counts
  recorded. Only admitted complete prepared-split runs enter submission
  results.
- Use exactly `planned`, `canary_passed`, `in_progress`,
  `completed_unadmitted`, `admitted`, and `failed` in the experiment ledger. A
  synthetic canary is not completion; a complete artifact is not publication
  evidence until admission succeeds.

## Data and tags

| Tag | Corpus directory | Full query file |
|---|---|---|
| `multihoprag` | `data/multihoprag_corpus` | `data/multihoprag_queries.json` |
| `musique` | `data/musique_corpus` | `data/musique_queries.json` |

Corpus files use `Title: ...`, optional `--- Page N ---` markers, then raw
text. Prehop and Naive share the parser and page-scoped fixed sentence-window
chunker. Raw pipe text remains raw.

## Common commands

```bash
# Validate/start Neo4j and validate the single external gateway
./run_servers.sh all

# One target
./run_index.sh --model prehop --dataset data/multihoprag_corpus --corpus-tag multihoprag

# One benchmark
./run_benchmark.sh --model prehop \
  --queries data/multihoprag_queries.json \
  --corpus-tag multihoprag

# Resume a checkpointed deterministic benchmark under its original identity
RAG_RUN_ID=<original-run-id> RAG_BENCHMARK_TIMESTAMP=<original-run-id> \
RAG_BENCHMARK_RESUME=true ./run_benchmark.sh \
  --model <strategy> --queries <queries.json> --corpus-tag <tag>

# Destructive maintenance only; never use in a paper target or concurrent run
.venv/bin/python main.py --mode clear_graph

# Rebuild only Prehop HOP/provenance edges after changing HOP construction
.venv/bin/python main.py --mode hop_rebuild --strategy prehop --corpus-tag multihoprag

# Run each dataset and strategy independently. Use a new run id for each cold run.
./scripts/run_paper_target.sh <multihoprag|musique> \
  <prehop|naive|ms_graphrag|lightrag|hipporag2|gfm_rag|linear_rag|youtu_graphrag> \
  <run-id>

# Preflight a pinned external runtime without exposing credentials
uv run python scripts/check_paper_runtime.py --strategy <strategy> \
  --dataset <multihoprag|musique>

# Optional supplemental LLM-judge analysis (off by default)
RAG_PAPER_MODE=true RAG_JUDGE_BATCH=false \
RAG_JUDGE_ENABLED=true EVAL_MODEL=<independent-judge-model> \
  .venv/bin/python main.py benchmark ...

# Non-paper debugging only; recorded as judge_independent=false
RAG_PAPER_MODE=true RAG_JUDGE_BATCH=false RAG_JUDGE_ALLOW_SELF=true \
RAG_JUDGE_ENABLED=true EVAL_MODEL=<generation-model> \
  .venv/bin/python main.py benchmark ...

# Verification
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

`run_servers.sh` never starts a model. If the inference gateway or either
served model ID is missing or unreachable, indexing and benchmarking fail
before work.

## Independent indexing runs

`run_index.sh`, `run_multihoprag.sh`, and `run_dataset.sh` accept exactly one
strategy per invocation. Dataset/strategy targets are never scheduled as an
implicit matrix. This makes the run ID, log, resource use, failure state, and
graph mutation attributable to one target.

Naive batches source documents for each embedding/write transaction. Paper
remote embeddings are capped at batch 16 and concurrency 1; generation uses a
separate limit. External methods run at exact upstream revisions in isolated
environments created by `scripts/setup_official_baselines.sh`. Their source and
model artifacts remain under ignored `data/official_baselines/`; the main
process exchanges structured JSON records with persistent strategy-specific
workers. LinearRAG's GPL source remains external to this MIT tree, and
Youtu-GraphRAG retains its upstream academic/research-use terms.

A measured cold run must:

1. check for conflicting indexing processes and postpone a new run when they conflict;
2. allocate a run-specific namespace; never globally clear a shared graph;
3. allocate fresh run-scoped output and stats paths; never delete or rewrite
   an existing run's result, log, database, index, or failure artifact;
4. set `RAG_CHUNK_CACHE=off` and disable baseline cache reuse;
5. use a new `RAG_RUN_ID`;
6. run endpoint/model/dimension preflight before launching the target.

The paper wrapper satisfies these rules without deleting another run's cache:
it uses run-specific graph namespaces and external output roots and disables
shared Prehop chunk and embedding caches. It also requires explicit generation
and embedding model identifiers in `.env`. A dirty tracked worktree produces a
warning and is recorded in provenance; it is not silently accepted as clean.

Each target writes an isolated stdout/stderr log. A target with any document,
workflow, graph-finalization, or integrity failure is failed, never silently
classified as complete.

Indexing is never combined across incompatible runs. A strictly complete index
may resume at benchmarking, and deterministic benchmark checkpoints may resume
only through the runner's identity gate. Do not merge or hand-edit result rows.
The default ten-query checkpoint interval reduces report rewrites and is
recorded in each new benchmark artifact.

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
Each strategy must preserve its declared answer boundary and native retrieval
semantics. MS GraphRAG retains official LocalSearch context construction;
LightRAG retains dual-level retrieval; HippoRAG2 retains native `rag_qa`;
GFM-RAG retains its checkpoint-defined GNN; LinearRAG retains its relation-free
  Tri-Graph and pinned MPNet mode. Youtu-GraphRAG is the declared controlled
  no-agent variant: it calls the pinned public `initial_question_decomposition`
  query API and preserves its ordered evidence, native retry behavior,
  retrieval, and deduplication with `top_k=top_k_filter=20`; it does not claim
  the non-returning agent batch orchestration. Its explicit dataset mapping is
MultiHop-RAG→`hotpot` and MuSiQue→`musique`; both retain the upstream no-chunk
document path. State unequal
official settings in the paper. Report any
controlled-backbone variant separately from the official-faithful mode.

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
- Record the exact run ID, git revision and dirty-worktree state,
  semantic-configuration ID and hash, environment/model settings, dataset
  split, seed, judge status, and artifact path for every reported number. A
  result may be reported in the paper only when the target completed without
  integrity or measurement failures. Optional judge output must be complete only when it is
  reported as a separately labelled supplemental analysis.
- Keep official method behavior and Prehop ablations clearly distinct. Do not
  silently equalize method-defining components or context budgets; label any
  controlled-backbone comparison separately.
- Report indexing and query-time costs separately. State that Prehop constructs
  Q-/Q+/HOP links offline, uses the configured generation model for bounded
  query refinement and candidate selection, and makes one final synthesis call
  only when context is non-empty.
- For every table or figure, retain its metric definition, denominator,
  aggregation rule, uncertainty estimate, and source artifact. Do not publish
  self-judged, partial, completed-but-unadmitted, or legacy-reserve numbers as
  primary results.
- Generate paired intervals from full artifacts with
  `scripts/paired_bootstrap.py`. Retain the complete query-ID digest and do not
  exclude former development-query IDs from submission analyses.
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

## Structured profiles and durable execution

Prehop JSON operations use the registered strict schemas in
`core/structured_outputs.py`; do not restore fenced-JSON repair or ranking
completion. Hippo and Youtu use separately declared controlled extraction
formats through public configuration/client interfaces. Schema/profile changes
must update semantic and generation-cache identity, observed native settings,
and admission tests. Native retrieval and parser source remain immutable.

Finalize all eight owned documents and tests before logical release commits.
The manuscript and submission note remain local-only; do not force-add them.
Generate the final independent attestation in the actual execution worktree
with the reviewed effective configuration. Git revision, dirty state and source
hashes are provenance only; unrelated commits, comments and documentation do
not invalidate compatible evidence. Maintain per-method contract versions in
`core/paper_compatibility.py` for semantic changes not expressed by settings.
Actual prompt/schema content, models, data and native/runtime integrity remain
compatibility inputs; current validators must recheck preserved artifacts.
Use byte-verified real corpus/query copies and fresh outputs in that worktree;
reuse approved external runtimes by explicit absolute paths. The owned
campaign supervisor must hold its resource lock, record child PID/start and
exit status, preserve failed attempts, and validate admissions after process
success. The default background launcher is actual `nohup` plus `setsid`, with
verified ignored SIGHUP, PID/start/boot/session ownership and a Linux subreaper.
Systemd and own-user linger are required only for the optional systemd backend.
The supervisor may send TERM to its individually verified native descendants on
owned failure or termination; it never escalates to KILL. Preserve other runs,
services and databases. Surviving owned descendants block subsequent campaigns.
A separate nohup monitor appends read-only observations every three hours and
writes a terminal receipt after cleanup or owner exit. It does not send chat
messages. The explicit recovery-test child retains its planned interruption.
