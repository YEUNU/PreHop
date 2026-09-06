# Changelog

This file records implementation and experiment-protocol changes.
Entries are listed in reverse chronological order. `ARCHITECTURE.md` defines
current behavior, and result values belong in `RESULTS.md`.

## 2026-09-06 — Backend-compatible ranking schema

The next cold PreHop attempt completed indexing, then the gateway rejected the
ranking request's unsupported `uniqueItems` key. Structured profile v3 removes
that wire key while retaining exact count, candidate membership and strict
local duplicate rejection. It also removes redundant length metadata already
implied by the nonblank pattern and rejects unreviewed schema keywords before
sending a request. Local validation semantics and generation budgets remain
unchanged; schema/index/query/cache identities advance.

All seven materialized variants—three index formats, rewrite, refinement and
single/multiple-candidate ranking—passed actual production-helper/gateway checks
with HTTP 200, one stopped completion and strict local validation. Both ranking
outputs were unique. These probes do not admit a complete cold target or full
benchmark. Earlier failed artifacts remain preserved.

## 2026-09-06 — Portable nonblank schema and completion diagnostics

The first released cold PreHop index stored 13 one-character questions under
the bare `\S` schema constraint. The v2 structured profile expresses the same
nonblank requirement under both search and full-match decoding semantics;
multiline sentences remain valid. Schema/index/query/cache identities change,
while generation budgets, prompts and existing artifacts remain unchanged.
Strict completion failures now distinguish choice-count and finish-reason
failures and record allowlisted metadata without response text. The separate
first-chunk diagnostic succeeded and did not reproduce the original completion
failure; truncation is not established. With the portable v2 schema, one
same-prompt request completed with six questions of 59–120 characters at the
unchanged cap, temperature and seed. This validates the format correction,
not a complete cold target or full benchmark. Full-target admission remains pending.

## 2026-09-06 — Versioned structured outputs and owned gate executors

Prehop now constrains and validates all four JSON-producing operations with
registered schemas. Query/index metadata and generation caches bind the new
schema profile; malformed outputs and incomplete rankings fail without repair.
Hippo uses its official JSON-object OpenIE setting, while Youtu adds a
construction-only strict schema through public SDK injection. Native prompts,
parsers, retrieval, and generation budgets remain unchanged.

Cold integration uses one preregistered synthetic two-document/one-query
fixture. Repository-owned executors cover runtime reattestation, actual owned
checkpoint interruption/resume, complete-corpus one-query targets, exact matrix
aggregation, and fresh full-target admission. A configuration-bound systemd supervisor now owns indexing and benchmarking, with
shared resource locking, process-start/cgroup checks, atomic disk status,
redacted logs, failed-stage stop and actual final admission validation.
Existing results, failed attempts, upstream checkouts, caches and services remain preserved. These
changes require a new static/live campaign and add no admitted full result.

## 2026-09-06 — GFM entity-linker cache localized

A GFM cold canary passed before its native entity linker wrote 26 cache files
under the repository's default `tmp` directory, invalidating the campaign's
code context. The adapter now sets the public entity-linker `root` to an
absolute directory within the current run's artifacts. Hydra propagates the
same setting to the graph constructor's entity linker. Existing cache files
remain preserved; no native cache, indexing or retrieval implementation changed.

## 2026-09-06 — Cold-canary adapter boundaries corrected

GFM now composes Hydra configuration from the same directory and basename as
the pinned workflow. Linear calls the native `qa()` once, using a typed
transport implementation of its configured `infer()` interface; native
retrieval, prompt, answer parsing, passage order and 2000-token generation
limit remain intact. Youtu's observer rejects non-string attribute items
without changing extraction values. Shared method environment preparation
preserves registry defaults and empty unknown dotenv aliases across native
imports while retaining explicit overrides for policy rejection.

The preserved cold-canary outcomes are recorded in `RESULTS.md`. These
corrections add no full-target admission and do not repair native malformed
JSON, empty evidence, or missing service prerequisites.

## 2026-09-06 — Main runtime dependency admission corrected

A read-only check found incompatible ambient packages in the original main
environment after the external runtime checks had passed. Preflight now checks
the running main interpreter's dependency compatibility and frozen project
lock synchronization for every target. Gate identity includes that interpreter
and its installed package metadata. Paper runners use the selected interpreter
directly, rejecting conflicting `PYTHON_BIN` and `UV_PROJECT_ENVIRONMENT`
settings. The original environment remains preserved; a fresh environment
can be synchronized separately from `uv.lock`.

## 2026-09-06 — Native query initialization and child environment corrected

Fresh Youtu query workers now call the public native `build_indices()` method
immediately after constructing the retriever, matching the pinned entrypoint.
Previously only the indexing worker made that call. Primary child environments
also retain empty forbidden-provider aliases and explicit typed defaults so
native dotenv imports cannot restore conflicting local settings. Regressions
use exact upstream exports for HippoRAG/GFM imports and initialize a copied
Youtu index with network connections forbidden. These checks do not establish
successful end-to-end retrieval or admission.

## 2026-09-06 — System-Python packaging compatibility corrected

The next setup attempt stopped before package installation because the staging
helper used `hashlib.file_digest`, which is unavailable in the launcher's
Python 3.10.12. The helper now hashes archive bytes in streaming blocks. Its
regression runs the actual system-Python CLI, verifies the archive digest, and
exercises setuptools output in the separate build directory. The failed
attempt and its clean source checkout were preserved.

## 2026-09-06 — Packaging moved outside pinned source checkouts

The first live setup completed package dependency checks but generated ignored
LightRAG and untracked HippoRAG build metadata inside their source checkouts.
The attempt remained unadmitted and its files were preserved. Package setup
now exports approved revision bytes into unique artifact-local build
directories, records an archive digest, and checks original sources including
ignored files before and after packaging. Worker path resolution now shares
the setup runtime-home setting, allowing a fresh attempt without moving or
cleaning the original directories. A temporary setuptools build regression
exercises real build and egg-info output outside the source checkout.

## 2026-09-06 — Independent audit corrections

Fixed local snapshot JSON parsing, exact run-ID admission and reuse ordering,
primary child transport alias rejection, native-answer adapter initialization,
and per-client generation-model override validation. Gateway identity now
requires the approved non-secret URL digest. Registry-derived shell defaults
replace repeated transport literals. HippoRAG's recorded query input reflects
its native unprefixed encoder and uses the typed request timeout. Youtu
malformed triples fail observational extraction validation without changing
native parse values; benchmark seeds no longer overwrite its native generation
seed policy.

Added current-corpus byte validation to admission and stage-specific bound
artifacts to live gates; full-target gate evidence reruns artifact and ledger
freshness checks. Existing dirty legacy source/checkpoint files were observed
and preserved. These changes do not establish clean primary installation,
passing live canaries, or admitted primary results.

## 2026-09-06 — Upstream boundary and live admission gates hardened

Removed Youtu runtime overrides of retriever chunk-ID extraction and builder
deduplication/serialization. The adapter now records staged coverage, native
extraction success, and native graph source reachability in observational
sidecars, preserves returned evidence order, and fails on malformed, empty,
sentinel, or swallowed-error results. Because the pinned agent batch entrypoint
does not return structured per-query evidence, the registry now labels Youtu's
primary target as a controlled public no-agent API variant instead of an
official-faithful agent result. MS GraphRAG no longer augments native
communities with synthetic singleton rows.

Centralized the paper transport, method environment keysets, runtime
identities, exact index-stat bytes, post-query artifact inventory, and
row-ordered query content in content-bound policy/admission records. Paper
entrypoints reject public legacy/provider aliases, and supplemental public
OpenAI Batch submission is disabled. Added ordered live-gate evidence and a
fresh-ledger requirement before a real 16-target matrix. These are engineering
changes; no live canary or primary empirical result is asserted here.

## 2026-09-06 — BrowseNet MuSiQue benchmark completed

Completed the official BrowseNet full-split evaluation (2,417 queries) on
MuSiQue under `naacl27-clean-20260905-musique-browsenet`. Official MuSiQue metrics:
Answer EM = 0.2466 (24.66%), Answer F1 = 0.3170 (31.70%), Support Precision = 0.2909,
Support Recall = 0.6797, Support F1 = 0.4016, Latency = 12.44s. The artifact is
preserved under `data/results/` as legacy/reserve baseline evidence.

## 2026-09-06 — Primary matrix and admission contract superseded

Superseded the earlier six-method, twelve-target selection with a primary
eight-method, sixteen-target matrix: Prehop, Naive RAG, MS GraphRAG, LightRAG,
HippoRAG2, GFM-RAG, LinearRAG, and Youtu-GraphRAG on MultiHop-RAG and MuSiQue.
BrowseNet, HopRAG, and PropRAG remain supported legacy/reserve adapters; their
existing artifacts are preserved but cannot be reused as primary matrix cells.

Added a typed strategy registry shared by Python and shell entrypoints, pinned
isolated runtimes for the external methods, one fail-closed LiteLLM transport
for remote generation and embeddings, and explicit pinned local components
for LinearRAG, Youtu-GraphRAG, and GFM-RAG. LinearRAG's GPL upstream
code remains outside the MIT tree, and Youtu-GraphRAG retains its upstream
academic/research-use terms.

Strengthened corpus, query, semantic-configuration, index-coverage, detail-row,
and aggregate admission checks. The maintained status vocabulary is
`planned`, `canary_passed`, `in_progress`, `completed_unadmitted`, `admitted`,
and `failed`. A canary no longer implies completion, and completion no longer
implies admission. Until all applicable checks pass, numerical values are not
copied into the primary result register or manuscript. The matrix now
continues across independent target failures and exits nonzero after reporting
the failed targets.

## 2026-09-06 — BrowseNet MultiHop-RAG benchmark completed

Completed the official BrowseNet full-split evaluation (2,556 queries) on
MultiHop-RAG under `naacl27-clean-20260905-multihoprag-browsenet`. Admitted
official metrics: Hits@4 = 0.9424, Hits@10 = 0.9676, MRR@10 = 0.7916,
MAP@10 = 0.4488, QA Acc = 0.3658, Null Refusal = 0.9900, Latency = 26.34s.
Updated `RESULTS.md` and `docs/prehop_paper.md` accordingly.

## 2026-09-05 — Previous evaluation artifacts invalidated

The previous evaluation artifacts and recorded metrics were removed after the
audit found dirty or inconsistent code provenance and adapter defects that can
change retrieval behavior. The replacement `naacl27-clean-20260905` campaign
uses clean committed code, run-scoped index namespaces, isolated baseline
outputs, and explicit one-strategy targets. Recently added PropRAG and
BrowseNet baselines run before the remaining strategies.

## 2026-09-02 — Clean 8B full-system evaluation
 
The embedding endpoint and recorded revision use `qwen3-embedding-8b` with
4,096-dimensional vectors. The generation model remains `gemma-4-31b-it`.

The evaluation runs Prehop, Naive RAG, HopRAG, MS GraphRAG, BrowseNet, and
PropRAG independently on the complete MultiHop-RAG and MuSiQue prepared splits.
Every target clears the graph, constructs a cold index, validates index
integrity, and evaluates the complete query set with concurrency 4 and the LLM
judge disabled. Official BrowseNet and PropRAG execute in isolated runtime
environments while routing semantic retrieval embeddings through the shared
LiteLLM endpoint.

Previous-model results, caches, baseline outputs, logs, and document records
were removed before the new result register was produced.

Documentation responsibilities were reduced to one source per concern.
Component-control definitions now belong to `ARCHITECTURE.md`, while artifact
admission and publication synchronization belong to `RESULTS.md`. The former
standalone ablation protocol and consistency-audit files were removed after
their unique requirements were transferred.

Submission documentation now consistently uses complete prepared splits,
query concurrency 4, and `qwen3-embedding-8b` with 4,096-dimensional vectors.
Former sample-exclusion rules were removed from the maintained workflow.
