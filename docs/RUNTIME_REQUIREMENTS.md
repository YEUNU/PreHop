# Paper Runtime Requirements

This document explains the machine-readable external-method requirements in
`configs/paper_runtime_requirements.json`. It does not replace that file or
`core/strategy_registry.py`, which owns strategy identity and primary order.

## Preserved setup attempts

The initial 2026-09-06 audit found no installed primary external source
checkouts. The first subsequent setup installed five primary research runtimes,
but packaging generated ignored build files in LightRAG's source and untracked
build files in HippoRAG's source. The attempt did not pass the live gate.
These directories, the legacy BrowseNet source-local checkpoint, and PropRAG's
pre-existing tracked modification are preserved. Setup now rejects tracked,
untracked, and ignored source changes; it does not clean or migrate them.

Local package installation exports the exact approved git revision into a new
`artifacts/builds/<revision>-<unique>/source` directory and builds there. A build
manifest records the revision and archive digest. The original source is
checked before and after packaging, including ignored files, and again before
the runtime freeze. Choose a new `RAG_OFFICIAL_BASELINE_HOME` for a fresh retry;
setup, worker source/Python resolution, snapshots, and freeze checks use the
same `<home>/<strategy>/{source,artifacts,venv}` layout. Explicit per-strategy
source/Python overrides still take precedence and must point to that same
validated runtime. Never remove or relocate an existing attempt to make room.

The approved gateway identity is stored without its address or credentials in
`configs/paper_gateway.json`. It was derived from the user's authorized local
configuration. The existing local configuration still uses legacy provider
field names; paper entrypoints require explicit canonical exports and reject
legacy aliases. A one-time authorized launch translation must export canonical values, keep
forbidden aliases empty, and skip reloading the legacy file; the repository does not support
two public transport contracts.

## Main Python environment

Paper entrypoints validate the actual running environment with `uv pip check`
and read-only `uv sync --check --frozen --no-install-project --offline` before
accepting any target, including Prehop and Naive. Gate evidence binds the
interpreter path and installed distribution metadata as well as the lockfile.
Preserve an incompatible environment and prepare a new directory instead:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/new/main-venv
uv sync --frozen --no-install-project
export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
```

Both selectors must identify the same environment when supplied together.
Paper runners execute this interpreter directly and do not synchronize it
implicitly. A missing explicit interpreter fails instead of selecting another.

## Common inference gateway

Paper mode uses one OpenAI-compatible LiteLLM base for both
`/chat/completions` and `/embeddings`. Configure the canonical transport at the
shell entrypoint with:

- `RAG_INFERENCE_BASE_URL`
- `RAG_INFERENCE_API_KEY`
- `RAG_GENERATION_MODEL=gemma-4-31b-it`
- `RAG_EMBEDDING_MODEL=qwen3-embedding-8b`

Legacy `VLLM_*`, ambient provider, and direct-vendor variables are rejected as
public inputs. A validated parent may inject compatibility names only into an
isolated upstream child process. Internal clients do not choose between
aliases. An empty base, different generation and
embedding bases, an unregistered model, or a public-vendor fallback fails
closed in paper mode. Do not print the API key or include it in artifacts.

Remote embeddings use batch size 16 and concurrency 1. Generation concurrency
is separate. Timeout, retry, batch, and concurrency are operational settings
and are recorded apart from the semantic method configuration.

Paper preflight also fixes the shared semantic service boundary: remote
embedding width `NEO4J_VECTOR_DIMENSIONS=4096`, embedding input limit
`MAX_EMBEDDING_LENGTH=32768`, `RAG_EMBEDDING_TOKEN_RESERVE=0`, generation
context `RAG_MAX_CONTEXT_LENGTH=262144`, query instruction exactly as recorded
in the canonical policy, and `NEO4J_FULLTEXT_ANALYZER=english`. MS GraphRAG's
`RAG_MS_EMBED_DIM` is the same 4096 and its report output cap is 4096. PreHop
and Naive additionally admit only the registry's legacy question schema,
enabled Q−/Q+ flags, disabled sentence channel, and reciprocal-hop
precomputation. Explicit conflicting values fail before indexing, including in
`--check` mode.

## Pinned external runtimes

Run `./scripts/setup_official_baselines.sh --primary` to create isolated primary checkouts and
environments under ignored `data/official_baselines/`. The setup uses the exact
repository revisions in `core/strategy_registry.py`; it does not copy upstream
source into this repository. Preflight rejects the wrong revision and dirty
tracked or untracked upstream files. For LightRAG, HippoRAG2, GFM-RAG,
LinearRAG, and Youtu, setup also consumes repository-owned reviewed direct constraints
whose paths and digests are bound by
`configs/paper_runtime_requirements.json`. Every setup records the complete
locally resolved environment as `runtime.freeze.txt`; preflight rejects later
drift from that local snapshot. This local freeze is not claimed to be a
portable cross-platform transitive lock for every upstream method.

Official checkouts are also immutable during execution. Adapters may prepare
run-local inputs and schemas, configure the approved LiteLLM route, write
observational sidecars, and validate native artifacts. They may not patch
method-defining retrieval, deduplication, serialization, or answer methods.
Such a change is a controlled deviation and requires a separate semantic
configuration before it can be evaluated. BrowseNet's ColBERT files therefore
live at `data/official_baselines/browsenet/artifacts/colbertv2.0`, not under
`source/`.

LinearRAG is GPL-licensed upstream and remains process-isolated from this MIT
tree. Its primary official-faithful mode uses the pinned
`sentence-transformers/all-mpnet-base-v2` snapshot and
`en_core_web_trf-3.6.1`. The snapshot is downloaded at the exact registry
revision and loaded from its local path. The controlled Qwen mode is a distinct
semantic configuration and is not a primary result mode until admitted.

Youtu-GraphRAG retains its upstream academic/research-use terms. It uses the
pinned `sentence-transformers/all-MiniLM-L6-v2` snapshot,
`en_core_web_lg-3.7.1`, and a dataset-specific schema selected from its pinned
checkout by the checked-in manifest. MultiHop-RAG maps explicitly to upstream
`hotpot` and MuSiQue maps to `musique`; their schema paths and SHA-256 values
are not caller-controlled. Both aliases use the upstream no-chunk document
path and the native effective retrieval/filter budget of 20.

The pinned agent batch entrypoint returns no structured answer/evidence and
also invokes an upstream evaluator. Without changing that upstream behavior,
it cannot satisfy content-bound per-query admission. The registry therefore
labels the primary Youtu target `controlled_adapter` with `query_mode=noagent`
and calls the pinned public `initial_question_decomposition` API. This is not
an `official_faithful` agent result. The adapter preserves returned evidence
order and native retrieval/deduplication, and rejects empty, malformed,
sentinel, exhausted-retry, and swallowed sub-question failures.

Youtu retains native graph and retrieval behavior, including its handling of
duplicate entities and triples. The runtime records complete staged
chunk/source coverage separately from native graph source reachability. A
false reachability flag is an observed method-native limitation, not permission
to patch the graph or reject an otherwise completed native index.

GFM-RAG loads `model.pth` and `config.json` from the pinned
`rmanluo/GFM-RAG-8M` snapshot and loads its ColBERT entity linker from the
pinned `colbert-ir/colbertv2.0` snapshot. Setup fetches both at the exact
revisions in the machine-readable manifest. Preflight resolves them locally
and fails closed if either approved snapshot or a required file is absent.
Their identities and observed content hashes are recorded in semantic
provenance.

GFM-RAG uses the upstream-compatible reviewed set `openai==2.29.0`,
`langchain-openai==0.3.35`, `torch==2.8.0`, and `vllm==0.10.2`. The complete
direct set resolves together under Python 3.12, setup applies it as a normal
constraint to the pinned source, and preflight runs `uv pip check` against the
installed metadata.

The isolated LinearRAG runtime uses Python 3.9; its pinned NumPy and
sentence-transformers versions are not declared compatible with Python 3.10+.

## Preflight

Run the check for every external primary method before assigning a final
campaign prefix:

```bash
uv run python scripts/check_paper_runtime.py --strategy lightrag --dataset multihoprag
uv run python scripts/check_paper_runtime.py --strategy hipporag2 --dataset multihoprag
uv run python scripts/check_paper_runtime.py --strategy gfm_rag --dataset multihoprag
uv run python scripts/check_paper_runtime.py --strategy linear_rag --dataset multihoprag
uv run python scripts/check_paper_runtime.py --strategy youtu_graphrag --dataset multihoprag
```

The check validates the pinned checkout, declared direct-constraint digest,
locally captured dependency set, local model snapshot, required spaCy model and
exact spaCy distribution version, checkpoint/configuration or schema path and
digest, and Python/runtime compatibility where applicable. It loads the
project `.env` without overriding
already-exported values, validates the same typed LiteLLM transport used by the
runner, and never prints its credential. It does not perform a full run, modify result
artifacts, or prove admission. A passing preflight is followed by a synthetic
canary; a canary is followed by a full target; a full target is reportable only
after admission. The target wrapper invokes the same preflight in both
`--check` and real-run modes, so these manual commands are diagnostics rather
than an alternative execution path.

## Live gates

Static review must be GO before live preparation. The live sequence is clean
primary setup; all 16 target preflights; one chat probe; embedding batch 16 at
concurrency 1 with count/index/4,096-dimension/finite checks; selective
oversize bisection; 16 cold two-document/one-query canaries; interruption,
resume, and stale-policy rejection; a 16-target one-query matrix; and one fresh
full target ending in content-bound admission. `scripts/paper_gate_ledger.py`
records each stage in order and binds it to current source, verifier, runtime,
target matrix, and evidence bytes. `scripts/run_paper_matrix.sh` refuses a real
matrix until every gate is fresh and unchanged. A passing gate is not a
reportable result.

## Live-gate evidence

`paper_gate_ledger.py` separates independent static-review attestation from
empirical stage evidence. Each empirical evidence object declares its schema
version and stage. Matrix stages map all 16 targets to SHA-256-bound artifact
references. Cold and one-query stages bind an invocation receipt, fresh index,
native query, and canary admission to the same target and run. A full-target
gate must reference the actual target admission ledger and pass the complete
artifact and freshness checks again. Status-only JSON and target-name lists
are insufficient. Verification rechecks evidence references on every launch.

HippoRAG's native OpenAI encoder ignores the upstream instruction argument and
normalizes newlines to spaces. Its canonical method policy therefore records
an empty query instruction and a `{query}` template; use no nonempty
`EMBEDDING_QUERY_INSTRUCTION` override for that method. This is an explicit
native input difference within the controlled remote-backbone comparison.

## Frozen stage executors

Cold protocol v2 uses the exact checked-in
`configs/cold_canary/museum_rich_entities_v2.json` fixture, SHA-256
`ce4cfea5eb2f6eb784afaa669c762ec82e5869c98144b9512ac85e3b6fcb841a`.
All 16 targets receive the same two synthetic documents and one bridge
question. Each document names 24 artifacts. Local ColBERT token counts are
design estimates, not Gemma output budgets or observed native entity catalogs;
Hippo NER remains capped at 512 tokens. Both native PLAID indexes must satisfy
their own training requirements in execution. Fixture bytes are fixed before
inference and are not tuned after failures. This protocol checks integration
only. One-query and full-target gates continue to require the exact current
real corpus and query records.

After independent static GO, run the repository-owned stage commands with the
selected clean main interpreter (`PYTHON_BIN`):

```bash
"$PYTHON_BIN" scripts/paper_stage_runner.py reattest CAMPAIGN --attempt a1
# Run the existing preflight and gateway probe stages in ledger order.
"$PYTHON_BIN" scripts/paper_cold_canary.py CAMPAIGN youtu_graphrag multihoprag --attempt a1
# Repeat the cold command for every primary strategy and both datasets.
"$PYTHON_BIN" scripts/paper_stage_runner.py cold-aggregate CAMPAIGN --attempt a1
"$PYTHON_BIN" scripts/paper_stage_runner.py recovery CAMPAIGN --attempt a1
"$PYTHON_BIN" scripts/paper_stage_runner.py one-query CAMPAIGN --strategy naive --dataset multihoprag --attempt a1
# Repeat one-query for all 16 targets before aggregation.
"$PYTHON_BIN" scripts/paper_stage_runner.py one-query-aggregate CAMPAIGN --attempt a1
"$PYTHON_BIN" scripts/paper_stage_runner.py full-target CAMPAIGN --strategy naive --dataset multihoprag --attempt a1
```

Reattestation checks existing runtimes without installation. Recovery builds a
separate complete MultiHop-RAG Naive index and selects the first two real
queries for an exploratory resume test. An explicit test-only barrier pauses
its newly owned child after the first complete checkpoint. The parent checks
PID, process start and a nonce before sending SIGTERM to that child, resumes
the actual benchmark, and verifies retained rows and traces. A preserved
checkpoint copy must reject a changed semantic setting. This does not stop
existing workers or services and produces no full benchmark admission.

Each one-query target builds the complete real corpus in a new namespace;
the full-target gate then builds another fresh index and runs every real
query. Aggregation requires all 16 content-bound target artifacts. Existing
attempt paths are rejected and preserved. Explicit Python settings must agree
with the selected main interpreter, including shell subprocesses. Static review binds the effective configuration contract. Project Git changes
remain provenance; preserved evidence is revalidated with the current validators.
Changed models, prompts, schemas, semantic versions, data or runtime content
require compatible new evidence. Existing result bytes are never rewritten.

## Durable campaign ownership

`scripts/paper_campaign.py` runs every indexing and benchmark stage under one
systemd user supervisor. Its launch command requires an available user manager
and a verified `Linger=yes` for the current user. The release executor may
enable only that user's linger under the authorized background-execution
scope, then verify it; the launcher never changes host persistence settings.
A transient unit survives the initiating task and logout under that verified
configuration. Reboot recovery is not claimed.

Prepare a dedicated detached worktree at the final pushed commit, copy the
real corpus/query files with byte verification, and generate independent
attestation and a new ledger in that worktree. Reuse the approved main and
external environments by absolute paths. A new ignored `.env` placeholder may
satisfy shell entrypoints while canonical connection settings are supplied by
a private mode-0600 service environment file. Do not copy the original `.env`
or place credential values in command arguments.
The service environment retains empty legacy-provider sentinels and selects
LiteLLM's supported `LITELLM_MODE=PRODUCTION` before native imports. Explicit
conflicting provider values or `DEV` mode fail validation rather than being
silently overwritten.

After recording a static GO whose `reviewed_configuration_sha256` equals the
central configuration projection (excluding dependencies not yet installed):

```bash
"$PYTHON_BIN" scripts/paper_campaign.py plan CAMPAIGN --attempt a1
"$PYTHON_BIN" scripts/paper_campaign.py launch data/results/CAMPAIGN/supervisor/plan.json
"$PYTHON_BIN" scripts/paper_campaign.py status data/results/CAMPAIGN/supervisor/plan.json
```

The immutable plan includes all ordered gates and the final full matrix. A
shared OS-user resource lock prevents overlapping campaigns across worktrees;
the owned child inherits that lock. Status records contain supervisor/child
PID, process-start and boot identity, stage, actual exit status, log paths,
checkpoint counts and bound evidence. Separate stdout/stderr files redact
canonical keys and connection URLs. Atomic status and append-only events stay
available after the initiating task ends. Launch verifies that the systemd
MainPID matches the running supervisor; a submitted unit name alone is not
success.

`launch ... --resume` refuses a live prior owner or child. It revalidates the
frozen context and completed evidence before proceeding. Existing incomplete
fresh-index/cold attempts remain preserved and are not silently replaced;
strict matrix checkpoint recovery is handled by the existing target runner.
For a failed cold or one-query target, create an immutable successor segment:

```bash
"$PYTHON_BIN" scripts/paper_campaign.py retry-plan data/results/CAMPAIGN/supervisor/plan.json --retry-step cold/multihoprag/ms_graphrag --attempt a2 --segment ms-retry
"$PYTHON_BIN" scripts/paper_campaign.py launch data/results/CAMPAIGN/supervisor/segments/ms-retry/plan.json
```

The successor binds the original plan and terminal status bytes, changes only
the failed target's attempt, and revalidates every inherited completed step.
The producer, target validator and matrix assembly use the same attempt map.
Original plans, status files and failed outputs remain in place. A changed
configuration or invalid prior evidence blocks continuation.
Automatic service restart is disabled. Any failed prerequisite stops dependent
stages. The final summary requires sixteen actual full-target admissions,
reports missing/invalid cells and exits nonzero otherwise. Effective configuration
or runtime drift stops execution; Git and documentation changes remain provenance.
