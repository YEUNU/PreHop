# Paper Runtime Requirements

This document explains the machine-readable external-method requirements in
`configs/paper_runtime_requirements.json`. It does not replace that file or
`core/strategy_registry.py`, which owns strategy identity and primary order.

## Source and setup isolation

Setup treats pinned upstream checkouts as immutable, including tracked,
untracked and ignored files. It exports the approved revision into a new
`artifacts/builds/<revision>-<unique>/source` directory before package installation
and records the revision and archive digest. Source checks run before and after
packaging and before the runtime freeze.

`RAG_OFFICIAL_BASELINE_HOME` selects the shared
`<home>/<strategy>/{source,artifacts,venv}` layout. Per-strategy source and Python
overrides must resolve to the same validated runtime. If setup fails or the
checkout is dirty, select a fresh runtime home. Setup does not clean or migrate
existing attempts. Explicit cleanup can retire superseded generated artifacts
without changing original source or an active runtime.

Gateway approval uses the non-secret digest in `configs/paper_gateway.json`.
Configure the canonical variables below in `.env`; historical provider aliases
are not a second supported public configuration interface.

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
- `RAG_EMBEDDING_MODEL=qwen3-embedding-4b`

Legacy `VLLM_*`, ambient provider, and direct-vendor variables are rejected as
public inputs. A validated parent may inject compatibility names only into an
isolated upstream child process. Internal clients do not choose between
aliases. An empty base, different generation and
embedding bases, an unregistered model, or a public-vendor fallback fails
closed in paper mode. Do not print the API key or include it in artifacts.

### Current serving contract

The current embedding contract uses `Qwen/Qwen3-Embedding-4B` at 2,560
dimensions. `configs/serving_observation_20260907_qwen3_4b.json` records a
successful gateway probe of the `qwen3-embedding-4b` alias and its returned
vector dimensions. This probe does not identify the loaded checkpoint commit;
the configured revision is the served alias until a backend revision is observed.

Execution profile v3 uses one combined request pool for generation and
embedding. Its current settings are defined in
[Throughput execution](THROUGHPUT_EXECUTION.md#shared-indexing-profile).

### Historical observations and profiles

The read-only observation in `configs/serving_observation.json`
records the server at 2026-09-06 22:28:51 UTC. The generation alias was associated
with `nvidia/Gemma-4-31B-IT-NVFP4`, ModelOpt NVFP4 quantization and cached snapshot
`4135a98a9b728a548947683219633b25682223ac`; the runtime dtype was `bfloat16`.
The 0.6B observation remains historical evidence only. The historical
serving commands did not explicitly pin `--revision`, and weights were not fully hashed.
These observations do not establish complete weight reproducibility. The
recorded single-deployment, tensor-parallel-size-one
layout is historical provenance, not the specification or evidence of a later
server redistribution. The observation files do not independently verify a later deployment.

Legacy profile versions 1 and 2 bound generation and embedding separately and
remain readable only for retained-run reproducibility. They do not describe the
current v3 queue. Timeout, retry, batch, and concurrency are operational
settings recorded apart from semantic method configuration.

Paper preflight also fixes the shared semantic service boundary: remote
embedding width `NEO4J_VECTOR_DIMENSIONS=2560`, embedding input limit
`MAX_EMBEDDING_LENGTH=32768`, `RAG_EMBEDDING_TOKEN_RESERVE=0`, generation
context `RAG_MAX_CONTEXT_LENGTH=262144`, query instruction exactly as recorded
in the canonical policy, and `NEO4J_FULLTEXT_ANALYZER=english`. MS GraphRAG's
`RAG_MS_EMBED_DIM` is the same 2560. Its native completion configuration omits
client output limits and temperature; the serving model supplies those defaults. Prehop
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

The adapter calls the pinned `agent_retrieval` entrypoint in `agent` mode,
including initial decomposition, up to five native IRCoT steps and final answer
generation. Observers capture the final answer and exact chunk order passed to
the final prompt; they do not reconstruct the agent loop. The post-answer native
LLM evaluator is replaced by the common paper evaluator. No ground truth is
passed into native generation. `agent_query_audit.jsonl` records the native
calls and returned evidence. The target remains `controlled_adapter` because
of backbone, structured-format and parallel-construction adaptations.

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
primary setup; all 16 target preflights; one chat probe; embedding at the selected profile batch/concurrency
with count/index/2,560-dimension/finite checks; selective
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

`scripts/paper_campaign.py` runs every indexing and benchmark stage under a
nohup/setsid supervisor by default. Launch verifies ignored SIGHUP, a distinct
session and process group, PID/start/boot identity and Linux child-subreaper
ownership. Systemd and own-user linger are needed only with `--backend systemd`;
neither is a default launch requirement. Reboot recovery is not claimed.
Owned descendants receive individually verified TERM cleanup with a 30-second
bound and no KILL escalation. Surviving identities block a new campaign.

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
available after the initiating task ends. Launch verifies actual supervisor
ownership; a submitted PID alone is not success. The separate nohup monitor
writes `monitor-*-observations.jsonl` every 10,800 seconds and checks terminal
state every five seconds. It verifies current admissions and records unavailable
ETA explicitly. `monitor-*-terminal.json` follows cleanup or supervisor exit;
these receipts are not automatic chat notifications.

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

Primary benchmarks use the selected content-bound profile concurrency (serial default one). Prehop structured format
retry shares the typed maximum of five wire attempts with transport retry and
disables the SDK's nested automatic retries. Response schemas, caps and prompts
remain fixed; the controlled retry profile is part of Prehop's method identity.

## Throughput execution

Select an absolute `RAG_EXECUTION_PROFILE` before starting Python. New campaign
supervisors own the bounded queue and preserve the profile in child environments.
Use the foreground queue wrapper for individual targets. Profiles change gate
context; rerun gates rather than transferring serial evidence. See
[THROUGHPUT_EXECUTION](THROUGHPUT_EXECUTION.md) for commands and cost definitions.


### Serving capacity

On 2026-09-07 the operator reported two serving GPUs with generation
`MAX_NUM_SEQS=32` per GPU (64 aggregate sequences) and embedding
`EMBEDDING_MAX_NUM_SEQS=512` per GPU (1,024 aggregate sequences). This is an
operator report, not a measured utilization or loaded-weight attestation.
Sequence capacity is not HTTP request concurrency: an embedding request can
contain several inputs, and KV cache and input length can limit actual work.

The shared profile and its validation limits are listed in
[Throughput execution](THROUGHPUT_EXECUTION.md#shared-indexing-profile).
Measure useful completed work under fixed resources before increasing limits;
a larger client queue alone does not add serving capacity.

### Indexing-only supervisor

`index_matrix.py` uses nohup and an independent process session for smoke builds
and full indexes. Its status file records each target; it has no separate
three-hour monitor, automatic chat notification, full-query evaluation or
benchmark admission stage. See the [index batch procedure](THROUGHPUT_EXECUTION.md#index-only-batch).


### Prehop trace storage

Prehop writes full stage and inference traces under ignored `data/traces` and
intermediate document output under `data/debug`. Its defaults apply to both
indexing and benchmark entrypoints; campaign environments preserve
`RAG_PREHOP_TRACE` and `RAG_PREHOP_TRACE_DIR`. Storage must be writable for the
selected main Python process. See
[Prehop tracing](ARCHITECTURE.md#prehop-tracing) for the event format and storage contract.

### Local files and Git

The repository ignores prepared runtimes (`data/runtime_envs`), native runtime
homes (`data/official_baselines`), generated corpora, indexes, results, debug
files, and traces. Keep these files on disk when retained runs reference them;
Git exclusion is not a cleanup policy. `.env` and local overrides stay private,
while `.env.example`, execution-profile templates, and `uv.lock` are tracked.
The working manuscript and submission logistics also remain local. A custom
trace directory must be outside tracked paths or explicitly ignored before
staging files.

### Controlled extraction and artifact validation

The adapters validate the inputs consumed by pinned native parsers and the
artifacts used for retrieval. Their versioned contracts are registered in
`core/strategy_registry.py`; external source files are not edited.

| Adapter | Boundary and failure handling |
|---|---|
| HippoRAG2 | Requests the native NER/triple item schemas; validates every response before native parsing. Explicit entity/text names and unambiguous triple objects can be unwrapped without adding facts. |
| GFM-RAG | Applies the same JSON item validation before native parsing/evaluation. A swallowed native extraction failure still prevents index completion. |
| MS GraphRAG | Checks completion, tuple fields and report schemas through a registered completion provider. Separates completion markers when concatenating gleanings. Profile `strict-ms-native-glean-v4` removes an unstructured leading glean preamble only when the remaining tuples fully validate; tuple fields are preserved verbatim and raw/normalized responses are both recorded. Native relationship filtering remains unchanged and unresolved endpoint observations are recorded. |
| Youtu | Keeps its native extraction schema; validates nonempty item values, retries incomplete/malformed responses within one budget and records each attempt. Native source reachability remains observational. |
| LinearRAG | Index construction uses native local NER and pinned MPNet, without LLM extraction. Checks embedding counts/dimensions/finite values, persisted stores and passage-node coverage; native QA responses must finish with nonempty text. |

Validation changes are controlled-adapter behavior, not claims of identical
upstream-default execution. Prompts, graph algorithms, retrieval cutoffs and
pinned method backbones retain their registered settings. HippoRAG2/GFM-RAG
request field-level JSON schemas rather than merely JSON objects; their original
token limits remain in force. Truncated JSON is not guessed or completed by the
adapter. An invalid response exhausts a bounded retry budget and fails the run.
HippoRAG2 format retries bypass invalid native cache entries. MS guarded calls
disable native response caching so retries cannot replay the same invalid entry.

`extraction_audit.jsonl` records prompts, original responses, available usage,
normalization and retry/exhaustion decisions. LinearRAG records native QA text
in `answer_audit.jsonl`. Format validation is separate from semantic correctness:
a valid but incorrect model answer must be scored as produced. Missing provider
telemetry remains unavailable rather than becoming a fabricated zero cost.

MS GraphRAG query streaming preserves native chunks and records the complete
stream, including its finish reason. An incomplete stream fails the query;
partially delivered streams are not replayed as fresh successful answers.

MS completion requests omit `max_tokens`, `max_completion_tokens` and
`temperature`, matching the pinned `ModelConfig.call_args={}` default. This
leaves effective limits with the server; it does not mean unlimited generation.
The temporary 4,096-token experiment was superseded before full indexing.
A `length` finish fails immediately without repeating the identical request and
budget. Other format errors retain bounded retries; partial tuples are never
admitted as a complete extraction.

Completed guarded indexes bind the audit's byte prefix and SHA-256. Queries may
append audit records without changing that index-time prefix. Publication
verification rejects missing, altered, exhausted or stale-profile evidence;
post-query inventories bind the final artifact bytes. Full source/query coverage
and metric recomputation remain mandatory under the
[result admission contract](RESULTS.md#admission-checks). Smoke tests and single
fixture queries do not admit a full benchmark result.

### Native parameter comparison

Run the following to compare selected method-defining defaults against
local pinned sources. The registry records HippoRAG2 retrieval candidates 200
separately from QA context 5, LightRAG top-k 40 and chunk top-k 20, LinearRAG
top-k 5, and Youtu native agent mode/top-k 20. GFM uses the upstream single-pass
`qa.py`/`qa_inference.yaml` workflow with top-k 5 and its native QA prompt and
omitted output cap. The optional IRCOT workflow is a different variant.

```bash
"$PYTHON_BIN" -m scripts.audit_native_parameters --home /absolute/pinned-runtime-home --output /absolute/receipt.json
```

| Method | Generation output policy | Retrieval policy |
|---|---|---|
| Prehop / Naive | Repository-owned question generation 4,096; short-answer synthesis 128. Naive does not generate index questions. | Shared six-sentence windows; repository top-k 12. |
| MS GraphRAG | Native omission of output caps and temperature. | Native local search: 10 entities, 10 relationships, 12,000 context tokens; one glean. |
| LightRAG | Native completion call arguments are forwarded. | Native mix mode, top-k 40, chunk top-k 20. |
| HippoRAG2 | Native NER 512, triples 2,048; QA default 2,048. | 200 retrieval candidates; five QA passages. |
| GFM-RAG | Native NER 300, triples 4,096; single-pass QA omits output cap. | Native single-pass QA top-k 5. |
| LinearRAG | Native QA 2,000 tokens, temperature 0. | Native top-k 5, BFS retrieval, three iterations. |
| Youtu | Native temperature 0.3 and omitted output cap. | Native agent mode, up to five IRCoT steps, top-k/filter 20. |

Prehop and Naive are repository-owned methods. Shared model replacement,
seed/transport controls, user-selected concurrency and structured-output guards
remain declared experiment adaptations. In particular, parallel Youtu schema
evolution is not claimed to be serial-equivalent. Matching selected native
parameters does not establish full upstream-identical execution.
