# Runtime Requirements

Use this guide to prepare method runtimes and understand their native output
contracts. Query behavior belongs in [Architecture](ARCHITECTURE.md); launch and
completion behavior belongs in [Execution](THROUGHPUT_EXECUTION.md).

`core/strategy_registry.py` defines supported methods, pinned upstream revisions,
and paper policies. `configs/paper_runtime_requirements.json` defines required
Python versions, packages, local model revisions, and content hashes. This guide
explains how to prepare and use those runtimes.

## Main Python environment

Select a prepared environment explicitly. Entrypoints use it without dependency
synchronization or automatic lockfile checks. HopRAG uses its separately prepared
native runtime. The main prepared environment uses Python 3.12 and includes
MS GraphRAG; package requirements are in `pyproject.toml` and `uv.lock`.

For a new main environment:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/new/main-venv
uv sync --locked --python 3.12
export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
```

When both selectors are supplied, they must identify the same environment.
Do not replace an environment used by an active process. Runtime entrypoints
execute the selected interpreter directly; a missing explicit interpreter does
not select a different one silently.

Without an explicit selector, shell entrypoints prefer `.venv/bin/python`,
then `python3` or `python` on `PATH`. This fallback selects an interpreter; it
does not install dependencies. Shell launchers load `.env` and preserve already
exported values. Direct Python tools may require exported settings; follow the
environment-loading instructions for the selected command.

## Common inference gateway

Configure `.env` or the exported environment with:

- `RAG_INFERENCE_BASE_URL`
- `RAG_INFERENCE_API_KEY`
- `RAG_GENERATION_MODEL=gemma-4-31b-it`
- `RAG_EMBEDDING_MODEL=qwen3-embedding-4b`

Generation and embeddings share one OpenAI-compatible LiteLLM base. Shell
launchers require the canonical base, key, and model variables and reject legacy
provider inputs. `core/inference_transport.py` records the normalized endpoint
identity without comparing it against `configs/paper_gateway.json`. Configure
the shared gateway; automatic gateway approval and model-equality checks are
removed. Credentials must not enter commands, logs, or artifacts.
Legacy provider variables are not an alternative public interface; compatibility
names are injected only into isolated native children.

The remote embedding contract is 2,560 dimensions, a 32,768-token input limit,
and zero token reserve. The generation context limit is 262,144. Query
instructions and templates are method-specific registry policy. An observed
serving alias does not prove the exact loaded weight revision; preserve backend
revision evidence when available and label historical observations as such.
LightRAG's registered native embedding input budget is 8,192 tokens, below the
shared endpoint limit. Method-specific client budgets remain part of provenance.

New paper generation requests omit the LLM seed even if a stale ambient
`RAG_LLM_SEED` is present. Dataset order, evaluation, and sampling use a separate
seed of 42. Historical results keep their recorded generation settings.

## Source and setup isolation

Pinned upstream checkouts are immutable. Package setup exports the pinned
revision into a unique build directory before installation and checks original
source integrity. Run setup from a Linux x86-64 host with Bash, Git, `flock`
(util-linux), `uv` and internet access. The native package pins target this
platform. Setup installs the required Python versions through `uv` and fetches
sources and model files; gateway credentials and Neo4j are not needed until
indexing or querying.

`RAG_OFFICIAL_BASELINE_HOME` selects the installation directory, defaulting to
`data/official_baselines` under the repository root. Export an absolute path
before both setup and execution to use another location. LightRAG, GFM-RAG and
LinearRAG use `<home>/<strategy>/{source,artifacts,venv}`. HopRAG's layout is
described [below](#hoprag-runtime). Use a fresh runtime home when preparing a
replacement; setup does not clean an existing attempt.

```bash
./scripts/setup_official_baselines.sh --primary
```

This provisions **HopRAG, LightRAG, GFM-RAG and LinearRAG** using registry
revisions. MS GraphRAG is installed with the main environment. The latter three
runtimes record `runtime.freeze.txt`; HopRAG records separate main/POS freezes.
Setup checks source integrity, declared constraints and installed dependencies.
The freeze records installed metadata; it is not an automatic dispatch check
or a universal cross-platform lock. Large checkpoint downloads can take time;
completion is reported only after installation checks pass.

## Pinned external runtimes

| Method | Python | Required local components |
|---|---|---|
| MS GraphRAG | 3.12 | `graphrag==3.1.2`, LiteLLM |
| LightRAG | 3.10 | Pinned upstream package and reviewed direct constraints |
| GFM-RAG | 3.12 | GFM-RAG-8M checkpoint/config, ColBERT entity linker |
| LinearRAG | 3.9 | Pinned MPNet snapshot and `en_core_web_trf-3.6.1` |
| HopRAG | 3.12 main; 3.10 POS | Pinned native source, PaddleNLP/PaddlePaddle POS runtime |

GFM-RAG loads `rmanluo/GFM-RAG-8M` at revision
`4da9e4655d126a783ae2b795ab73b7c7a7c3f4ac` and
`colbert-ir/colbertv2.0` at `c1e84128e85ef755c096a95bdb06b47793b13acf`.
Required checkpoint and model-file hashes are in the runtime manifest. Mutable
PLAID caches use run-local storage rather than modifying the pinned snapshot.
Its pinned upstream requires a CUDA development toolkit (CUDA 12 or newer;
upstream recommends 12.6.3) and a C++ compiler for native graph operations.
Provision these host tools before running GFM-RAG; Python package setup does not
install a GPU driver or system compiler. These local components are separate
from the remote generation/embedding gateway.

LinearRAG loads `sentence-transformers/all-mpnet-base-v2` at
`e8c3b32edf5434bc2275fc9bab85f82640a19130` from a local snapshot. Its GPL upstream
source remains process-isolated from this repository. Its controlled Qwen mode
is a separate configuration, not the primary MPNet result mode.

### HopRAG runtime

HopRAG is included in the primary setup command. To install only HopRAG:

```bash
python3 scripts/setup_hoprag_runtime.py
```

The installer fetches native upstream revision
`a6e425b8f8a5d8131dd7805db40185ac76e09903`, creates Python 3.12 main and Python
3.10 POS environments, and downloads the native PaddleNLP POS model. Package
versions are pinned in `configs/runtime_constraints/hoprag_main.txt` and
`hoprag_pos.txt`; their digests are declared in the runtime manifest. The main
environment uses CPU PyTorch. Model files must match
`configs/hoprag_pos_model.json`.

`models/hoprag/runtime_paths.py` owns the shared `<home>/hoprag` location:

| Path | Contents |
| --- | --- |
| `main-env/bin/python` | Coordinator and native HopRAG imports |
| `pos-env/bin/python` | PaddleNLP/PaddlePaddle CPU POS worker |
| `source/` | Unmodified pinned HopRAG checkout |
| `pos-model/` | Downloaded, hash-checked native model inputs |
| `main-env.freeze.txt`, `pos-env.freeze.txt` | Installed package identities |
| `setup.json` | Successful setup receipt |

Setup first builds under `<home>/.builds/` and checks package compatibility,
native imports, source integrity and an actual POS request. It then publishes
`<home>/hoprag` as a symlink to the verified build, preserving virtualenv
interpreter paths. Repeating the command verifies and reuses that installation.
A failed build remains unpublished for inspection. A mismatched existing
installation is left intact; select a fresh `RAG_OFFICIAL_BASELINE_HOME` and
rerun setup. No existing runtime is upgraded or removed by this command.

The repository's root `third_party/HopRAG` is reference-only. Generated POS
files use run-local directories; the downloaded model and upstream checkout
remain unchanged. Launchers automatically select the prepared main interpreter.

`scripts/run_hoprag_scheduled.py` performs a 16-document source canary, full
MultiHop-RAG indexing, query benchmarking, and completion recording. The canary
executes a query after indexing. Native chunks remain sequential within
a document; the adapter uses ten document workers. The runtime uses the native
POS model, not a substitute spaCy path.

## Runtime selection

Launchers select the configured interpreter and upstream runtime directly.
Dispatch records settings and runtime identities without dependency/revision,
gateway-approval, model-hash or semantic-equality gates. Required transport
inputs and explicit interpreter selection still apply; missing dependencies or
files surface through actual imports or I/O. Setup-time source and dependency
checks are separate from runtime dispatch.

## Native output handling and recording

Adapters preserve native source and declare response interventions separately.

| Method | Response contract |
|---|---|
| Prehop | Requested structured schemas, JSON decoding retries, and final text synthesis; no local response-schema rejection |
| Naive RAG | Shared text synthesis; no index question generation |
| HopRAG | `adapter-json-recovery-v2`: parse valid plain/fenced JSON before the native cleaner; return native completion results unchanged without adapter retries |
| GFM-RAG | Native JSON mode and empty-list fallback; 300-token NER limit |
| MS GraphRAG | Observe response/stream data while retaining native parsing and glean/report behavior |
| LightRAG | Native extraction and document status; failed insertion is not successful processing |
| LinearRAG | Native local NER and QA text without an additional answer-quality gate |

Prehop's `prehop-native-json-retry-v2` metadata describes JSON decoding failures:
invalid JSON syntax or an invalid raw input type. Transport and decoding retries
share the configured maximum of five wire attempts, with unchanged requests and
SDK retries disabled. Decodable duplicate properties, nonfinite values or
schema-mismatched values are not rejected by an additional local validator.
This metadata correction does not change parsing behavior or relabel saved runs.

HopRAG retains raw wire, embedding, and recovery records. Recovery does not
invent questions or facts. Other native observation records distinguish returned
responses from native exceptions; observation alone does not establish indexing
success. Source counts and index metadata are recorded without an additional integrity gate.
Missing response or cost telemetry remains unavailable.

| Method | Generation output policy | Retrieval settings |
|---|---|---|
| Prehop | Index questions: 4,096 tokens; non-thinking answer synthesis: 256, with versioned system instructions and a question-first user message | Six-sentence passages; final top-k 12 |
| Naive RAG | Answer synthesis: 128 tokens, using the shared answer prompt | Six-sentence passages; final top-k 12 |
| MS GraphRAG | Native omission of client output caps and temperature | LocalSearch: 10 entities, 10 relationships, 12,000 context tokens; one glean |
| LightRAG | Native completion arguments | Mix mode; top-k 40; chunk top-k 20 |
| GFM-RAG | NER 300; triples 4,096; single-pass QA omits output cap | Native QA top-k five |
| LinearRAG | Native QA 2,000 tokens, temperature zero | Top-k five; `iteration_threshold=0.4`, `passage_ratio=2`, `top_k_sentence=3` |
| HopRAG | Native temperature 0.1 and 4,096-token cap | BFS; five hops; top-k eight |

GFM-RAG's adapter applies the shared 600-second QA transport timeout instead of
the upstream 60-second call timeout. This is a transport override, not a source
edit. Client waiting may include the queue; a timeout does not prove upstream
inference was cancelled. Selected native parameter parity does not imply fully
upstream-identical execution after shared model/transport adaptations.

## Completion and campaign scope

See [completion and continuation](THROUGHPUT_EXECUTION.md#completion-and-continuation)
for receipts and resume behavior, and
[persistent ownership and recovery](THROUGHPUT_EXECUTION.md#persistent-ownership-and-recovery)
for supervisors and the legacy evidence-ledger workflow.

### Serving capacity

Client request concurrency and embedding batch size do not specify GPU sequence
capacity. Local checkpoints, input lengths, KV cache, and other clients affect
realized throughput. The configured shared profile is documented in
[shared indexing profile](THROUGHPUT_EXECUTION.md#shared-indexing-profile);
no utilization or physical GPU count is inferred from that configuration.

## Local artifacts and trace storage

Prepared runtimes, generated corpora, indexes, results, and traces remain
ignored. Ignoring a file is not permission to delete evidence referenced by a
retained or active run. `.env` stays private; public examples contain no secrets.
Prehop trace and intermediate-output storage must be writable. See
[Prehop tracing](ARCHITECTURE.md#prehop-tracing) for its data and timing contract.


### HopRAG edge recovery

Start new index attempts in fresh campaign namespaces. A directory's existence
does not establish completed indexing.

Large HopRAG edge groups use bounded exhaustive scoring in the adapter. All
answerable candidates are scored; no dense-only top-k prefilter is used. Native
document-pair sparse scores, duplicate-question grouping, tie order and final
edge selection are retained. Twenty-four fixtures matched native edge outputs
under the pinned pandas 2 runtime, including tied scores. This is validation of
those fixtures, not a guarantee of every possible numerical boundary case.
Cached nodes are reused; recovery costs retain prior attempt time separately.

The cached recovery launcher, `scripts/resume_hoprag_cached.py`, applies the
registered `RAG_HOP_DOC_WORKERS=10` before indexing. It removes the
retired `RAG_HOP_MAX_THREADS`, `RAG_HOP_GATHER_WAVE`,
`RAG_HOP_BUILD_CONCURRENCY`, `RAG_HOP_SEMANTIC_VARIANT`, and
`RAG_HOP_EDGE_FILTER` launch overrides. The launcher records the resumed
execution and checks its index completion status; selecting a runtime does not
establish completed edges.
The pinned upstream source is unchanged.

## Dataset compatibility

Repository-owned launchers support MultiHop-RAG and the pinned HippoRAG
HotpotQA release. [HOTPOTQA](HOTPOTQA.md) owns its corpus, preparation, and
sentence projection contract. Additional datasets supported by upstream
packages are outside this comparison.

### HopRAG launcher selection

Model-specific shell and Python launchers select HopRAG's pinned main runtime
instead of the common main interpreter. Its observed runtime identity is read
from that interpreter. The HopRAG document worker setting is 10. Prehop's shared
`RAG_HOP_GATHER_WAVE`, `RAG_HOP_BUILD_CONCURRENCY`, `RAG_HOP_SEMANTIC_VARIANT` and
`RAG_HOP_EDGE_FILTER` settings do not configure HopRAG and are not treated as
unknown HopRAG overrides. Native upstream files remain unchanged.


### Native output aliases and timeout alignment

MS GraphRAG may assign the same content-derived document ID to multiple corpus
files. The adapter retains all source aliases when decoding citations, without
rewriting native parquet tables, graph membership, search results or answers.
LightRAG's public `default_llm_timeout` and `default_embedding_timeout` arguments
use the configured transport timeout (600 seconds in the campaign). Native
worker watchdog behavior remains unchanged; the HTTP deadline alone does not
configure these internal wrappers. Index receipts record the applied values.
External index completion no longer invokes the removed extraction-validation
function. Native extraction observations and actual errors remain available.

LightRAG resume reads native status by staged source ID. Existing failed sources
receive one explicit native manual-retry request per invocation; ordinary insert
calls do not retry them. Only absent sources are inserted. Processed sources and
native caches remain in place, and pending work is drained by the native pipeline.
Completion uses source IDs because retries retain their original track IDs;
duplicate-attempt status stubs are not corpus sources. A repeated native failure
remains an error rather than being relabeled as successful processing.
