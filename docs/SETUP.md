# Runtime and infrastructure setup

For operators preparing the services and native runtimes used by the
[README](../README.md). Experiment commands and measurement rules are in
[Reproducing](REPRODUCING.md); adapter behavior is in [Method](METHOD.md).

## Main Python environment

Use the README installation for a new checkout. The main Python environment is
owned by `pyproject.toml` and `uv.lock`; it includes MS GraphRAG. To prepare a
separate environment without replacing an active one:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/new/main-venv
uv sync --locked --python 3.12 --extra dev
export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
```

Both selectors must refer to the same environment. Shell entrypoints otherwise
prefer `.venv/bin/python`, then Python on `PATH`. A missing explicit interpreter
is an error. Running commands never install or synchronize dependencies.
HopRAG launchers select its separately prepared coordinator automatically.

## Common inference gateway

Use the public inputs in [.env.example](../.env.example): one OpenAI-compatible
LiteLLM base and credential, registered generation/embedding aliases, and the
actual embedding dimensions. Both endpoints must be served. Generation must
support the requested JSON-schema format. Model aliases, dimensions and request
limits for the paper are defined in the [registry](../core/strategy_registry.py).
Changing them defines another experiment. Preserve actual weight-revision
evidence when available; a serving alias alone does not establish it.

`RAG_INFERENCE_TIMEOUT` and `RAG_INFERENCE_RETRY_ATTEMPTS` control the common
transport. A method's local model or input budget can differ from the gateway
model/limit. Native compatibility variables are injected into child processes;
legacy provider variables are not another public configuration interface.
Credentials stay in the local environment and must not enter saved artifacts.

`run_servers.sh all` checks Neo4j and both gateway model names. A model-list check
does not exercise inference; the README smoke does. The script can start a
local default Neo4j, but does not start model processes. Prepare remote hosts,
custom ports, database permissions and model services yourself.

## Configuration ownership and precedence

| Source | Responsibility |
| --- | --- |
| `core/strategy_registry.py` | Static method/transport defaults, native budgets and pinned revisions |
| `core/config.py` | Retrieval/evaluation settings consumed by core code |
| `core/execution_profile.py` | Resolve operational defaults, caller values and selected throughput profile |
| `core/inference_transport.py` | Effective request contract for clients, workers and provenance |
| `core/paper_policy.py` | Run identity, namespace, output/cache paths and paper generation seed semantics |
| `configs/paper_runtime_requirements.json` | Native interpreter, dependency, local model and content requirements |

Launchers load `.env` through `scripts/lib.sh` or
`scripts/runner_environment.py`, preserving exports. Priority is registry
defaults < `.env` < exported values < the profile's matching throughput fields.
`RAG_SKIP_PROJECT_ENV=true` skips only file loading. Select a profile before
Python starts; `RAGConfig` retains import-time values. Read-only configuration
inspection passes copied settings through `resolved_target_environment` instead
of temporarily changing `os.environ`.

The README selects [direct-8.json](../configs/execution_profiles/direct-8.json).
Use that file to inspect/change producer and request limits; the registry owns
unprofiled defaults. External workers' existing `RAG_<STRATEGY>_EMBEDDING_*`
controls can override global embedding batch/concurrency/retries and are recorded
in provenance. Requests go directly to the gateway. Per-client limits are not
a cross-process semaphore or a statement of GPU capacity; nested native workers
and unrelated clients affect load.

Paper generation omits the LLM seed; evaluation/sampling seeds are separate.
Recorded native settings and historical results remain unchanged. Setup checks
source/dependency integrity; dispatch does not repeat model-hash, dependency or
semantic-equality approval checks. Missing dependencies surface through imports.

## Source and setup isolation

Run baseline setup on Linux x86-64 with Bash, Git, `flock` (util-linux), `uv` and
internet access. The setup fetches pinned sources, provisions interpreters and
downloads local models. It does not install host GPU drivers or compilers.
GFM-RAG additionally needs a CUDA development toolkit (CUDA 12 or newer; its
upstream recommends 12.6.3) and a C++ compiler for native graph operations.
These requirements are independent of the remote inference gateway.

```bash
./scripts/setup_official_baselines.sh
```

The script prepares the native methods declared by the registry. Their exact
Python versions, constraints and model snapshot hashes are in
[the runtime manifest](../configs/paper_runtime_requirements.json), with package
versions in its referenced constraints. GFM-RAG includes its learned checkpoint
and ColBERT linker; LinearRAG uses pinned local MPNet/spaCy components. MS
GraphRAG comes from the main lockfile.

`RAG_OFFICIAL_BASELINE_HOME` selects the installation root, defaulting to
`data/official_baselines`. Export an absolute path before setup and execution
when relocating. External methods use `<home>/<strategy>/{source,artifacts,venv}`
and record installed dependency freezes. Upstream revisions are exported to
unique build directories and checked without patching original source. Freezes
record installed metadata; they are not universal cross-platform locks.

Use a fresh home for a replacement. Preserve existing failed attempts for
inspection and never replace an environment used by an active process.

### HopRAG runtime

The common setup includes HopRAG. For that method alone:

```bash
python3 scripts/setup_hoprag_runtime.py
```

`models/hoprag/runtime_paths.py` owns `<home>/hoprag`. It contains `main-env`
for coordinator/native imports, `pos-env` for the PaddleNLP CPU worker, `source`,
`pos-model`, separate environment freezes and a successful `setup.json` receipt.
Constraint files and the POS model manifest own versions/hashes.

Setup builds under `<home>/.builds`, checks dependencies, native imports, source
integrity and an actual POS request, then publishes a symlink to preserve venv
interpreter paths. Repeating setup verifies/reuses the installation. A mismatch
leaves it intact; choose a fresh home. Generated POS files use run-local storage.

## Storage and ongoing work

Keep corpus/query manifests, original indexes, results, traces and prepared
runtimes used by retained or active runs. Git exclusion is not deletion authority.
Source checkouts and model snapshots remain immutable; generated native caches
belong in run-local storage. `.env` stays private.

Prehop trace storage must be writable; its data and timing contract is in
[Method](METHOD.md#prehop-tracing). Use new run IDs/namespaces for independent
experiments. `--clear-graph` and `clear_graph` delete the entire selected Neo4j
database/schema and are not normal setup or cleanup steps.
