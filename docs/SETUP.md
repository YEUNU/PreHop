# Runtime and infrastructure setup

Configure Python, Neo4j and the inference gateway before running the
[quick start](../README.md#quick-start).
Experiment commands and measurement rules are in [Reproducing](REPRODUCING.md);
method behavior is in [Method](METHOD.md).

## Main Python environment

Use the README installation for a new checkout. The main Python environment is
owned by `pyproject.toml` and `uv.lock`. To prepare a separate environment
without replacing an active one:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/new/main-venv
uv sync --locked --python 3.12 --extra dev
export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
```

Both variables must refer to the same environment. Shell entrypoints otherwise
prefer `.venv/bin/python`, then Python on `PATH`. A missing explicit interpreter
is an error. Running commands never install or synchronize dependencies.

## Common inference gateway

Use the public inputs in [.env.example](../.env.example): one OpenAI-compatible
LiteLLM base and credential, registered generation/embedding aliases, and the
actual embedding dimensions. Both endpoints must be served. Generation must
support the requested JSON-schema format. Model aliases, dimensions and request
limits for the paper are defined in the [registry](../core/strategy_registry.py).
Changing them defines another experiment. Preserve actual weight-revision
evidence when available; a serving alias alone does not establish it.

`RAG_INFERENCE_TIMEOUT` and `RAG_INFERENCE_RETRY_ATTEMPTS` control the common
transport. Legacy provider variables are not another public configuration
interface. Credentials stay in the local environment and must not enter saved
artifacts.

`run_servers.sh all` checks Neo4j and both gateway model names. A model-list check
does not exercise inference; the README smoke does. The script can start a
local default Neo4j, but does not start model processes. Prepare remote hosts,
custom ports, database permissions and model services yourself.

## Configuration ownership and precedence

| Source | Responsibility |
| --- | --- |
| `core/strategy_registry.py` | Static method and transport defaults for Prehop and the dense baseline |
| `core/config.py` | Retrieval/evaluation settings consumed by core code |
| `core/execution_profile.py` | Resolve operational defaults, caller values and selected throughput profile |
| `core/inference_transport.py` | Effective request contract for clients and provenance |
| `core/paper_policy.py` | Run identity, namespace, cache paths and paper generation seed semantics |
| `core/runtime_requirements.py` | Interpreter, installed-distribution and lockfile identity recorded with artifacts |

Launchers load `.env` through `scripts/lib.sh` or
`scripts/runner_environment.py`, preserving exports. Priority is registry
defaults < `.env` < exported values < the profile's matching throughput fields.
`RAG_SKIP_PROJECT_ENV=true` skips only file loading. Select a profile before
Python starts; `RAGConfig` retains import-time values. Read-only configuration
inspection passes copied settings through `resolved_target_environment` instead
of temporarily changing `os.environ`.

The benchmark runtime exposes HOP/NEXT expansion, scoring and ranking controls. Question-role indexing, expansion depth, path decay and
reciprocal metadata materialization are fixed in the registry; their former
environment overrides are inactive. Use the
[experiment inventory](REPRODUCING.md#paper-experiment-inventory) to select a
reported comparison. The paper's deeper-neighborhood and admission experiments
use saved-input analysis packages; they are not additional runtime depth flags.

The README selects [direct-8.json](../configs/execution_profiles/direct-8.json).
Use that file to inspect/change producer and request limits; the registry owns
unprofiled defaults. Requests go directly to the gateway. Per-client limits are
not a cross-process semaphore or a statement of GPU capacity; unrelated clients
affect load.

Generation in the paper experiments omits the LLM seed; evaluation and sampling
seeds are separate. Temperature zero does not guarantee identical responses to
repeated requests. Preserve each generation separately, including its actual
model provenance and settings.

Re-scoring stored answers and replaying fixed candidate sets require the
archived inputs listed in [Reproducing](REPRODUCING.md#fixed-input-analyses).
Those offline checks do not require starting the gateway or rebuilding indexes.

## Data storage

Keep corpus/query manifests, original indexes, results and traces together
with the run IDs that produced them.
Downloaded data and generated runs are excluded by the repository's
[Git ignore rules](../.gitignore). Keep credentials in `.env` or the process
environment; [.env.example](../.env.example) documents the public inputs.

Prehop trace storage must be writable; its data and timing contract is in
[Method](METHOD.md#prehop-tracing). Use new run IDs/namespaces for independent
experiments. `--clear-graph` and `clear_graph` delete the entire selected Neo4j
database/schema and are not normal setup or cleanup steps.
