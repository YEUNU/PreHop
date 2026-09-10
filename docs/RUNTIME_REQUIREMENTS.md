# Runtime Requirements

The revised Prehop design removes initial query rewriting and evidence-conditioned
query regeneration/re-search. The original query searches all three channels at
every input length. The runtime uses this single-pass policy. Earlier Prehop scores and costs have been removed from paper reporting. The completed
original-query MultiHop-RAG run is now reported in [the result register](RESULTS.md);
other pending experiments require their own compatible full-run artifacts.

`core/strategy_registry.py` defines supported methods, pinned upstream revisions,
and paper policies. `configs/paper_runtime_requirements.json` defines required
Python versions, packages, local model revisions, and content hashes. This guide
explains how to prepare and use those runtimes.

## Main Python environment

Select a prepared environment explicitly. Standard paper entrypoints check its
installed dependencies and lockfile consistency without synchronizing it during
execution. HopRAG uses its separately prepared native runtime.

For a new main environment:

```bash
export UV_PROJECT_ENVIRONMENT=/absolute/new/main-venv
uv sync --frozen --no-install-project
export PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"
```

When both selectors are supplied, they must identify the same environment.
Do not replace an environment used by an active process. Runtime entrypoints
execute the selected interpreter directly; a missing explicit interpreter does
not select a different one silently.

## Common inference gateway

Configure `.env` or the exported environment with:

- `RAG_INFERENCE_BASE_URL`
- `RAG_INFERENCE_API_KEY`
- `RAG_GENERATION_MODEL=gemma-4-31b-it`
- `RAG_EMBEDDING_MODEL=qwen3-embedding-4b`

Generation and embeddings share one OpenAI-compatible LiteLLM base. Paper mode
checks its normalized identity against `configs/paper_gateway.json`. Empty
bases, URL userinfo, unsupported model overrides, and direct public-provider
fallbacks are rejected. Credentials must not enter commands, logs, or artifacts.
Legacy provider variables are not an alternative public interface; compatibility
names are injected only into validated isolated native children.

The remote embedding contract is 2,560 dimensions, a 32,768-token input limit,
and zero token reserve. The generation context limit is 262,144. Query
instructions and templates are method-specific registry policy. An observed
serving alias does not prove the exact loaded weight revision; preserve backend
revision evidence when available and label historical observations as such.

New paper generation requests omit the LLM seed even if a stale ambient
`RAG_LLM_SEED` is present. Dataset order, evaluation, and sampling use a separate
seed of 42. Historical results keep their recorded generation settings.

## Source and setup isolation

Pinned upstream checkouts are immutable. Package setup exports the approved
revision into a unique build directory before installation and checks original
source integrity. `RAG_OFFICIAL_BASELINE_HOME` selects the shared
`<home>/<strategy>/{source,artifacts,venv}` layout. Source and interpreter
overrides must resolve to the validated runtime. Use a fresh runtime home when
preparing a replacement; setup does not clean an existing attempt.

```bash
./scripts/setup_official_baselines.sh --primary
```

This provisions packaged external methods using registry revisions. It does not
provision HopRAG's separate main/POS environments. Each prepared environment
records `runtime.freeze.txt`. Preflight checks installed metadata and retained
freeze/constraint hashes; a local freeze is not a universal cross-platform lock.

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

LinearRAG loads `sentence-transformers/all-mpnet-base-v2` at
`e8c3b32edf5434bc2275fc9bab85f82640a19130` from a local snapshot. Its GPL upstream
source remains process-isolated from this repository. Its controlled Qwen mode
is a separate configuration, not the primary MPNet result mode.

### HopRAG runtime

The native upstream revision is `a6e425b8f8a5d8131dd7805db40185ac76e09903`.
`models/hoprag/native_runtime.py` resolves the prepared installation, including
`main-env`, `pos-env`, and immutable `source/third_party/HopRAG`. The repository's
root `third_party/HopRAG` is reference-only. POS model hashes are bound by
`configs/hoprag_pos_model.json`; generated POS files use run-local directories.

`scripts/run_hoprag_scheduled.py` performs a 16-document source canary, full
MultiHop-RAG indexing, query benchmarking, and completion recording. The canary
executes a query after indexing. Native chunks remain sequential within
a document; the adapter uses ten document workers. The runtime uses the native
POS model, not a substitute spaCy path.

## Runtime selection

Launchers select the configured interpreter and upstream runtime directly.
Automatic dependency/revision checks, gateway approval, model hash comparisons,
and semantic-setting equality checks have been removed. The selected settings
and runtime locations remain recorded in provenance. Missing dependencies or
files surface through the actual import or I/O operation.

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

HopRAG retains raw wire, embedding, and recovery records. Recovery does not
invent questions or facts. Other native observation records distinguish returned
responses from native exceptions; observation alone does not establish indexing
success. Source counts and index metadata are recorded without an additional integrity gate.
Missing response or cost telemetry remains unavailable.

| Method | Generation output policy | Retrieval settings |
|---|---|---|
| Prehop / Naive | Prehop index questions: 4,096 tokens; shared synthesis: 128 | Six-sentence passages; final top-k 12 |
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

### Original-question retrieval

Prehop sends the original question to its enabled search channels and retrieves
once. Initial rewriting and evidence-conditioned refinement have been removed,
including their response schemas and 32-word condition. Existing indexes remain
reusable when their construction settings match. Historical combined prompt
hashes are recognized only for the exact unchanged construction contracts;
source index records retain their original provenance.

## Completion and campaign scope

`record_paper_completion.py` records finished execution without final paper-policy
validation. `verify_paper_target.py` forwards to it for compatibility. Runtime,
source coverage, index-reuse, and checkpoint checks remain enabled; historical
seed differences do not require exception records. See
[completion records](RESULTS.md#completion-records).

### Durable campaign ownership

Index and rolling dispatch do not require a clean source digest merely to launch
the next job. The separate legacy full-matrix workflow in `paper_campaign.py`
and `run_paper_matrix.sh` still consumes its explicit evidence ledger. That path
is not a prerequisite for ordinary target execution or an automatic final
validation stage. It derives 14 targets from seven methods and two datasets;
legacy stage names containing `16` are identifiers, not current target counts.

Supervisors own process sessions and retain identity-bound status and logs.
They do not promise reboot recovery or chat notifications. Use
[execution procedures](THROUGHPUT_EXECUTION.md) for foreground targets and the
index-only supervisor. Do not modify an active supervisor's artifacts to change
its recorded history.

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

Runtime validation dispatches to the selected method's own validator. Start new
index attempts in fresh campaign namespaces; directories from failed readiness
checks do not establish reusable indexes.

Large HopRAG edge groups use bounded exhaustive scoring in the adapter. All
answerable candidates are scored; no dense-only top-k prefilter is used. Native
document-pair sparse scores, duplicate-question grouping, tie order and final
edge selection are retained. Twenty-four fixtures matched native edge outputs
under the pinned pandas 2 runtime, including tied scores. This is validation of
those fixtures, not a guarantee of every possible numerical boundary case.
Cached nodes are reused; recovery costs retain prior attempt time separately.

The cached recovery launcher, `scripts/resume_hoprag_cached.py`, applies the
registered `RAG_HOP_DOC_WORKERS=10` before runtime validation. It removes the
retired `RAG_HOP_MAX_THREADS`, `RAG_HOP_GATHER_WAVE`,
`RAG_HOP_BUILD_CONCURRENCY`, `RAG_HOP_SEMANTIC_VARIANT`, and
`RAG_HOP_EDGE_FILTER` launch overrides. The launcher validates the actual
HopRAG runtime before reuse; passing preflight does not establish completed
edge construction.
The pinned upstream source is unchanged.

## Paper benchmark scope

The paper targets MultiHop-RAG and the pinned HippoRAG HotpotQA release
(9,221 passages, 1,000 occurrences, 944 original question IDs). All seven native
runtime mappings remain available; the corpus change does not change model
implementations. See [HOTPOTQA](HOTPOTQA.md) for preparation and population
identity. Official scorer parity is tested. The common support projection uses
complete original sentences in returned passages without gold-driven selection.
Fullwiki data is retained separately and is not the active comparison protocol.
Repository-owned launchers support these two datasets; additional dataset support
inside pinned upstream packages is outside the paper.


Model-specific shell and Python launchers select HopRAG's pinned main runtime
instead of the common main interpreter. Its observed runtime identity is read
from that interpreter. The HopRAG document worker setting is 10. Prehop's shared
`RAG_HOP_GATHER_WAVE`, `RAG_HOP_BUILD_CONCURRENCY`, `RAG_HOP_SEMANTIC_VARIANT` and
`RAG_HOP_EDGE_FILTER` settings do not configure HopRAG and are not treated as
unknown HopRAG overrides. Native upstream files remain unchanged.
