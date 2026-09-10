# Prehop maintainer policy

This file defines repository maintenance rules. User setup belongs in
`README.md`; implementation behavior belongs in `docs/ARCHITECTURE.md`;
external runtime requirements belong in `docs/RUNTIME_REQUIREMENTS.md`; profile
and cost details belong in `docs/THROUGHPUT_EXECUTION.md`; and result evidence
belongs in `docs/RESULTS.md`.

## Sources of truth

- `core/strategy_registry.py` owns strategy identity, primary order, upstream
  revision, runtime worker, output root, transport profile, and legacy status.
  Do not restore retired strategy branches outside the registered comparison set.
- `core/config.py` and checked-in configuration files own runtime defaults and
  semantic settings. Documentation must describe them, not redefine them.
- `docs/ARCHITECTURE.md` owns module boundaries and indexing/query behavior.
- `docs/RUNTIME_REQUIREMENTS.md` owns pinned environments and gateway checks.
- `docs/THROUGHPUT_EXECUTION.md` owns execution-profile fields, queue semantics,
  launch procedures, and cost definitions.
- `docs/RESULTS.md` owns result status and artifact-to-number traceability.
  Keep provisional diagnostics outside `docs/`; completed values must identify
  their query population, denominators and evidence paths.

Do not make a gitignored manuscript or private submission note a prerequisite
for understanding, running, or validating the public repository.

## Change policy

External source files remain unchanged. Preserve native behavior unless a
registered adapter profile explicitly declares a recovery intervention. HopRAG
JSON/return-value recovery is such an
intervention; retain raw responses, transformations and bounded retry records.
Do not fabricate missing entities, answers, usage or successful completion. See
`docs/RUNTIME_REQUIREMENTS.md` for the per-method output-handling contract.

- Write all documents under `docs/` in English. Keep temporary reviews, audit
  reports, revision logs, and live progress snapshots out of `docs/`; retain
  durable specifications and completed benchmark evidence.
- Use technical-writing guidance for runtime and developer documentation, and
  paper-writing guidance for manuscript claims and experiment specifications.
  Keep the checklist limited to evidenced checks and outstanding requirements.
- Keep one detailed contract per topic. Other documents should link to it and
  include only the context their readers need.
- Update the implementation and its owning document together. Keep current
  contracts rather than separate chronological change logs.
- Keep README examples representative and short. Do not copy experiment gates,
  full option catalogs, progress counters, or result tables into it.
- Keep this file prescriptive. Do not duplicate tutorials, architecture maps,
  model tables, queue internals, or live campaign state here.
- Delete a document only when its entire role is covered by another document;
  otherwise narrow its scope and preserve links.
- Preserve exact code symbols, commands, model IDs, dimensions, dates, metrics,
  and warnings when editing technical prose.

## Implementation boundaries

- Treat pinned upstream source trees as immutable. Build packages from exported
  exact revisions in isolated runtime directories; do not patch or install from
  the source checkout.
- Adapters may normalize inputs, configure the declared transport and producer
  concurrency, prepare run-local schemas, record sidecars, and validate native
  outputs. A method-defining retrieval, graph, serialization, or answer change
  requires a distinct semantic identity.
- Remote generation and embedding use the single fail-closed LiteLLM gateway.
  Do not introduce ambient-provider or direct-vendor fallback paths.
- A model, revision, vector dimension, corpus fingerprint, semantic setting, or
  execution profile change requires compatible new evidence. Never relabel an
  old artifact as current.
- Preserve dataset-specific units and denominators. MultiHop-RAG and HotpotQA
  results are not interchangeable.

## Execution and repository hygiene

- Use a unique run ID and strategy-scoped output namespace. Never clear shared
  graph or artifact state while another run may be active.
- Source/configuration edits do not by themselves block index dispatch. Keep
  executed code and settings in provenance; retain artifact, corpus and resume
  compatibility checks. Preserve failed attempts and original phase costs.
- Keep credentials out of commands, logs, artifacts, and tracked files.
- Generated corpora, indexes, results, traces, runtime homes, private submission
  notes, and manuscript drafts remain ignored. Do not force-add them.
- Before cleanup, resolve exact targets and confirm that no active supervisor or
  child owns them. Cleanup must be explicit and must not alter upstream source.
- Do not modify original model implementations merely to improve instrumentation
  or throughput. Keep scheduling and observation changes in repository-owned
  adapters and execution infrastructure.

## Verification

Run checks proportional to the change. For documentation-only changes, verify
Markdown links, referenced files and anchors, terminology, and configuration
values. For code or configuration changes, also run the affected tests plus:

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q core cli models utils scripts main.py
uv run --extra dev pytest -q
```

Never describe a runtime, index, benchmark, or result as complete based only on
configuration validation or a running process. The owning artifact and its
documented verification must support the claim.

- Final paper-policy validation is disabled. `record_paper_completion.py` records
  completion; the legacy verifier entry is a compatibility shim. Do not require
  seed exceptions or restore automatic final validation. Keep final result links
  in `docs/RESULTS.md`; retain runtime, index-reuse and checkpoint checks.
