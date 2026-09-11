# Prehop maintainer policy

This file defines repository maintenance rules. User setup belongs in
`README.md`; implementation behavior belongs in `docs/ARCHITECTURE.md`;
external runtime requirements belong in `docs/RUNTIME_REQUIREMENTS.md`; profile
and cost details belong in `docs/THROUGHPUT_EXECUTION.md`; and result evidence
belongs in `docs/RESULTS.md`.

Research result registers, paper checklists, experiment plans, manuscripts,
presentation exports, and paper-only renderers are local-only. Their paths below
identify local owners, not files required by a public checkout. Never force-add
them; an intentional research release requires explicit authorization.

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
  concurrency, prepare run-local schemas, record sidecars, and decode native
  outputs. A method-defining retrieval, graph, serialization, or answer change
  requires a distinct semantic identity.
- Remote generation and embedding use the configured shared LiteLLM gateway.
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
  executed code and settings in provenance. Do not reintroduce approval, preflight,
  hash/configuration matching, output-schema rejection, or duplicate-run guards.
  Preserve actual execution errors and original phase costs.
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
  in `docs/RESULTS.md`. Preserve runtime observations, index-reuse provenance and
  checkpoints without turning them into approval gates.

## Controlled experiment reporting

The active HotpotQA setting is the pinned HippoRAG v1 corpus, not fullwiki.
Preserve all 1,000 released occurrences and their 944 original question IDs;
cluster uncertainty estimates by original identity. Store experimental timing
connections in Neo4j. Keep frozen-prefix latency distinct from end-to-end latency
and co-evidence reachability distinct from semantic correctness. Label additive
full-query latency estimates separately from existing measured latency; never
present connection-only replay as a new end-to-end execution. Experiment
design belongs in `docs/PAPER_ABLATION_DESIGN.md`; do not add progress diaries.

## Manuscript: research argument, not an engineering specification

These rules apply to `docs/prehop_paper.md`, its rendered paper, captions, and
all manuscript appendices, presentation slides and speaker notes, and text
inside embedded diagrams and graphs. Apply them when adding results as well as
when revising prose. They do not prohibit implementation detail in developer docs.

- Write for a research reader assessing the question, method, evidence, and
  implications, not an operator running the repository. Organize the argument
  around research questions and experimental comparisons, not execution stages,
  artifact lifecycle, or implementation ownership.
- Keep a detail only if it defines the method, affects experimental validity,
  explains a result, or is necessary to reproduce a scientific comparison.
  Technical specificity is not itself an engineering problem. Preserve model
  identities, representation definitions, budgets, metrics, populations,
  statistical procedures, and timing boundaries when they affect conclusions.
- Exclude commands, local paths, code symbols, environment settings, hashes,
  runtime namespaces, database relationship names, schema/retry mechanics,
  dispatch rules, completion gates, and instructions to fill or verify results.
  Keep these in their owning developer or experiment-design document; do not
  move them into a manuscript appendix as a workaround.
- Describe experimental controls by what is held fixed and what changes.
  Replace storage/replay/adaptor mechanics with the scientific operation and
  its measurement boundary. Do not obscure differing latency scopes, unequal
  candidate budgets, or corpus adaptations in the process.
- Use declarative academic prose. Do not address the author/operator with
  'report', 'preserve', 'validate', 'do not copy', or similar task instructions.
  Pending experiments may be described prospectively; never imply they ran.
- Keep pending result cells empty as requested. State their status concisely
  in the relevant caption or paragraph, not as repeated project-management
  narration. Captions explain the comparison, units, and population rather
  than announcing a reserved result slot or future insertion procedure.
- Put broad qualifications and threats to validity in Limitations. Keep local
  distinctions essential to interpreting a metric or comparison near that
  definition. Do not repeat defensive disclaimers throughout the paper.
- Appendices contain research material: derivations, evaluation definitions,
  supplementary results, and method-defining prompts/settings. They are not
  runbooks. Preserve exact prompt text when it is presented as the prompt used.
- Prefer a short scientific explanation to a bookkeeping table. Merge tables
  only when the resulting comparison remains readable and metric definitions
  remain distinct; preserve measured values and unmeasured cells.
- Before rendering, review every section and caption for repository-specific
  narration, operational instructions, redundant status text, and unsupported
  claims. Check result preservation, references, and rendered layout. This is
  an editorial check, not a runtime approval or benchmark execution gate.

- In manuscript CI reporting, retain the confidence level, paired/cluster
  procedure, resample count, and seed. Do not add explanatory prose about what
  uncertainty CIs represent or repeated disclaimers about their interpretation.
- Timing comparisons use one measured execution per condition and question,
  without repeated timing trials or per-question warm-up. Bootstrap resampling
  does not dispatch additional benchmark executions.
- Manuscript schematics use diagram nodes, arrows, and concise labels rather
  than explanatory footers or bottom legends. Keep necessary explanations in
  the paper caption. Preserve editable SVGs and synchronized PDF/PNG exports;
  check text padding, arrow endpoints, and readability at final paper size.

## Claim validity when updating research results

- Tie every empirical claim to its population, configuration, measured endpoint,
  and matched comparison. Do not treat structural reachability as observed
  retrieval success, semantic link correctness, or full reasoning-path recovery.
- A fixed-graph search-channel contrast tests starting-passage retrieval, including
  candidate breadth; do not attribute its gain to link construction or a changed
  traversal algorithm. Compare link construction with search inputs held fixed.
- Intervals spanning zero do not establish equivalence. Report endpoint-specific
  findings rather than universal superiority; identify exploratory follow-ups
  and secondary endpoints without portraying them as prespecified confirmation.
- Stored-link evidence gains do not establish precomputation speedups. Keep
  measured connection timing, shared-service downstream latency, measured full
  queries, and estimated full-query time distinct.
- Keep conclusions consistent across abstract, results, discussion, limitations,
  and conclusion. Supplementary diagnostics may move to appendices regardless
  of direction; retain consequential positive and negative findings in the main
  argument. Pending outcomes remain blank and partial results are not final.

## Consistent terminology and visual comparisons

- Use the terminology mapping in `docs/PAPER_CHECKLIST.md` across prose, table
  headers, captions, diagram nodes, graph axes, legends, and speaker notes.
  Distinguish the original query from passage body text, starting passages from
  source documents, and primary Prehop from the common representation-comparison
  policy. Use descriptive conditions rather than unexplained A/B/C labels.
- Keep system names and order consistent with the registered comparison set.
  Keep representation-condition order stable across tables and graphs. Do not
  silently reorder by score or mix primary and alternative-policy results.
- Performance plots default to actual mean scores with 95% confidence intervals,
  not differences alone. Compute condition-mean intervals from saved per-question
  results; never reuse paired-difference intervals as condition-mean intervals.
  Retain paired analyses for claims about the effect of changing a component.
- Use point-and-interval plots for CIs. A boxplot represents quartiles unless
  explicitly defined otherwise; do not label quartile boxes as confidence intervals.
- Crop point-plot axes to the complete interval range with sensible padding.
  A zero origin is not mandatory. All conditions within a metric panel share
  one scale; show actual ticks and units and never clip intervals. Do not apply
  cropped-baseline point-plot conventions to bar lengths without justification.
- Report MAP and MRR on their native 0–1 scale, percentage-valued metrics in %, and
  percentage differences in percentage points. Graph labels and numeric annotations
  must match the source table after the documented rounding.
- Merge repeated system rows or shared metric headers when the result remains
  readable. Keep different timing scopes explicit. Prefer one-column figures
  when labels remain legible; do not shrink text to force a dense layout.
- Keep completed numerical evidence traceable in `docs/RESULTS.md` when a table
  is replaced by a graph. Retain pending result slots and consequential findings.

## Synchronized research deliverables

- The canonical deliverables are `docs/prehop_paper.md`, `prehop_paper.pdf`,
  `prehop_presentation.pptx`, and `prehop_presentation.pdf`. Keep one current
  presentation per format; do not leave suffixed delivery copies unless requested.
- Edit the source of a diagram or graph, regenerate its editable/vector and
  preview exports, and replace embedded presentation images. Updating a caption
  or an external image file alone does not update an embedded slide image.
- After changing claims, terminology, visuals, or numbering, update the manuscript,
  presentation, captions, internal references, and speaker notes together.
  Re-export both PDFs; verify the actual files delivered, not only source text.
- Check embedded images against current assets, exported text completeness,
  font embedding, figure/table references, clipping, overlap, blank pages, and
  final-size legibility. Check the current venue's page limit without modifying
  official style files or shrinking the prescribed type size.
- Record durable criteria in `docs/PAPER_CHECKLIST.md`; keep transient audit outputs
  outside `docs/`. Do not declare a criterion satisfied without checking the
  current artifact. These checks do not authorize benchmark reruns or restore
  automatic execution/approval gates.

## Anonymous research distribution

Before sharing an anonymous repository, inspect its distributed file list for
author names, account names, personal paths, contact details and identifying
repository links, including LICENSE and package metadata. Preserve third-party
attribution and license terms. Verify the served snapshot after publishing;
a local check does not establish that an anonymous proxy has refreshed.
