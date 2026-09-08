# Changelog

## 2026-09-08 — Batch LinearRAG queries and repair result verification

The adapter now coalesces concurrent LinearRAG questions into the original QA
batch API, recording batch size without changing upstream code or parameters.
Fixed aggregate-variable shadowing, compact-detail verification with separate
traces, and serving-alias versus observed-checkpoint revision validation. Naive
and Prehop retained results passed verification without rewriting predictions.

## 2026-09-08 — Clarify retrieval branches and component attribution

Documented original-text search across enabled channels when rewriting is
skipped, and retention of rewrite/refinement policies at depth zero. Added a
dependency-edge availability contrast that keeps NEXT enabled, with explicit limits
on channel, fixed-candidate and bootstrap interpretations. These are method
and evaluation-contract clarifications, not new experimental findings. Clarified
ANN/source-file scope, inherited rank evidence and additional body-only
candidate handling; attributed the question-role formulation to HopRAG.

## 2026-09-08 — Align documentation with native outcomes

Reconciled the manuscript and runtime guide with native response observation,
terminal-query failure scoring and resume retention. Distinguished version-1
and version-2 index reuse and historical submission handoffs. Reconstructed the
260-query diagnostic comparison from stable IDs and verified its primary scores
against gold data. Documented the independent aggregate and JSONL admission
blockers; no publication admission or executable repair is claimed.

## 2026-09-08 — Benchmark completed indexes first

Added version-2 index reuse links for verified full-index supervisor receipts.
Original index costs and artifact inventories remain bound; file-backed query
workspaces are separate copies. This supports benchmarking completed datasets
without rebuilding indexes or claiming a canary gate that was not run.


## 2026-09-07 — Isolate failures without dropping queries

Terminal query failures now remain in quality denominators and paired bootstrap
as zero-score observations, with failure rates and measured latency retained.
Resume preserves failed rows. Source integrity failures still block a target.
Index supervisors emit execution outcome tables and continue independent targets;
failed indexes have unavailable quality rather than fabricated scores.


## 2026-09-07 — Preserve native fallback behavior

External adapters now observe native outputs without extraction repair,
additional format retries or stronger content-quality failure gates. GFM's
native empty-entity fallback is retained. Hippo entity normalization, MS glean
repair and Youtu forced schemas are no longer used by production adapters.
Audit hashes and source identity checks remain enforced; native failures remain
failures. All six external strategies require fresh indexes under the new policy.


## 2026-09-07

- Aligned method documentation and diagrams with role-view body retrieval,
  per-view embeddings, iterative refinement and candidate-count limits. Moved
  the primary MS GraphRAG indexer out of the legacy section and replaced the
  stale HippoRAG2 format profile. Distinguished development-inspected evaluation
  from confirmatory evidence and marked the vendored walkthrough as reference.

- Fixed MS GraphRAG adapter rejection of native glean responses containing a
  prose preamble followed by valid tuples. The adapter removes only a leading
  unstructured preamble, preserves every tuple verbatim, and still rejects
  incomplete or ambiguous structured output. Raw and normalized responses
  remain auditable. Profile `strict-ms-native-glean-v4` distinguishes this
  normalization; upstream code, prompts, glean count and generation parameters
  are unchanged. This is an adapter formatting intervention, not a claim of
  byte-identical upstream parsing.

- Audited pinned native parameters across the primary methods. Removed MS's
  adapter-added output caps and temperature (native call arguments omit them),
  restored HippoRAG2 retrieval candidates 200 with QA context 5, and restored
  LightRAG top-k 40. Connected GFM's native single-pass QA prompt/generator and
  Youtu's original agent loop. Truncated MS outputs fail without identical
  retries. The temporary 4,096-token proposal was superseded after native-default
  review. Changed policies require fresh compatible evidence.

- Hardened pending-model adapters at extraction and retrieval boundaries.
  HippoRAG2/GFM-RAG validate structured NER/triple responses before native parsing;
  MS GraphRAG checks native tuple/report formats and glean delimiters; Youtu
  records bounded validation retries; LinearRAG validates stores and QA completion.
  Exhausted extraction cannot become a successful index. Index-time audit prefixes
  are content-bound and checked during publication admission. External source
  files are unchanged; registered adapter identities distinguish the new behavior.
  MS query streaming retains native chunks and validates termination. Publication
  checks recompute primary row metrics from saved predictions and authoritative
  gold evidence, and reject non-finite or out-of-range primary averages.

- Extended the HippoRAG2 entity adapter to accept either explicit name field
  (`entity` or `text`) with optional `type` and/or `label` annotations. Versioned
  the normalization profile to v2 and index identity to v4; ambiguous names
  and unsupported fields still fail closed. Replayed 608 cached responses:
  all 214 dictionary-key failures recovered; 394 other results were unchanged.

- Added a HippoRAG2 adapter for explicitly named entity objects that otherwise
  produce `unhashable type: 'dict'` and empty native NER results. Preserved raw
  responses, recorded per-chunk recovery/rejection, and versioned the index
  identity. External source files and generation settings remain unchanged.

- Updated the remote embedding contract to Qwen3-Embedding-4B with 2,560
  dimensions after a successful gateway probe. Updated local settings, adapter
  defaults, documentation and contract tests. Superseded 0.6B results and their
  Neo4j indexes were removed; current traces and migration evidence were retained.
  Historical serving observations remain separate from current run evidence.
  All 16 method/dataset runtime and configuration checks passed; full index
  construction is a separate validation step.
- Added execution profile v3 with a single 120-request pool shared by generation
  and embedding. Client ceilings follow the shared bound; aggregate queue
  metrics retain the combined peak alongside per-kind observations. Previous
  profile formats remain readable without changing retained run identities.
- Documented stalled-child diagnosis separately from supervisor liveness and
  clarified local artifact retention. Explicitly excluded prepared runtime
  environments from Git alongside trace payloads and generated results.

- Fixed a Prehop trace deadlock under concurrent embedding load. HTTP trace
  writes no longer wait for the default executor used by embedding permit
  acquisition. Added a 60-request regression that fails on the previous code.
  Related tests passed (21); a traced 64-document live pilot completed with
  zero HTTP errors and a peak of 60 concurrent generation requests. This is
  engineering validation, not a publication result.

- Enabled full Prehop tracing for indexing and benchmarking: stage payloads,
  raw HTTP requests/responses, native validation outcomes and retry attempts,
  source/query identities, graph operations and final outputs. Added a local
  hash-verifying inspector and enabled intermediate chunk files by default.
  Prehop source now contains observation hooks; external native sources remain
  unchanged. Trace I/O is part of measured execution where it occurs.

- Added a shared bounded inference queue and content-bound execution profiles.
- Separated amortized indexing/query wall cost from response latency, and
  excluded incomplete or resumed query batches from throughput claims.
- Connected adapter document producers to the execution profile: Prehop's
  bounded file window and ordered chunk lookahead, native LightRAG insertion workers and native Youtu
  construction workers. Synchronized Youtu schema updates in an adapter subclass;
  pinned upstream source files remain unchanged. Youtu extraction rejects invalid
  raw JSON and retries unchanged requests within a shared total attempt budget.
- Added isolated runtime validation, producer pilots and a detached index-only
  matrix runner for eight primary methods on both complete datasets.
- Fixed LightRAG adapter shutdown under the pinned Python 3.10 runtime;
  cancellation excludes the timeout supervisor and joins native workers.
- Clarified document roles, separated indexing-only execution from paper
  admission, and consolidated profile and cost definitions in the throughput
  guide. Removed stale setup narratives and live campaign details from design
  documentation.
- Consolidated the completed adapter parallelism plan into the architecture
  and throughput guides, then removed the redundant plan document.
- Retired outdated result narratives and superseded generated artifacts. The
  current evidence register is RESULTS.md; historical numbers are not retained
  as current publication evidence.

Current runtime, structured-output, native-component and admission contracts
are documented in ARCHITECTURE.md and RUNTIME_REQUIREMENTS.md.
