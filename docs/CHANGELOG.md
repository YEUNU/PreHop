# Changelog

## 2026-09-07

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
