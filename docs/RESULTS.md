# Result Evidence Register

This file is the canonical register for reportable full-run results. At this
revision no cell in the primary paper matrix has passed admission, so this
document intentionally contains no primary numerical result. Literature
citations elsewhere describe prior work; they are not evidence for a local
result.

Admission now also binds the current validated corpus manifest and source
bytes. A prior ledger is stale when these inputs or the verifier contract
change. This audit adds no admitted result or numerical claim.

The current Prehop controlled format-retry profile and serial benchmark
configuration introduce no admitted full result. Index reuse preserves original
index cost and requires a current content-bound link plus fresh full-query
admission; a linked one-query gate is not a reportable benchmark.

## Current profile boundary

The structured Prehop, Hippo and Youtu profiles and fixed synthetic cold
protocol require a new campaign bound to their effective configuration contract. Earlier format diagnostics and
cold canaries remain historical observations under their original code and
inputs. They cannot be transferred to these profiles or counted as full-target
admission. Background service startup, stage completion and process exit zero
also do not constitute a reportable result; the final supervisor summary must
reference all sixteen independently revalidated real full-target admissions.

## Preserved cold-canary attempt

Campaign `paper-live-l5-20260906-091016` finished its diagnostic traversal with
three cold canaries passed, eight executed failures, and five blocked targets.
LightRAG passed on both datasets and Youtu passed on MuSiQue. These are small
canaries, not full-target admissions. Later gates were not executed.

The preserved failures exposed GFM configuration composition, Linear native QA
initialization, MS dotenv environment reinjection, and a Youtu malformed
attribute observer gap. Adapter corrections require a new gate context before
fresh execution. Youtu MultiHop also returned no native chunk evidence; a
separate copy-only diagnostic reproduced malformed nested attributes and
missing native communities. Hippo's native NER returned no valid
`named_entities` JSON on both datasets. Prehop's read-only namespace check
could not reach the configured Neo4j service; dependent targets remained
blocked. None of these observations justifies repairing native responses or
changing method budgets.

## Primary matrix

The primary matrix has 16 independent targets: eight methods on MultiHop-RAG
and MuSiQue. Its order comes from `core/strategy_registry.py`, which is the
single source of truth used by the Python CLI and the shell runners.

| Strategy | MultiHop-RAG | MuSiQue | Backbone policy |
|---|---|---|---|
| Prehop | `planned` | `planned` | controlled remote generation and embedding |
| Naive RAG | `planned` | `planned` | controlled remote generation and embedding |
| MS GraphRAG | `planned` | `planned` | official pipeline with controlled remote backbone |
| LightRAG | `planned` | `planned` | official method with controlled remote backbone |
| HippoRAG2 | `planned` | `planned` | official method with controlled remote backbone |
| GFM-RAG | `planned` | `planned` | official method with checkpoint-defined local components |
| LinearRAG | `planned` | `planned` | official-faithful pinned MPNet mode |
| Youtu-GraphRAG | `planned` | `planned` | controlled native no-agent API with pinned MiniLM and NER |

The experiment ledger uses only these status values:

- `planned`: the target has not passed a canary.
- `canary_passed`: the strategy contract passed a small non-reportable canary.
- `in_progress`: the complete target is running or has a resumable checkpoint.
- `completed_unadmitted`: the complete target produced artifacts, but admission
  has not passed.
- `admitted`: the complete target passed every check below and may supply
  numerical results.
- `failed`: the target or its admission check failed.

A canary is not target completion, and target completion is not admission.
Only `admitted` artifacts may supply numbers to this register, the manuscript,
or presentation material. A successful benchmark artifact has execution
status `completed_unadmitted` until `scripts/verify_paper_target.py` writes a passing
`data/results/<run-id>/admission.json` record with status `admitted`. That
record binds the result JSON, complete detail JSONL, exact index-stats path and
bytes, canonical index-policy digest, runtime freeze and constraints,
post-query retrieval-artifact inventory and versioned effective model
configuration. Git and verifier-source hashes remain separate provenance;
only semantic/evidence contract changes affect compatibility.
Every strict skip reruns current verification. An unchanged admitted binding
keeps the original ledger bytes; a changed or invalid binding is rejected and
preserved, requiring a fresh namespace. Failed probes write separate receipts
and do not reserve the eventual successful admission ledger.

## Evaluation configuration

| Setting | Value |
|---|---|
| Remote generation model | `gemma-4-31b-it` |
| Controlled remote embedding model | `qwen3-embedding-0.6b`, 1,024 dimensions |
| Remote embedding batch/concurrency | 16 / 1 |
| Query concurrency | 1 |
| Seed | 42 |
| LLM judge | disabled |

LinearRAG instead uses pinned
`sentence-transformers/all-mpnet-base-v2` embeddings at 768 dimensions in the
primary official-faithful mode. Youtu-GraphRAG uses pinned
`sentence-transformers/all-MiniLM-L6-v2`
embeddings at 384 dimensions and its pinned NER dependency. GFM-RAG uses the
embedding and graph components identified by its validated checkpoint and
configuration files. These declared method components are not mislabeled as
the controlled remote embedding backbone.

Youtu's pinned agent batch entrypoint does not return structured per-query
answer and evidence. The registered primary target therefore uses the pinned
public native no-agent query API and is labelled `controlled_adapter`, not
`official_faithful`. Its returned evidence order, native retrieval, and native
deduplication are left unchanged.

## Admission checks

`scripts/verify_paper_target.py` and
`scripts/verify_submission_consistency.py` must establish all of the following
before a cell becomes `admitted`:

1. The full target completed with zero error rows: 2,556 ordered query rows for
   MultiHop-RAG or 2,417 for MuSiQue.
2. Detail rows have unique, gap-free indices in input order. Query IDs,
   query-record digests, ground-truth identities, eligible counts, and all
   aggregates recompute exactly from those rows.
3. Corpus manifests and index statistics bind the full source-ID set, source
   count, content digest, query-ID set, and query-record digest. MuSiQue schema
   v2 keeps paragraph IDs distinct from source filenames.
4. The completed index has exact source coverage derived from its stored
   retrieval artifacts, plus a content-addressed artifact inventory. A staged
   input directory alone is not proof of coverage. For Youtu, admission binds
   exact staged-source-to-native-chunk coverage separately from observational
   native graph reachability. Native duplicate deduplication may make the
   latter incomplete; the ledger records that method-native limitation without
   modifying retrieval or mislabelling input coverage.
5. The semantic configuration ID and hash match the checked-in per-strategy
   specification, including the upstream revision and method-defining model,
   checkpoint, schema, and retrieval settings. Operational throughput settings
   are recorded separately.
6. The generation and embedding transports match the strategy's declared
   policy. Remote calls use the single fail-closed LiteLLM gateway; pinned
   local method components match their exact revisions and dimensions.
7. Dataset metrics remain separate, and indexing cost, query service latency,
   worker-queue delay, and end-to-end latency retain their distinct meanings.
   Missing upstream token or cost telemetry is marked incomplete rather than
   estimated.

## Legacy and reserve artifacts

BrowseNet, HopRAG, and PropRAG remain supported legacy adapters, but they are
not primary matrix methods. The previously verified BrowseNet MultiHop-RAG and
MuSiQue full-split artifacts (`naacl27-clean-20260905-multihoprag-browsenet` and
`naacl27-clean-20260905-musique-browsenet`) are retained as legacy/reserve
evidence. They cannot fill, replace, or be relabelled as any primary matrix
cell, and their numerical values are therefore not reproduced in this primary
register.

## Publication synchronization

- `README.md`, this register, `docs/prehop_paper.md`, and presentation sources
  may copy numbers only from `admitted` rows in this file.
- Relative changes, uncertainty intervals, chart dimensions, and latency
  summaries must be recomputed from the admitted detail artifacts.
- A dirty tracked worktree is recorded in provenance; it is not silently
  described as clean. Semantic compatibility is decided by the recorded
  configuration and artifact identities; project commit numbers are provenance only.
- The matrix continues after independent target failures, reports every failed
  target, and exits nonzero when any target failed.
