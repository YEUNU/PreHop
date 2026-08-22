# Changelog

## 2026-08-22 — Individual question graph and evidence-directed HOP/NEXT

Q-/Q+ are now individual nodes with document-scoped embeddings rather than
concatenated chunk properties. Every Q+ searches cross-document Q-, body, and
Q+ channels; Q-/body matches are required for `HOP_ANSWER`, while Q+ same-need
similarity is provenance/support only. Rank fusion replaces the obsolete 0.82
cross-encoder-era cosine threshold. `NEXT` remains stored in document order and
is traversed bidirectionally; `HOP_ANSWER` is traversed only toward evidence.
Re-indexing a document atomically replaces its old chunk/question subgraph.
Files removed from a corpus snapshot are pruned before a new run, and Naive
now atomically replaces all chunks belonging to a changed/shortened source.
Prehop no longer creates an empty document record before its embeddings have
passed count/dimension validation.

ANN over-fetch is now sized per source (`own representations + 15`, floor 50)
instead of using the longest document for every request. Generation concurrency
is enforced once per target event loop, embeddings use loop-local semaphores for
HopRAG's threaded hooks, and the `max_num_seqs=128` capacity is validated. The
unused threshold sweep, local-model/reranker loaders, and upstream end-to-end
reranker script were removed.

The matrix capacity planner now treats identical generation/embedding URLs as
one shared `max_num_seqs` budget and separate URLs as independent budgets. It
samples both queues and automatically reduces effective client concurrency
before launch; official HopRAG's synchronous embedding hook now obeys the same
batch/request limits and validates every returned vector.
After observing Prehop throughput drop from about 7 to 2–3 documents/minute
when HopRAG joined the same generation endpoint, the matrix scheduler now caps
generation-heavy targets at one and fills spare width with Naive targets from
later datasets.

Judge hallucination is no longer inferred from an incorrect-answer score.
OpenAI Batch reconciliation requires both explicit fields for every row and
keeps the manifest/result untouched when either field is missing.

Graph traversal now begins with the same 12 seeds as depth-0 retrieval. The
old `top_k-1` combined with a graph-search cap of 10 silently removed two flat
candidates before any edge was evaluated, which made the depth comparison
invalid and caused avoidable gold-document losses.

## 2026-08-22 — Dataset-neutral Q⁺ storage

Removed the post-generation Q⁺ heuristic gate entirely. All benchmark
datasets now store the non-empty, deduplicated Q⁺ strings produced by the
same shared prompt and validated JSON schema; no domain-specific metric,
period, statement, or keyword rule affects graph construction.

## 2026-08-22 — HopRAG empty-output recovery

The full cold matrix exposed an official HopRAG question-generation response
with a valid JSON shape but an empty question list. That case now receives
three bounded retries of the same official generation step. A still-empty,
blank, or malformed paragraph result uses the upstream empty-list skip with an
explicit warning and a measured skip count; if every paragraph is skipped,
the resulting empty document still fails the target. No alternative indexing
or retrieval method is substituted.

What was tried, what changed, and why — kept separate from `CLAUDE.md` so
that file can stay pure current-state reference. Newest first. Entries
reconstructed from earlier (compacted) session history are ordered as named
milestones rather than forced into exact dates; entries from directly
observed recent work carry dates.

## End-to-end consistency and fail-loud completion (2026-08-22)

Completed the repository-wide audit beyond Prehop's mixins. Missing dataset
or query inputs now raise and exit non-zero; the MultiHop-RAG wrapper now
points at the real sample200 file; the deleted report-tool invocation and
unneeded reranker/Neo4j server startups were removed. Indexing restores a
failed Neo4j batch for idempotent replay, persists its failure artifact, and
then exits non-zero on any file/flush/HOP failure. HopRAG no longer substitutes
plain vector retrieval when official traversal fails, and shared embedding /
reranker paths reject missing vectors or malformed score responses instead of
producing zero scores. The same validation now covers Prehop's sparse Q-/Q+
embedding batches, HopRAG's required JSON keys, and the optional MS GraphRAG
dense-retrieval helper.

The unused MS GraphRAG dense-retrieval helper was subsequently removed: the
benchmark adapter now exposes only the official local/global GraphRAG search
APIs, preventing its old `top_k=5` auxiliary default from being mistaken for
an experimental retrieval setting.

LLM JSON handling now distinguishes transport failures from parse failures and
validates required keys/types at every Prehop call site. Prehop, Naive, and
HopRAG share one synthesis prompt and one empty-context abstain policy; MS
GraphRAG retains synthesis inside its official API. The configured judge model
is now fixed for a run (no per-row fallback model), partial Batch API outputs
retain their reconciliation manifest, benchmark aggregates use the union of
numeric fields across rows, and any per-query runtime error marks the artifact
`completed_with_errors` and makes the command exit non-zero. `benchmark_all`
still attempts every strategy before reporting their combined failure, and the
main result JSON remains slim while full traces stay in the dedicated report
artifacts. Optional debug output now respects `--save-intermediate`, and
source-sweep smoke tests exclude the local virtualenv/third-party/data trees.
Dataset wrappers validate query files only for stages that actually benchmark,
so an index-only run is not coupled to preparation of an unused query sample.

Run diagnostics are now isolated by `RAG_RUN_ID`: intermediate JSON uses
`data/debug/<run>/<strategy>/<corpus>/<source>/`, index logs use
`logs/index/<run>/<strategy>.log`, and failure/stat filenames carry the same
run ID. `--clear-graph` executes once rather than racing in every parallel
strategy or wiping a strategy indexed earlier by a dataset wrapper. Removed
the old shared-layout debug files, logs, obsolete source-less chunk cache,
Python/test/linter caches, stale exception log, and generated egg metadata
(about 309 MB total); the removal was moved to the desktop trash for recovery.
Also deleted unreferenced remnants of the previous agent/OCR architecture
(`core/schemas.py`, Prehop schemas/trace helpers, tool definitions), the old
HypoReflect migration cleanup script, and the duplicate dedicated MultiHop
sampler now covered by `data/make_sample.py`.

## Full-corpus exhaustive scan (not sampling) after a deep content audit (2026-08-22)

User asked for a deeper audit than the pattern-regression monitor — actually
reading real Q-/Q+/chunk content, not just counting known-pattern matches
over a random sample. Two outcomes:

- **Q-/Q+ generation quality is genuinely good** across all three datasets:
  Q⁻ consistently self-contained/answerable from its own chunk, Q⁺
  consistently points at real information the chunk doesn't contain (not
  restating the chunk) — read ~20 real samples by hand across
  multihoprag/hotpotqa/musique to confirm this, not just trusted the
  pipeline's own design intent.
- **A full-corpus frequency scan (not sampling) found more residual
  HotpotQA/MuSiQue markup** the earlier wiki-markup fix's narrow
  `<nowiki>`/`<br>`/`[[...]]`/`{{...}}` patterns missed: ruby-annotation
  tags (`<ruby>`/`<rb>`/`<rt>`/`<rp>`, Chinese/Japanese pronunciation
  guides), `<ref>`, `<a href="...">`, `<onlyinclude>`, `<section begin=...
  />`, `<small>` (293 occurrences across 90 hotpotqa files) — generalized
  `clean_wiki_markup()`'s tag-handling from an enumerated list to a single
  catch-all `<[^>]+>` strip (tag removed, inner/surrounding text kept;
  `<br>` still special-cased to a space so words don't get joined). Also
  added stripping of bare citation markers (`[12]`) and stray `[[`/`]]`
  left over from malformed/mismatched-bracket wikilinks the primary
  wikilink regex doesn't match. Applied to both `prepare_hotpotqa.py` and
  `prepare_musique.py` identically. musique_corpus was already fully clean
  even before this fix (0 occurrences) — applied there anyway since it
  shares the same Wikipedia source and could hit the same issue on a
  different data slice later.
- One thing investigated and *not* fixed: ~30 hotpotqa paragraphs are
  genuine Wikipedia disambiguation-page stubs ("X may refer to: ..." with
  no real facts) — this is the source page's actual content, not
  corruption, and is exactly the kind of low-information distractor
  paragraph HotpotQA's own distractor-config design already expects.
- The ongoing 15-minute background quality monitor was upgraded from
  `ORDER BY rand() LIMIT 30` sampling to a full-population scan (`MATCH
  (c:...) RETURN count(...)` with no LIMIT) across all three corpus tags,
  per the same "sampling misses things" lesson — a 70-count full-population
  hit on multihoprag right after this change was manually verified as a
  false positive (live-blog-format articles like "MLB Winter Meetings
  tracker" genuinely cite "Source: X" inline per update, as normal
  journalism convention — not a scraper artifact, left alone).

## MultiHop-RAG scraper boilerplate in article bodies, found live (2026-08-22)

Third data-quality bug caught by the same periodic Neo4j quality sampler,
right after the two below — ~9% of MultiHop-RAG articles (54/609) had
newsletter-subscription UI text scraped into the article `body` field
itself in the source `corpus.json`, predating any of this project's own
code. Two structurally distinct patterns by outlet:
- Independent-sourced articles: a Mustache-template signup widget
  (`{{ #verifyErrors }} ... {{ /verifyErrors }} ... {{ /verifyErrors }}`,
  closing tag always exactly twice) at the very start of body, preceded by
  section-specific opening copy that varies too much to pattern-match
  directly ("Sign up to Simon Calder's...", "Stay ahead of the trend...",
  etc. depending on section) — so the fix anchors on the position (start of
  body) and the two `{{ /verifyErrors }}` closing tags instead, not the
  opening phrase.
- Guardian-sourced articles: text between `skip past newsletter promotion`
  and `after newsletter promotion` (an accessibility skip-link pair
  wrapping an embedded newsletter widget) — can appear anywhere in the
  body, stripped wherever it occurs.
`_strip_scraper_boilerplate()` added to `prepare_multihoprag.py`, applied to
`article.get("body")` only (the separate `evidence_facts` field in
`MultiHopRAG.json` comes from a different, already-clean `fact` field, not
the scraped body). Caught and fixed twice in quick succession — first
attempt anchored on the opening phrase and missed the Lifestyle-section
variant, which starts differently; broadened to anchor on the fixed closing
markers instead. Verified against the full corpus after each attempt
(byte-size distribution, zero remaining marker occurrences) before
resuming indexing.

## Two more corpus-prep leak bugs, found live during the full reindex (2026-08-22)

Caught during the from-scratch full-dataset reindex by a periodic quality
monitor that samples real indexed chunks from Neo4j (not just watching
progress/error counts) — both stopped and fixed within ~1 minute of the
run starting, minimal cost sunk.

- **HTML entities in HotpotQA/MuSiQue titles**: `prepare_hotpotqa.py`/
  `prepare_musique.py` never called `html.unescape` on article titles
  (`prepare_multihoprag.py` already did), so entity-named articles' titles
  and filenames carried literal `&quot;`/`&amp;`/etc. (e.g.
  `&quot;J&quot; Is for Judgment`). Body text itself was already clean.
  Gold `evidence_docs` happened to use the same escaped form, so automated
  title-matching wasn't silently broken — but citations would have read the
  escaped form forever. Folded `html.unescape` into both scripts'
  `clean_wiki_markup()` and applied it to titles too (previously only body
  text/evidence_facts went through it).
- **`Category:` header leaking into MultiHop-RAG's first chunk**: the
  earlier "News-metadata injection removed" pass (see below) trimmed the
  corpus-file header from 4 lines to 2 (`Title:`/`Category:`), but the
  chunker's header-skip logic only ever strips exactly one line (`Title:`)
  — so `Category: <cat>` became parts of `sent_id=0`'s body text for every
  single document, undetected because this 2-line header format had never
  actually gone through a real indexing run until this reindex. Checked
  whether `category` is read anywhere downstream first (it isn't — the
  benchmark's own `category`/`question_type` comes from `queries.json`, a
  completely different field) and removed the header line entirely rather
  than teaching the chunker about a second header line, so MultiHop-RAG's
  corpus header format now matches HotpotQA/MuSiQue's (`Title:` only).

## Baseline top-k provenance and alignment (2026-08-22)

Audited the unequal synthesis-context counts across strategies against their
upstream implementations. HopRAG now keeps the official repository's
end-to-end `--topk 20` setting (the local adapter previously returned 10 after
reranking). Naive RAG has no upstream fixed value, so its prior local default
of 5 was removed and it now shares Prehop's domain-aware
`RAGConfig.DEFAULT_TOP_K` (news=12, financial=8). MS GraphRAG was left alone:
the benchmark uses its official `local_search` / `global_search` APIs and does
not call the adapter's separate dense `retrieve(top_k=5)` helper. This is a
benchmark-only change and requires no reindexing.

## Fail-loud sweep across indexing/retrieval (2026-08-22)

Removed 9 "keep-system-alive" fallbacks from `models/prehop/indexing/` and
`models/prehop/retrieval/` — places where a real failure (bad JSON from the
LLM, an empty query embedding, an empty candidate pool) was being silently
absorbed into a default/empty value instead of surfacing. Explicitly scoped
to prehop's own indexing/retrieval code, not `core/vllm_client.py` (shared
by every strategy — its own retry-then-degrade behavior stays) and not the
baseline adapters.

- `graph_writer.py`: found and fixed a real correctness bug the fallback was
  masking, not just a style issue. `embedding` (the primary/body field) was
  silently substituted with `q_minus_embedding` whenever Q⁻ existed. Since
  `body_vector_index` is built on the `embedding` property, this meant the
  "body" search channel was actually searching Q⁻ content for any chunk
  with a non-empty Q⁻. Fixed: `embedding` is now always the true body
  embedding; missing body embedding raises instead of falling back.
- `knowledge_mapping.py`, `traversal.py`, `rewrite.py`, `rerank.py`: each
  had a `generate_json(...)` call whose failure (after `core/vllm_client.py`
  exhausts its own retries and returns `{}`) was silently treated as "empty
  result" (empty Q-/Q+, "INSUFFICIENT" continuation decision, no query
  rewrites, unsimplified rerank query) rather than raising. All four now
  raise, so a genuine failure shows up as a failed chunk/query
  (`data/index_failures/` at indexing time, a 0-score `runtime_error` result
  at query time — `cli/benchmark.py` already isolates failures per query) instead
  of silently degrading quality with no signal.
- `hop_edges.py`: `_find_hop_candidates` fell back to querying the Q⁻
  vector index (with a Q⁺ embedding as the query vector — a semantically
  odd cross-channel search) whenever the primary Q⁺↔Q⁺ ANN search came back
  empty. Undocumented anywhere in the module's own docstring. Removed —
  empty candidates now just mean no HOP edge for that chunk.
- `hybrid.py`: an empty query embedding was silently treated as "this
  channel has no candidates" rather than raising.
- `retrieve.py`: Stage 2 (Q⁺ expansion) falling back to Stage 1's
  *un-thresholded* reranked candidates when Stage 2 rerank produced nothing
  above `RERANKER_THRESHOLD` — silently returning content that had already
  failed the quality gate. Removed; the query now correctly falls through to
  "Insufficient evidence" instead.
- One test (`test_retrieve_prefers_company_matched_candidate`) turned out to
  depend on this exact failure mode: its query is >80 chars, which triggers
  `_simplified_rerank_query`'s LLM call, and the test never mocked
  `generate_json` — it was silently relying on the network call failing and
  falling back to the original query. Fixed by mocking `generate_json`
  properly.

## Shared-module refactor (2026-08-22)

Immediately followed the fail-loud sweep, which had mechanically introduced
the same 3-line "call generate_json, raise if empty" pattern at 4 call
sites — a clear signal a shared helper was overdue.

- Deleted `rerank.py::hybrid_search` — confirmed zero callers anywhere in
  the repo, fully superseded by `retrieve.py`'s `RetrieveMixin.retrieve()`.
- Added `TextUtilsMixin._rrf_accumulate` (`text_utils.py`) unifying the two
  independent reciprocal-rank-fusion implementations in `hybrid.py`
  (`update_rrf`) and `retrieve.py` (`_accumulate`).
- Deleted `traversal.py`'s local `_rerank_and_gate` closure and pointed both
  of its call sites at `rerank.py`'s existing `_rerank_and_select` instead.
  This fixed a real, previously undocumented inconsistency as a side effect:
  `_rerank_and_select` ran the query through `_simplified_rerank_query`
  (strips verbose/role-played preludes — empirically verified elsewhere to
  collapse a rerank score from 0.94 to 0.03) before scoring;
  `_rerank_and_gate` did not, scoring `graph_search`'s often-long synthetic
  query raw. Decided to unify on always simplifying.
- Added `models/prehop/llm_json.py` (`generate_json_or_raise`) — the shared
  fail-loud wrapper the 4 call sites above now use, instead of each
  duplicating the same raise block.
- Hard constraint respected throughout: no changes to `models/naive/`,
  `models/hoprag/`, `models/ms_graphrag/`, or `core/`/`cli/`.

## Dead-field cleanup

- `Document.summary` (a document-level LLM-generated summary, written at
  indexing time) removed after an actual quality A/B experiment — not a
  cost argument. Same retrieval, two synthesis-context conditions (with vs.
  without the summary injected), same judge. Result: no measurable benefit,
  plus one concrete case where the summary actively misled the synthesis
  answer. Removed `graph_writer.py::summarize_document`, `cli/index.py`'s
  summarization pass, and the now-unused `GLOBAL_SUMMARY_PROMPT`/
  `GLOBAL_SUMMARY_FORMAT_INSTRUCTION` prompts. (Distinct from `chunk_summary` —
  the per-chunk summary produced alongside Q-/Q+ — which is still generated
  and still feeds the body fulltext index; not touched.)
- Following the same "is this actually read anywhere" question generally
  (not just for the summary): `body_embedding` (computed but never queried —
  the primary `embedding` field already serves body search) and `corpus`
  (redundant with the Neo4j label itself, which already namespaces every
  query by corpus tag) were both confirmed dead by direct code inspection
  and removed from `graph_writer.py`'s Chunk node writes. Two more
  never-referenced dict keys (`q_plus`, `q_plus_embed`) were found in the
  same batch-data payload and dropped.
- The identical pattern was found in `models/naive/naive_rag.py` — a
  completely separate indexing implementation from prehop's own
  `graph_writer.py` — which independently wrote both a `corpus` and a
  `branch` property on its Chunk nodes. Same root cause: `self.chunk_label`
  (built from `corpus_tag` + `ablation_profile`) already namespaces every
  `MATCH`/`MERGE` via the node label, so both properties were pure
  unread redundancy. Removed there too.

## Data-quality bugs found via code-intent audit

A background subagent audited real indexed Neo4j data against what the
source code's own docstrings/comments claimed it should look like, across
all three in-progress corpora.

- MultiHop-RAG: 665/8,442 chunks carried raw source-file header boilerplate
  (`Title:`/`Source:`/`Category:`/`Published:` lines) leaking into chunk
  text because `prepare_multihoprag.py`'s article header format didn't match
  what the chunker expected to strip. Fixed at the source (`prepare_multihoprag.py`),
  corpus regenerated, affected corpus re-indexed.
- HotpotQA (caught by the audit) and MuSiQue (fixed proactively, same
  pattern, before the audit reached it): unstripped MediaWiki markup
  (`<nowiki>`, `<br>`, `[[wikilink|display]]`, `{{template}}`) leaking into
  both corpus body text and `evidence_facts`. Added a shared
  `clean_wiki_markup()` function (duplicated per prep script, matching the
  established per-script-duplication convention in `data/prepare_*.py`) and
  applied it at both corpus-build and query-build time.

## Full-dataset regeneration

`prepare_hotpotqa.py`/`prepare_musique.py` default to `--limit 2000`
(questions, not documents) — an early full-corpus indexing run turned out to
have been built from that default subset, not the true full datasets.
Regenerated with `--limit 0` (true full: HotpotQA 7,405 / MuSiQue 2,417
questions) for a real from-scratch reindex.

## News-metadata injection and comparison-query decomposition removed entirely

Two MultiHop-RAG-specific engineering features — injecting `Source:`/
`Published:` metadata lines into the synthesis context, and decomposing "A
vs B" comparison queries into per-entity retrieval variants — were removed
entirely from the code (not just disabled behind a flag), after deciding
they sat outside the paper's three headline claims and broke the
dataset-general/domain-agnostic parity the benchmark suite is built around.
`_build_context_from_nodes` now always uses the single `[[title, Page N,
Chunk N]]` format; `QUERY_REWRITE_PROMPT` always does plain paraphrase
rewriting.

## Reranker replaced: cross-encoder → embedding cosine-similarity

An earlier iteration used a dedicated reranker model
(`Qwen3-Reranker-0.6B`, served separately) for two purposes: query-time
retrieval reranking, and — critically — scoring candidate HOP edges at
indexing time. That model isn't part of the current inference setup, so
both uses were replaced with embedding cosine-similarity: HOP-edge scoring
now reuses the score Neo4j's own `db.index.vector.queryNodes` ANN query
already returns (no extra model call at all), and query-time reranking uses
a shared `_embedding_rerank_scores(query, texts)` helper. Both
`HOP_THRESHOLD`/`RERANKER_THRESHOLD` were originally calibrated for
cross-encoder classifier scores (roughly a 0-1 probability) and now gate raw
bi-encoder cosine similarity instead — flagged as likely needing empirical
re-tuning once real benchmark numbers exist; not yet done.

Separately, the Q⁺ quality gate (`_is_high_quality_q_plus`, requires an
entity/period/metric/source-anchor signal) was originally a strict 4-of-4
AND — this collapsed acceptance to ~2.8% of generated Q⁺ (~1,640/47k
chunks), leaving the HOP graph effectively empty. Relaxed to "at least 2 of
4" after observing that bridge questions about the same metric across
periods, or a related metric in the same period, naturally drop one signal.

## Model infra migrated to LiteLLM proxy

Generation and embeddings now route through a LiteLLM proxy
(OpenAI-compatible, internal network) rather than directly-served local
vLLM models. Verified working: `gemma-4-31b-it` (chat) and
`qwen3-embedding-4b` (embeddings, dim=2560) — a dimension change from
whatever was configured before, requiring `NEO4J_VECTOR_DIMENSIONS` to be
updated and the vector indexes rebuilt from scratch (dimension is fixed at
index-creation time).

Live smoke test (2-doc toy corpus, before any real-scale benchmark existed)
showed retrieval — not traversal — dominating total query latency (LiteLLM
proxy round-trips for query rewrite + per-channel hybrid RRF embedding
calls). Useful context before assuming the graph-traversal step is a
latency risk; not re-verified at real scale yet.

## hoprag/ms_graphrag baseline dependency fixes

Neither baseline had been exercised even once before this pass. hoprag
crashed on a missing `pandas` import — initially misdiagnosed as needing
`paddlenlp`/`modelscope` too, but those are already stub-injected via
`sys.modules` before the vendored `third_party/HopRAG/tool.py` imports them,
so only `pandas` was a genuine gap. Both baselines' endpoint/auth defaults
were hardcoded to dead local ports and a hardcoded `"EMPTY"` API key from an
earlier local-serving setup; fixed to fall back through the same
`VLLM_URL`/`VLLM_EMBED_URL`/`VLLM_API_KEY` env vars prehop's own pipeline
uses. ms_graphrag additionally had a hardcoded `vector_size=1024`, stale
from before the embedding-model switch — now reads
`NEO4J_VECTOR_DIMENSIONS`. A known remaining limitation: hoprag's official
(unmodified, verified byte-identical to upstream) edge-construction code can
throw `KeyError` on an imbalanced pending/answerable question-category
cross-join at small corpus scale — not yet re-verified at real scale.

## Full rename: prehypo → prehop

Package directory, strategy id, Neo4j label prefix, docker container name,
and loggers all renamed from the project's original working name
(`prehypo`) to `prehop` — `hypohop` is the paper's title only, not used
anywhere in code/config.

## Original scope narrowing (pre-dates this session)

An earlier submission under a different name/scope was rejected, with three
pieces of criticism: too many bundled components obscuring the real
contribution, a directional Q-/Q+ weighting claim that wasn't statistically
significant, and low absolute performance. The project was deliberately
narrowed to the three core claims `CLAUDE.md`'s "Project identity" section
now states. Directional Q-/Q+ weighting stays explicitly out of the
headline claim as a direct result of this. (Target venue/deadline/cycle
specifics are intentionally not named here — see `SUBMISSION_TARGET.md`,
local-only and gitignored.)
