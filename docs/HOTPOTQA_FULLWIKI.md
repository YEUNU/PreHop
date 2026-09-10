# HotpotQA fullwiki preparation and evaluation

Use the official 7,405-question fullwiki development split with the independent
2017-10-01 Wikipedia introductory-paragraph corpus. The ten contexts attached
to each development question are not the retrieval corpus. The fullwiki test
answers are not public, so this repository reports development-set results.

The local preparation passed full integrity verification on 2026-09-10:
5,233,329 paragraphs and all 7,405 development queries. The audit compared every
archive record with the sentence store and checked every prepared file digest.
All seven primary methods passed runtime and public-entrypoint readiness checks;
331 related integration tests passed. Local receipts are
`data/hotpotqa_raw/readiness.json` and `data/hotpotqa_raw/runtime-readiness.json`.
These receipts describe the source and runtime snapshots checked at the time
of verification. Rerun the applicable checks after relevant changes; do not
relabel an old receipt with new source hashes. They establish preparation and
execution readiness, not completed fullwiki indexes or benchmark results. The
prepared corpus identity is listed in [RESULTS](RESULTS.md#prepared-dataset-identities).

## Prepare data

From the repository root, use the prepared main Python environment:

```bash
"$PYTHON_BIN" scripts/datasets/download_hotpotqa.py
"$PYTHON_BIN" scripts/datasets/prepare_hotpotqa.py \
  --archive data/hotpotqa_raw/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
```

The downloader verifies the official archive's 1,553,565,403-byte size and MD5
`01edf64cd120ecc03a2745352779514c`. See the
[official dataset site](https://hotpotqa.github.io/) and
[Wikipedia release instructions](https://hotpotqa.github.io/wiki-readme.html).
If the original development endpoint is unavailable, it uses the
[author-owned fullwiki mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
at revision `1908d6afbbead072334abe2965f91bd2709910ab`. The download provenance
records the origin, revision and hashes; the prepared manifest binds content
and exact query identities. Retain these records alongside results.

Preparation streams the archive without extracting archive paths. Each wiki
paragraph becomes one `hotpotqa_<wiki-id>.txt` with its title and original plain
sentences. `sentences.sqlite3` retains exact title/sentence-index identities.
The release selects the first paragraph longer than 50 characters after removing
hyperlinks; preparation consumes the released plain `text` field directly.
The builder records the coverage of every development supporting fact against
this independent corpus and never inserts gold context to fill a gap. It publishes
`corpus_manifest.json` and `data/hotpotqa_queries.json` only after validation.
An interrupted directory is incomplete; the builder refuses to overwrite it.
If conversion finished but metadata was not published, `--finalize-existing`
rechecks every archive record against the existing paragraph store before
publishing metadata. It does not rewrite paragraph files, and rejects missing,
extra or changed store records.
After preparation, run the same command with `--verify-only` to audit all
prepared file digests, the sentence store, exact queries and download provenance.
This read-only audit reports `status: ready` only after every check passes.
The corpus contains millions of files, so provision disk space and inodes.
Raw and prepared data are ignored by Git.

### Upstream annotation anomaly

The released dev split includes one unavailable sentence label:
`5ae61bfd5542992663a4f261` contains `("Jimmy Butler (basketball)", 902)`, while
both the released corpus paragraph and dev context contain five sentences.
The same label is reported in [upstream issue 47](https://github.com/hotpotqa/hotpot/issues/47).
The prepared manifest records 18,004 addressable entries out of 18,005 supporting
fact entries and one affected query. All 7,405 queries and every original gold
label remain unchanged. Official scoring retains the unavailable label in its
denominator; no query is excluded and no replacement label is inferred.
Annotation coverage is separate from complete archive/file integrity.

## Run a paper target

Use the shared queue and fresh target commands in
[THROUGHPUT_EXECUTION](THROUGHPUT_EXECUTION.md), selecting dataset `hotpotqa`.
The supported datasets are MultiHopRAG and HotpotQA. Preparing fullwiki does
not start indexing automatically.

The general dataset wrapper also accepts:

```bash
./run_dataset.sh hotpotqa all --model prehop --queries full
```

Use the prepared runtime required by each model. Fullwiki uses the whole corpus
and all 7,405 development queries; sampled smoke checks are not paper results.
Original baseline implementations are unchanged by dataset preparation and
evaluation adapters. Whole-corpus graph construction can still exceed the
resources required by smaller datasets; a passing fixture is not a completed
full-corpus experiment.

## Metrics and supporting-fact outputs

Report the official answer, supporting-fact and joint EM, precision, recall and
F1: twelve metrics in total. The implementation is checked against the
[official evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py),
including its yes/no/noanswer rules. Failed queries remain in quality-score
denominators and are counted separately. Report mean query latency separately
from elapsed benchmark time divided by query count.

Models return passages rather than a common native sentence-label output.
The evaluation adapter therefore selects every complete original sentence
present in a retrieved passage after whitespace normalization and exports its
exact `(title, sentence index)` pair. This common deterministic projection uses
no gold facts to choose sentences. Supporting-fact and joint scores describe
this projection; do not present them as native model sentence predictions.
Unresolved passages contribute no supporting facts. Retain the policy field,
retrieved passages and sentence-store hash with results.

Export a completed full result for the official evaluator:

```bash
"$PYTHON_BIN" scripts/export_hotpotqa_predictions.py /absolute/path/result.json \
  --output /absolute/path/new-predictions.json
```

The exporter checks exact prediction/gold query IDs and writes the official
`answer`/`sp` dictionaries. It maps failed queries to empty answers and facts.
It does not truncate model answers before scoring.

Index cost uses recorded successful indexing wall time, document count, usage
and request traces. Report total time and time per original wiki paragraph;
record concurrency alongside cost. Do not equate amortized throughput with an
individual request's latency or silently omit retries and failed attempts.
