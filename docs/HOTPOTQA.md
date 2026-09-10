# HotpotQA: HippoRAG release preparation and evaluation

The active paper setting is the original HippoRAG v1 pooled retrieval corpus:
**9,221 passages and 1,000 released query rows**. It is not the official fullwiki
search setting. All methods and ablations use the same released corpus and rows.

## Source and population

The source is [HippoRAG v1.0.0](https://github.com/OSU-NLP-Group/HippoRAG/tree/b144c46df14cabe5f5822d8caded4bec5f709461),
revision `b144c46df14cabe5f5822d8caded4bec5f709461`, specifically
`data/hotpotqa.json` and `data/hotpotqa_corpus.json`. The
[paper, Section 3.1](https://arxiv.org/html/2405.14831v1#S3.SS1) describes pooling
supporting and distractor contexts from selected validation questions. Corpus
membership is therefore benchmark-derived; it does not model searching all of
Wikipedia. Retrieval sees the common pool, not the per-question gold contexts.

The 1,000 released rows contain **944 unique original question IDs**. Preserve
all rows in their released order. Prepared `_id` combines the original ID and
row ordinal; `original_query_id` retains the upstream identity. These occurrence
IDs prevent trace/result overwrites without discarding or inventing questions.
Report row-weighted scores for release comparability and original-question
macro scores as a sensitivity analysis. Confidence intervals resample original
question clusters. This sampling convention is separate from LLM generation
seeds.

## Preparation

```bash
python scripts/datasets/prepare_hotpotqa_hipporag.py --download
```

The script downloads the two pinned files under `data/hotpotqa_hipporag_raw/`.
It writes paragraph text and `sentences.sqlite3` under `data/hotpotqa_corpus/`,
queries to `data/hotpotqa_queries.json`, and provenance/population metadata to
`corpus_manifest.json`. Use explicit `--output` and `--queries-output` paths for
an additional prepared copy. Use fresh destinations for preparation.

Titles, sentence order and original sentence text are preserved. Source IDs
are deterministic title-derived IDs. The corpus comes directly from the
released dictionary; gold labels are not used to repair corpus content.
Missing supporting sentences are recorded in `annotation_coverage` and remain
in the original evaluation denominator. The protocol identity is
`hotpotqa-hipporag-v1-1000`; completed evaluation of all released rows is marked
`released_benchmark`, not an official fullwiki completion.

## Evaluation

Use the official HotpotQA Answer, Supporting Fact and Joint EM/F1 rules through
`utils/hotpotqa.py`. The common adapter predicts all complete original corpus
sentences present in returned passage text, mapped to title and sentence index.
It never uses gold labels to select predicted support. These are official
scoring rules on the declared reduced-corpus protocol, not a leaderboard score.

```bash
./run_dataset.sh hotpotqa all --model prehop --queries full
```

Here `full` means all **1,000 rows of the active prepared release**. Apply the
same inputs to every registered comparison method. See
[PAPER_ABLATION_DESIGN](PAPER_ABLATION_DESIGN.md) for paired contrasts and
[RESULTS](RESULTS.md) for completed evidence.

The discontinued fullwiki corpus and its preparation tools have been removed.
The official evaluator remains available locally for scoring parity tests.
