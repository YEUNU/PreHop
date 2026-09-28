# Answer synthesis from saved retrieval

Use answer-only replay to compare saved evidence with the same reader. It does
not rerun retrieval, graph expansion, evidence selection, indexing, or embeddings.
`TraceContextProvider` supplies the recorded context to a common final-answer
hook. There is no live-retriever fallback.

## Prepare inputs

Pass one completed result file per dataset/system in the intended execution
order. The preparation command reads `details[].retrieved_sources`, includes
every returned passage in its original order, and applies common title/page/chunk
labels. It adds no passage-count or token cutoff. Gold answers and previous
answers are excluded from generation inputs.

```bash
uv run python -m scripts.prepare_synthesis_inputs \
  path/to/prehop_multihoprag.json path/to/hoprag_multihoprag.json \
  path/to/prehop_hotpotqa.json path/to/hoprag_hotpotqa.json \
  --output data/synthesis/inputs.jsonl
```

Add the remaining system results to the same command. Each JSONL record contains
`dataset`, `method`, `query_id`, `query`, the complete `context`, its digest,
and source provenance. `SynthesisInput.messages()` is the final-answer hook:
it wraps that evidence in the shared prompt without modifying the evidence.
Other trace importers can supply the same record format with an explicit source
identity; never silently substitute a live search when a trace is unavailable.

This preparation policy evaluates exposed returned passages. It does not add
structured graph context available only in some native request logs. It is a
common-reader evidence comparison, not a reproduction of each native answer
pipeline. All returned passages have equal eligibility; their lengths and total
amounts remain system dependent.

## Generate answers

Configure the gateway and model in `.env` as described in
[RUNTIME_REQUIREMENTS](RUNTIME_REQUIREMENTS.md). Run:

```bash
uv run python main.py --mode synthesize \
  --trace-inputs data/synthesis/inputs.jsonl \
  --output-dir data/results/common-reader --concurrency 24
```

The default synthesis concurrency is 24 questions within one dataset/system.
Groups run sequentially in first-seen input order. Generation uses the configured
model, the shared evidence-checking prompt, temperature zero, non-thinking mode,
and a 256-token output limit. Inputs are sent directly to the configured gateway
without a local queue server or client-side truncation. An upstream context
length rejection is an execution error, not a request to remove evidence.

The output directory contains `responses.jsonl`, `status.json`, `events.jsonl`,
and `execution_config.json`. The configuration can change concurrency between
request windows; `pause: true` stops after the current window. Requests start at
least one second apart by default. A 429 causes a shared cooldown, including
`Retry-After`, and retries the same input. Other eligible transport errors have
bounded retries. An empty successful generation remains an empty output and is
not regenerated based on answer quality.

Resume with the same command. Reuse requires the same question/context messages,
reader settings and model; changed-input responses are removed from the current
response file. Retrieval source files remain untouched. `generation_complete`
means every input has a successful response record, including any empty outputs;
it is not an official-score or publication verification receipt. Score the
answers with the dataset evaluator while retaining the original retrieval
predictions, populations and question identities.

Recorded reader-request durations exclude retrieval and do not provide new
end-to-end query latency. The serving model must accommodate the entire input;
inspect actual contexts and the gateway's model limits before a long run.
