#!/bin/bash
#
# run_multihoprag.sh — run the MultiHop-RAG dataset end to end.
#
# Thin wrapper over run_index.sh / run_benchmark.sh that pins the MultiHop-RAG
# corpus, queries, and corpus tag. MultiHop-RAG articles are plain text, so
# there is no OCR stage.
#
# Tagging: all four strategies share one dataset-level corpus tag (`multihoprag`)
# because the strategy is already encoded in the Neo4j label prefix
# (PR_/NA_/HO_) and the ms_graphrag parquet path. Benchmarks are still run
# per strategy.
#
# Usage:
#   ./run_multihoprag.sh all --model prehop           # index + benchmark (sample200)
#   ./run_multihoprag.sh index --model naive          # one strategy only
#   ./run_multihoprag.sh benchmark --model prehop --queries full
#
# Options:
#   --model   {prehop|naive|hoprag|ms_graphrag}       default: prehop
#   --queries {sample200|full}                         default: sample200
# Any other flags are forwarded to the underlying run_*.sh (e.g. --clear-graph,
# --skip-server).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"

STAGE="${1:-all}"; shift || true
MODEL="prehop"
QUERIES="sample200"
COMMON_PASS=()
INDEX_PASS=()
BENCH_PASS=()
while [ $# -gt 0 ]; do
    case $1 in
        --model)   MODEL="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        --clear-graph|--save-intermediate) INDEX_PASS+=("$1"); shift ;;
        --skip-server) INDEX_PASS+=("$1"); BENCH_PASS+=("$1"); shift ;;
        *) COMMON_PASS+=("$1"); shift ;;
    esac
done

# Map the query-set selector to its corpus dir, corpus tag, and queries file.
case "$QUERIES" in
    sample200) CORPUS_DIR="data/multihoprag_corpus";       CORPUS_TAG="multihoprag";       QUERIES_FILE="data/multihoprag_sample200_queries.json" ;;
    full)      CORPUS_DIR="data/multihoprag_corpus";       CORPUS_TAG="multihoprag";       QUERIES_FILE="data/multihoprag_queries.json" ;;
    *) echo "Unknown --queries '$QUERIES' (use sample200|full)"; exit 1 ;;
esac

if { [ "$STAGE" = "benchmark" ] || [ "$STAGE" = "bench" ] || [ "$STAGE" = "all" ]; } \
   && [ ! -f "$QUERIES_FILE" ]; then
    echo "Queries file not found: $QUERIES_FILE" >&2
    exit 1
fi

case "$MODEL" in
    prehop|naive|hoprag|ms_graphrag) ;;
    *) echo "Unknown --model '$MODEL' (use prehop|naive|hoprag|ms_graphrag)" >&2; exit 1 ;;
esac

do_index() {
    echo ">>> [MultiHop-RAG index] $MODEL  (dataset $CORPUS_DIR, corpus-tag $CORPUS_TAG)"
    ./run_index.sh --model "$MODEL" --dataset "$CORPUS_DIR" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${INDEX_PASS[@]}"
}

do_benchmark() {
    echo ">>> [MultiHop-RAG benchmark] $MODEL  (queries $QUERIES_FILE, corpus-tag $CORPUS_TAG)"
    ./run_benchmark.sh --model "$MODEL" --queries "$QUERIES_FILE" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${BENCH_PASS[@]}"
}

case "$STAGE" in
    index)           do_index ;;
    benchmark|bench) do_benchmark ;;
    all)             do_index; do_benchmark ;;
    *) echo "Usage: $0 <index|benchmark|all> [--model prehop|naive|hoprag|ms_graphrag] [--queries sample200|full] [extra run_*.sh flags]"; exit 1 ;;
esac
