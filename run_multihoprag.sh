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
#   ./run_multihoprag.sh all                          # index all 4 + benchmark (sample200)
#   ./run_multihoprag.sh index                        # index all 4 strategies
#   ./run_multihoprag.sh benchmark --queries full     # benchmark all on full 2556
#   ./run_multihoprag.sh index --model prehop         # one strategy only
#
# Options:
#   --model   {all|prehop|naive|hoprag|ms_graphrag}   default: all
#   --queries {sample200|full}                         default: sample200
# Any other flags are forwarded to the underlying run_*.sh (e.g. --clear-graph,
# --skip-server).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"

STRATEGIES=(prehop naive hoprag ms_graphrag)

STAGE="${1:-all}"; shift || true
MODEL="all"
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

models_to_run() {
    if [ "$MODEL" = "all" ]; then printf '%s\n' "${STRATEGIES[@]}"; else echo "$MODEL"; fi
}

do_index() {
    local first_model=true
    for m in $(models_to_run); do
        local model_index_pass=()
        for flag in "${INDEX_PASS[@]}"; do
            if [ "$flag" = "--clear-graph" ] && [ "$first_model" != true ]; then
                continue
            fi
            model_index_pass+=("$flag")
        done
        echo ">>> [MultiHop-RAG index] $m  (dataset $CORPUS_DIR, corpus-tag $CORPUS_TAG)"
        ./run_index.sh --model "$m" --dataset "$CORPUS_DIR" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${model_index_pass[@]}"
        first_model=false
    done
}

do_benchmark() {
    for m in $(models_to_run); do
        echo ">>> [MultiHop-RAG benchmark] $m  (queries $QUERIES_FILE, corpus-tag $CORPUS_TAG)"
        ./run_benchmark.sh --model "$m" --queries "$QUERIES_FILE" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${BENCH_PASS[@]}"
    done
}

case "$STAGE" in
    index)           do_index ;;
    benchmark|bench) do_benchmark ;;
    all)             do_index; do_benchmark ;;
    *) echo "Usage: $0 <index|benchmark|all> [--model all|<strategy>] [--queries sample200|full] [extra run_*.sh flags]"; exit 1 ;;
esac
