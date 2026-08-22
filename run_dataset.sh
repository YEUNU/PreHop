#!/bin/bash
#
# run_dataset.sh — run a HotpotQA/MuSiQue-shaped dataset end to end.
#
# Generalized version of run_multihoprag.sh's wrapper pattern for datasets
# added after MultiHop-RAG (HotpotQA, MuSiQue) — same corpus/tag/queries
# convention, one script instead of two near-duplicates. MultiHop-RAG keeps
# its own dedicated run_multihoprag.sh (unchanged, already documented).
# All are plain text, so there is no OCR stage.
#
# Usage:
#   ./run_dataset.sh hotpotqa index                          # index all 4 strategies
#   ./run_dataset.sh hotpotqa benchmark --queries sample200   # benchmark all on the n=200 sample
#   ./run_dataset.sh musique all --model prehop              # index + benchmark, one strategy
#
# Options:
#   --model   {all|prehop|naive|hoprag|ms_graphrag}          default: all
#   --queries {sample200|full}                                default: sample200
# Any other flags are forwarded to the underlying run_*.sh (e.g. --clear-graph,
# --skip-server).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"

STRATEGIES=(prehop naive hoprag ms_graphrag)

DATASET="$1"; shift || true
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

case "$DATASET" in
    hotpotqa|musique) ;;
    *) echo "Unknown dataset '$DATASET' (use hotpotqa|musique)"; exit 1 ;;
esac

CORPUS_DIR="data/${DATASET}_corpus"
CORPUS_TAG="$DATASET"
case "$QUERIES" in
    sample200)
        QUERIES_FILE=""
        for candidate in data/${DATASET}_sample*_queries.json; do
            if [ -f "$candidate" ]; then QUERIES_FILE="$candidate"; break; fi
        done
        ;;
    full)      QUERIES_FILE="data/${DATASET}_queries.json" ;;
    *) echo "Unknown --queries '$QUERIES' (use sample200|full)"; exit 1 ;;
esac
if { [ "$STAGE" = "benchmark" ] || [ "$STAGE" = "bench" ] || [ "$STAGE" = "all" ]; } \
   && { [ -z "$QUERIES_FILE" ] || [ ! -f "$QUERIES_FILE" ]; }; then
    echo "Queries file for --queries $QUERIES not found. Run data/prepare_${DATASET}.py (and data/make_sample.py --dataset ${DATASET}) first."
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
        echo ">>> [$DATASET index] $m  (dataset $CORPUS_DIR, corpus-tag $CORPUS_TAG)"
        ./run_index.sh --model "$m" --dataset "$CORPUS_DIR" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${model_index_pass[@]}"
        first_model=false
    done
}

do_benchmark() {
    for m in $(models_to_run); do
        echo ">>> [$DATASET benchmark] $m  (queries $QUERIES_FILE, corpus-tag $CORPUS_TAG)"
        ./run_benchmark.sh --model "$m" --queries "$QUERIES_FILE" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${BENCH_PASS[@]}"
    done
}

case "$STAGE" in
    index)           do_index ;;
    benchmark|bench) do_benchmark ;;
    all)             do_index; do_benchmark ;;
    *) echo "Usage: $0 <hotpotqa|musique> <index|benchmark|all> [--model all|<strategy>] [--queries sample200|full] [extra run_*.sh flags]"; exit 1 ;;
esac
