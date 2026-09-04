#!/bin/bash
#
# run_dataset.sh — run a multi-hop QA dataset end to end.
#
# Generalized version of run_multihoprag.sh's wrapper pattern for datasets
# added after MultiHop-RAG (MuSiQue) — same corpus/tag/queries
# convention, one script instead of two near-duplicates. MultiHop-RAG keeps
# its own dedicated run_multihoprag.sh (unchanged, already documented).
# All are plain text, so there is no OCR stage.
#
# Usage:
#   ./run_dataset.sh musique all --model prehop              # index + benchmark, one strategy
#
# Options:
#   --model   {prehop|naive|hoprag|ms_graphrag|browsenet|proprag} default: prehop
#   --queries {sample200|full}                                default: sample200
# Any other flags are forwarded to the underlying run_*.sh (e.g. --clear-graph,
# --skip-server).
set -e
ulimit -n 65536 2>/dev/null || true
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$SCRIPT_DIR"
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"

DATASET="$1"; shift || true
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

case "$DATASET" in
    musique) ;;
    *) echo "Unknown dataset '$DATASET' (use musique)"; exit 1 ;;
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
    echo "Queries file for --queries $QUERIES not found. Run scripts/datasets/prepare_${DATASET}.py (and scripts/datasets/make_sample.py --dataset ${DATASET}) first."
    exit 1
fi

case "$MODEL" in
    prehop|naive|hoprag|ms_graphrag|browsenet|proprag) ;;
    *) echo "Unknown --model '$MODEL'" >&2; exit 1 ;;
esac

do_index() {
    echo ">>> [$DATASET index] $MODEL  (dataset $CORPUS_DIR, corpus-tag $CORPUS_TAG)"
    ./run_index.sh --model "$MODEL" --dataset "$CORPUS_DIR" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${INDEX_PASS[@]}"
}

do_benchmark() {
    echo ">>> [$DATASET benchmark] $MODEL  (queries $QUERIES_FILE, corpus-tag $CORPUS_TAG)"
    ./run_benchmark.sh --model "$MODEL" --queries "$QUERIES_FILE" --corpus-tag "$CORPUS_TAG" "${COMMON_PASS[@]}" "${BENCH_PASS[@]}"
}

# Auto-chain fallback: ensure MultiHop-RAG hoprag is definitely completed before starting MuSiQue
if [ ! -f "data/results/.hoprag_multihoprag_completed" ]; then
    if find data/results -name "*hoprag_multihoprag.summary.json" 2>/dev/null | grep -q .; then
        touch "data/results/.hoprag_multihoprag_completed"
    else
        echo "================================================================="
        echo ">>> [Auto-Chain Fallback] Running MultiHop-RAG hoprag before starting MuSiQue..."
        echo "================================================================="
        ulimit -n 65536 2>/dev/null || true
        ./run_multihoprag.sh all --model hoprag --queries full --clear-graph
        touch "data/results/.hoprag_multihoprag_completed"
    fi
fi

case "$STAGE" in
    index)           do_index ;;
    benchmark|bench) do_benchmark ;;
    all)             do_index; do_benchmark ;;
    *) echo "Usage: $0 <musique> <index|benchmark|all> [--model prehop|naive|hoprag|ms_graphrag|browsenet|proprag] [--queries sample200|full] [extra run_*.sh flags]"; exit 1 ;;
esac
