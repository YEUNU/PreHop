#!/bin/bash
#
# run_index.sh - Index text files into Neo4j graph
#

set -e
set -o pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"
load_project_env "$SCRIPT_DIR/.env"
canonicalize_inference_transport || exit 1

# Environment (.env values override defaults; exported values override .env).
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"

SAFE_RUN_ID="${RAG_RUN_ID//[^A-Za-z0-9_.-]/_}"

# Default values
MODEL="prehop"
LLM="default"
DATASET=""
CORPUS_TAG=""
CLEAR_GRAPH=false
SAVE_INTERMEDIATE=false
SKIP_SERVER=false

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --clear-graph) CLEAR_GRAPH=true; shift 1 ;;
        --corpus-tag) CORPUS_TAG="$2"; shift 2 ;;
        --save-intermediate) SAVE_INTERMEDIATE=true; shift 1 ;;
        --skip-server) SKIP_SERVER=true; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTHON_BIN="$(resolve_python "$SCRIPT_DIR")" || exit 1
export PYTHON_BIN

LOG_DATASET="${CORPUS_TAG:-${DATASET##*/}}"
LOG_DATASET="${LOG_DATASET:-default}"
SAFE_LOG_DATASET="${LOG_DATASET//[^A-Za-z0-9_.-]/_}"
LOG_ROOT="${RAG_LOG_ROOT:-logs}"
INDEX_LOG_DIR="${LOG_ROOT}/index/${SAFE_RUN_ID}/${SAFE_LOG_DATASET}"
mkdir -p "$INDEX_LOG_DIR"
export RAG_INDEX_LOG_DIR="$INDEX_LOG_DIR"

echo "========================================="
echo "     Indexing Pre-flight Check           "
echo "========================================="
"$PYTHON_BIN" -c '
from core.config import RAGConfig as config
print(f"Indexing: analyzer={config.FULLTEXT_ANALYZER}, chunk_sentences={config.CHUNK_SENTENCES}")
'

# Validate one strategy per invocation.
if [ "$MODEL" = "all" ]; then
    echo "Run each strategy in a separate run_index.sh invocation." >&2
    exit 1
fi
if ! python3 core/strategy_registry.py --is-valid "$MODEL"; then
    echo "❌ Unknown registered model '$MODEL'."
    exit 1
fi

if [ "$SKIP_SERVER" != true ]; then
    echo "Step 1: Checking indexing services..."

    ./run_servers.sh neo4j

    # Validate the externally supplied models.
    ./run_servers.sh gen
    ./run_servers.sh embed

    # HOP-edge pre-scoring uses the Neo4j ANN vector-index score directly.
else
    echo "Step 1: Skipping server startup (Requested by caller)"
fi

# [3] Run indexing
echo ""
echo "[Step] Running indexing..."

INDEX_ARGS=(main.py --mode index --strategy "$MODEL" --model "$LLM")
[ -n "$DATASET" ] && INDEX_ARGS+=(--dataset "$DATASET")
[ -n "$CORPUS_TAG" ] && INDEX_ARGS+=(--corpus-tag "$CORPUS_TAG")
[ "$CLEAR_GRAPH" = true ] && INDEX_ARGS+=(--clear-graph)
[ "$SAVE_INTERMEDIATE" = true ] && INDEX_ARGS+=(--save-intermediate)

"$PYTHON_BIN" "${INDEX_ARGS[@]}" 2>&1 | tee "$INDEX_LOG_DIR/${MODEL}.log"
echo "Log: $INDEX_LOG_DIR/${MODEL}.log"
