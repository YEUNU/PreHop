#!/bin/bash
#
# run_index.sh - Index text files into Neo4j graph
#

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"
load_project_env "$SCRIPT_DIR/.env"

# [환경 설정] (.env values win over defaults; explicit shell exports win over .env)
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export NEO4J_VECTOR_DIMENSIONS="${NEO4J_VECTOR_DIMENSIONS:-1024}"
export MAX_EMBEDDING_LENGTH="${MAX_EMBEDDING_LENGTH:-16384}"
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"

PYTHON_BIN="$(resolve_python "$SCRIPT_DIR")" || exit 1
SAFE_RUN_ID="${RAG_RUN_ID//[^A-Za-z0-9_.-]/_}"
INDEX_LOG_DIR="logs/index/${SAFE_RUN_ID}"
mkdir -p "$INDEX_LOG_DIR"

# Default values
MODEL="prehop"
LLM="default"
DATASET=""
CLEAR_GRAPH=""
CORPUS_TAG=""
SAVE_INTERMEDIATE=""

SKIP_SERVER=""

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --dataset) DATASET="--dataset $2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --clear-graph) CLEAR_GRAPH="--clear-graph"; shift 1 ;;
        --corpus-tag) CORPUS_TAG="--corpus-tag $2"; shift 2 ;;
        --save-intermediate) SAVE_INTERMEDIATE="--save-intermediate"; shift 1 ;;
        --skip-server) SKIP_SERVER="true"; shift 1 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "     Indexing Pre-flight Check           "
echo "========================================="
echo "Indexing: analyzer=${NEO4J_FULLTEXT_ANALYZER}, chunk_sentences=${RAG_CHUNK_SENTENCES:-6}"

# [0] Check Arch (skip if running all models)
if [ "$MODEL" = "all" ]; then
    echo "Use scripts/run_index_matrix.py for bounded multi-strategy indexing; run_index.sh accepts one strategy." >&2
    exit 1
fi
if [ ! -d "models/$MODEL" ]; then
    echo "❌ Arch model '$MODEL' not found in models/ folder."
    exit 1
fi

if [ "$SKIP_SERVER" != "true" ]; then
    echo "Step 1: Checking indexing services..."

    # Start Neo4j
    ./run_servers.sh neo4j
    if ! wait_for_server "http://localhost:7474" "Neo4j"; then
        echo "Fatal: Neo4j failed." >&2
        exit 1
    fi

    # Start Generation Server
    ./run_servers.sh gen
    if ! wait_for_server "${VLLM_URL%/}/models" "Generation Model" "200"; then
        echo "Fatal: Generation model failed." >&2
        exit 1
    fi

    # Start Embedding Service
    ./run_servers.sh embed
    if ! wait_for_server "${VLLM_EMBED_URL%/}/models" "Embedding Model" "200"; then
        echo "Fatal: Embedding service failed." >&2
        exit 1
    fi

    # HOP-edge pre-scoring uses the Neo4j ANN vector-index score directly.
else
    echo "Step 1: Skipping server startup (Requested by caller)"
fi

# [3] Run indexing
echo ""
echo "[Step] Running indexing..."

"$PYTHON_BIN" main.py --mode index $DATASET --strategy "$MODEL" --model "$LLM" $CLEAR_GRAPH $CORPUS_TAG $SAVE_INTERMEDIATE 2>&1 \
    | tee "$INDEX_LOG_DIR/${MODEL}.log"
echo "Log: $INDEX_LOG_DIR/${MODEL}.log"
