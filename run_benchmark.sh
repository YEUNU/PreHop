#!/bin/bash
#
# run_benchmark.sh - Run benchmark evaluation
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"
load_project_env "$SCRIPT_DIR/.env"

# Environment (.env values override defaults; exported values override .env).
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export NEO4J_FULLTEXT_ANALYZER="${NEO4J_FULLTEXT_ANALYZER:-english}"
export RAG_RUN_ID="${RAG_RUN_ID:-$(date +"%Y%m%d_%H%M%S_%N")_$$}"
export RAG_BENCHMARK_TIMESTAMP="${RAG_BENCHMARK_TIMESTAMP:-$RAG_RUN_ID}"

PYTHON_BIN="$(resolve_python "$SCRIPT_DIR")" || exit 1

# Default values
QUERIES_FILE="data/multihoprag_sample200_queries.json"
MODEL="prehop"
LLM="default"
RUN_ALL=false
CORPUS_TAG=""
SKIP_SERVER=false

# Parse arguments
while [ $# -gt 0 ]; do
    case $1 in
        --queries) QUERIES_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --corpus-tag) CORPUS_TAG="--corpus-tag $2"; shift 2 ;;
        --skip-server) SKIP_SERVER=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================="
echo "     Benchmark Pre-flight Check          "
echo "========================================="
echo "Python: $PYTHON_BIN"
echo "Retrieval: analyzer=${NEO4J_FULLTEXT_ANALYZER}, top_k=${RAG_DEFAULT_TOP_K:-12}"

echo "Step 0: Python/Dependency preflight..."
if [ "$MODEL" = "hoprag" ] || [ "$MODEL" = "ms_graphrag" ] || [ "$RUN_ALL" = true ]; then
    if ! PREFLIGHT_MODEL="$MODEL" PREFLIGHT_ALL="$RUN_ALL" "$PYTHON_BIN" - <<'PY'
import importlib
import os
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
importlib.import_module("loguru")
importlib.import_module("typing_extensions")
model = os.environ["PREFLIGHT_MODEL"]
run_all = os.environ["PREFLIGHT_ALL"] == "true"
if model == "hoprag" or run_all:
    from models.hoprag.hoprag_adapter import HopRAGAdapter  # noqa: F401
if model == "ms_graphrag" or run_all:
    from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter  # noqa: F401
print("Dependency preflight: OK")
PY
    then
        echo "ERROR: Python preflight failed."
        exit 1
    fi
fi

if [ "$SKIP_SERVER" != true ]; then
    echo "Step 1: Checking benchmark services..."

    # MS GraphRAG's benchmark reads its parquet/LanceDB artifacts directly. The
    # other strategies query Neo4j, as does benchmark_all.
    if [ "$MODEL" != "ms_graphrag" ] || [ "$RUN_ALL" = true ]; then
        ./run_servers.sh neo4j
        if ! wait_for_server "http://localhost:7474" "Neo4j"; then exit 1; fi
    fi

    # Start Generation Server
    ./run_servers.sh gen
    if ! wait_for_server "${VLLM_URL%/}/models" "Generation Model" "200"; then exit 1; fi

    # Start Embedding Service
    ./run_servers.sh embed
    if ! wait_for_server "${VLLM_EMBED_URL%/}/models" "Embedding Model" "200"; then exit 1; fi

else
    echo "Step 1: Skipping server startup (requested by caller)"
fi

# [2] Run benchmark
echo ""
echo "[Step] Running benchmark..."
if [ "$RUN_ALL" = true ]; then
    "$PYTHON_BIN" main.py --mode benchmark_all --queries_file "$QUERIES_FILE" --model "$LLM" $CORPUS_TAG
else
    "$PYTHON_BIN" main.py --mode benchmark --queries_file "$QUERIES_FILE" --strategy "$MODEL" --model "$LLM" $CORPUS_TAG
fi

# JSON/JSONL report artifacts are written by cli/benchmark.py during the run.
