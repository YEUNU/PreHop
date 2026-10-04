#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <multihoprag|hotpotqa> <primary-strategy> <run-id> [--check]" >&2
    exit 2
}

[ "$#" -eq 3 ] || [ "$#" -eq 4 ] || usage
dataset=$1
strategy=$2
run_id=$3
check_only=false
if [ "$#" -eq 4 ]; then
    [ "$4" = "--check" ] || usage
    check_only=true
fi

case "$dataset" in
    multihoprag|hotpotqa) ;;
    *) usage ;;
esac
if ! python3 "$(dirname "$0")/../core/strategy_registry.py" --is-primary "$strategy"; then
    usage
fi
case "$run_id" in
    ""|*[!A-Za-z0-9._-]*)
        echo "Run ID may contain only letters, digits, dot, underscore, and hyphen." >&2
        exit 2
        ;;
esac

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"
. "$repo_root/scripts/lib.sh"
load_project_env "$repo_root/.env"
PYTHON_BIN=$(resolve_python "$repo_root") || exit 1
export PYTHON_BIN

canonicalize_inference_transport "$strategy" "$dataset" "$run_id" || exit 1
# Preserve an observed or explicitly pinned backend revision when supplied;
# otherwise the served alias remains the only available revision identity.
export RAG_GENERATION_REVISION="${RAG_GENERATION_REVISION:-$RAG_GENERATION_MODEL}"
export RAG_EMBEDDING_REVISION="${RAG_EMBEDDING_REVISION:-$RAG_EMBEDDING_MODEL}"
export RAG_PAPER_MODE=true
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Warning: tracked worktree changes will be recorded in code provenance." >&2
fi
export RAG_BENCHMARK_TIMESTAMP=$run_id
export RAG_CHUNK_CACHE=off
export RAG_EMBEDDING_CACHE=off

index_stats="$RAG_INDEX_STATS_PATH"

reuse_index=false
if [ -f "$index_stats" ]; then
    reuse_index=true
fi
partial_result="data/results/$run_id/$strategy/$dataset/seed_42/${strategy}_${dataset}.json"
if [ -f "$partial_result" ]; then
    export RAG_BENCHMARK_RESUME=on
fi

if [ "$check_only" = true ]; then
    echo "Ready: dataset=$dataset strategy=$strategy run_id=$run_id concurrency=$RAG_BENCHMARK_CONCURRENCY embedding_batch=$RAG_EMBEDDING_BATCH_SIZE embedding_concurrency=$RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS"
    exit 0
fi

if [ "$dataset" = multihoprag ]; then
    stage=all
    [ "$reuse_index" = true ] && stage=benchmark
    ./run_multihoprag.sh "$stage" --model "$strategy" --queries full
else
    stage=all
    [ "$reuse_index" = true ] && stage=benchmark
    ./run_dataset.sh "$dataset" "$stage" --model "$strategy" --queries full
fi

"$PYTHON_BIN" scripts/record_paper_completion.py "$run_id" "$dataset" "$strategy" --exact-run-id \
    --output "data/results/$run_id/admission.json"
