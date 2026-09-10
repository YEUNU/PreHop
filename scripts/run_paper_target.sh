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
PYTHON_BIN=$(resolve_method_python "$repo_root" "$strategy") || exit 1
export PYTHON_BIN
if [ "$strategy" = "hoprag" ]; then export UV_PROJECT_ENVIRONMENT="$(dirname "$(dirname "$PYTHON_BIN")")"; fi

if [ ! -f .env ]; then
    echo "Missing .env; copy .env.example and configure the required endpoints first." >&2
    exit 1
fi
load_project_env "$repo_root/.env"
canonicalize_inference_transport || exit 1
# Preserve an observed or explicitly pinned backend revision when supplied;
# otherwise the served alias remains the only available revision identity.
# Local method revisions come from the typed strategy registry.
export RAG_GENERATION_REVISION="${RAG_GENERATION_REVISION:-$RAG_GENERATION_MODEL}"
export RAG_EMBEDDING_REVISION="${RAG_EMBEDDING_REVISION:-$RAG_EMBEDDING_MODEL}"
export RAG_PAPER_MODE=true
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Warning: tracked worktree changes will be recorded in code provenance; semantic compatibility is checked from model config." >&2
fi
output_row=$(python3 core/strategy_registry.py --output-tsv | awk -F '\t' -v strategy="$strategy" '$1 == strategy {print $2 "\t" $3}')
if [ -n "$output_row" ]; then
    IFS=$'\t' read -r output_env output_default <<< "$output_row"
    strategy_output="$output_default/runs/$run_id"
    export "$output_env=$strategy_output"
fi

export RAG_CHUNK_CACHE_DIR="data/index_cache/runs/$run_id/$strategy/$dataset"
export RAG_RUN_ID=$run_id
export RAG_PAPER_MODE=true
export RAG_BENCHMARK_TIMESTAMP=$run_id
# Keep the public corpus tag stable for result aggregation while placing all
# Neo4j-backed strategies in fresh, run-scoped labels/indexes. A global graph
# clear is intentionally not used: it is not namespace-scoped and can destroy
# a concurrently running target's index.
export RAG_INDEX_NAMESPACE="${dataset}_${run_id}"
export RAG_CHUNK_CACHE=off
export RAG_EMBEDDING_CACHE=off
generation_seed=$(python3 core/strategy_registry.py --generation-seed "$strategy")
if [ -z "$generation_seed" ]; then
    export RAG_LLM_SEED=""
else
    export RAG_LLM_SEED="${RAG_LLM_SEED:-$generation_seed}"
fi
# A method-native empty instruction is an explicit override of the shared
# default. Preserve empty exports so dotenv cannot restore the shared value.
method_instruction=$(python3 core/strategy_registry.py --query-instruction "$strategy")
shared_instruction=$(python3 core/strategy_registry.py --query-instruction prehop)
export EMBEDDING_QUERY_INSTRUCTION="$method_instruction"
export RAG_JUDGE_ENABLED=false
export RAG_JUDGE_BATCH=false

index_stats="data/index_stats/${strategy}_${dataset}_${run_id}.json"
export RAG_INDEX_STATS_PATH="$index_stats"

reuse_index=false
if [ -f "$index_stats" ]; then
    reuse_index=true
fi
partial_result="data/results/$run_id/$strategy/$dataset/seed_42/${strategy}_${dataset}.json"
if [ -f "$partial_result" ]; then
    export RAG_BENCHMARK_RESUME=on
fi

if [ "$check_only" = true ]; then
    echo "Ready: dataset=$dataset strategy=$strategy run_id=$run_id concurrency=$RAG_BENCHMARK_CONCURRENCY embedding_batch=$RAG_EMBEDDING_BATCH_SIZE embedding_concurrency=$RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS judge=false"
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
