#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <multihoprag|musique> <primary-strategy> <run-id> [--check]" >&2
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
    multihoprag|musique) ;;
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
if [ -n "${EMBEDDING_QUERY_INSTRUCTION:-}" ] && [ "$EMBEDDING_QUERY_INSTRUCTION" != "$shared_instruction" ] && [ "$EMBEDDING_QUERY_INSTRUCTION" != "$method_instruction" ]; then
    echo "Embedding query instruction differs from the registered method policy." >&2
    exit 1
fi
export EMBEDDING_QUERY_INSTRUCTION="$method_instruction"
export RAG_JUDGE_ENABLED=false
export RAG_JUDGE_BATCH=false

index_stats="data/index_stats/${strategy}_${dataset}_${run_id}.json"
export RAG_INDEX_STATS_PATH="$index_stats"

# Validate the exact pinned runtime and method-defining local artifacts before
# both dry-run readiness and a real target. This check is read-only.
"$PYTHON_BIN" scripts/check_paper_runtime.py --strategy "$strategy" --dataset "$dataset"

resume_benchmark=false
reuse_index=false
if [ -e "data/results/$run_id" ]; then
    admission_args=()
    if [ "$check_only" = false ]; then
        admission_args=(--output "data/results/$run_id/admission.json")
    fi
    if "$PYTHON_BIN" scripts/verify_paper_target.py "$run_id" "$dataset" "$strategy" --exact-run-id \
        "${admission_args[@]}"; then
        echo "Strictly verified completed target; skipping: $run_id"
        exit 0
    fi
    partial_result="data/results/$run_id/$strategy/$dataset/seed_42/${strategy}_${dataset}.json"
    if [ -f "$partial_result" ] && TARGET_STRATEGY="$strategy" TARGET_DATASET="$dataset" PARTIAL_RESULT="$partial_result" "$PYTHON_BIN" - <<'PY'
import json, os, sys
path = os.environ["PARTIAL_RESULT"]
try:
    payload = json.load(open(path, encoding="utf-8"))
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
required = {
    "status": "in_progress",
    "strategy": os.environ["TARGET_STRATEGY"],
    "corpus_tag": os.environ["TARGET_DATASET"],
    "evaluation_scope": "full_benchmark",
}
raise SystemExit(0 if all(payload.get(k) == v for k, v in required.items()) else 1)
PY
    then
        resume_benchmark=true
        export RAG_BENCHMARK_RESUME=on
        echo "Strict partial benchmark artifact found; resuming: $run_id"
    else
        echo "Existing result is corrupt, incompatible, or not safely resumable: data/results/$run_id" >&2
        exit 1
    fi
fi
if [ ! -f "data/${dataset}_corpus/corpus_manifest.json" ] || [ ! -f "data/${dataset}_queries.json" ]; then
    echo "Prepared full corpus, manifest, or query file is missing for $dataset." >&2
    exit 1
fi
index_stats="data/index_stats/${strategy}_${dataset}_${run_id}.json"
export RAG_INDEX_STATS_PATH="$index_stats"
if [ -f "$index_stats" ]; then
    if "$PYTHON_BIN" scripts/verify_index_policy.py "$index_stats" "$strategy" "$dataset" "$run_id"
    then
        reuse_index=true
        echo "Strict completed index artifact found; benchmark stage only: $run_id"
    else
        echo "Existing index statistics are incomplete or incompatible: $index_stats" >&2
        exit 1
    fi
elif [ "$resume_benchmark" = true ]; then
    echo "Partial benchmark cannot resume without its completed index stats: $index_stats" >&2
    exit 1
fi

if [ "$reuse_index" = false ] && [ -e "$RAG_CHUNK_CACHE_DIR" ]; then
    echo "Fresh index cannot reuse an existing chunk-generation cache: $run_id" >&2
    exit 1
fi

if [ "$reuse_index" = false ] && [ -n "${strategy_output:-}" ] && [ -e "$strategy_output" ]; then
    echo "Isolated baseline output already exists for run ID: $run_id" >&2
    exit 1
fi


# PropRAG performs long-form extraction during indexing and fans requests out
# across the full corpus.  Keep that strategy serial so a slow generation
# provider cannot turn one target into a timeout/retry storm.  Other strategies
# retain their existing indexing and benchmark concurrency.
if [ "$strategy" = proprag ]; then
    export RAG_PROPRAG_CONCURRENT_REQUESTS="${RAG_PROPRAG_CONCURRENT_REQUESTS:-1}"
    export RAG_PROPRAG_MAX_NEW_TOKENS="${RAG_PROPRAG_MAX_NEW_TOKENS:-2048}"
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
    ./run_dataset.sh musique "$stage" --model "$strategy" --queries full
fi

"$PYTHON_BIN" scripts/verify_paper_target.py "$run_id" "$dataset" "$strategy" --exact-run-id \
    --output "data/results/$run_id/admission.json"
