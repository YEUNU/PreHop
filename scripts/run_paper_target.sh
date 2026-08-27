#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <multihoprag|musique> <prehop|naive|hoprag|ms_graphrag> <run-id> [--check]" >&2
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
case "$strategy" in
    prehop|naive|hoprag|ms_graphrag) ;;
    *) usage ;;
esac
case "$run_id" in
    ""|*[!A-Za-z0-9._-]*)
        echo "Run ID may contain only letters, digits, dot, underscore, and hyphen." >&2
        exit 2
        ;;
esac

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

if [ ! -f .env ]; then
    echo "Missing .env; copy .env.example and configure the required endpoints first." >&2
    exit 1
fi
generation_revision=$(awk -F= '$1 == "RAG_GENERATION_REVISION" {sub(/^[^=]*=/, ""); print; exit}' .env)
embedding_revision=$(awk -F= '$1 == "RAG_EMBEDDING_REVISION" {sub(/^[^=]*=/, ""); print; exit}' .env)
if [ -z "$generation_revision" ] || [ -z "$embedding_revision" ]; then
    echo "Paper targets require RAG_GENERATION_REVISION and RAG_EMBEDDING_REVISION in .env." >&2
    exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Paper targets require a clean tracked worktree." >&2
    exit 1
fi
if [ -e "data/results/$run_id" ]; then
    echo "Result directory already exists: data/results/$run_id" >&2
    exit 1
fi
if [ ! -f "data/${dataset}_corpus/corpus_manifest.json" ] || [ ! -f "data/${dataset}_queries.json" ]; then
    echo "Prepared full corpus, manifest, or query file is missing for $dataset." >&2
    exit 1
fi
if find data/index_stats -maxdepth 1 -type f -name "*_${run_id}.json" -print -quit 2>/dev/null | grep -q .; then
    echo "Index statistics already exist for run ID: $run_id" >&2
    exit 1
fi

hop_root="data/hoprag_output/runs/$run_id"
ms_root="data/ms_graphrag_output/runs/$run_id"
if [ -e "$hop_root" ] || [ -e "$ms_root" ]; then
    echo "Isolated baseline output already exists for run ID: $run_id" >&2
    exit 1
fi

export RAG_RUN_ID=$run_id
export RAG_BENCHMARK_TIMESTAMP=$run_id
export RAG_CHUNK_CACHE=off
export RAG_EMBEDDING_CACHE=off
export RAG_HOP_OUTPUT_ROOT=$hop_root
export RAG_MS_OUTPUT_ROOT=$ms_root
export RAG_BENCHMARK_CONCURRENCY=4
export RAG_BENCHMARK_CHECKPOINT_EVERY=10
export RAG_JUDGE_ENABLED=false

if [ "$check_only" = true ]; then
    echo "Ready: dataset=$dataset strategy=$strategy run_id=$run_id concurrency=4 judge=false"
    exit 0
fi

if [ "$dataset" = multihoprag ]; then
    exec ./run_multihoprag.sh all --model "$strategy" --queries full --clear-graph
fi
exec ./run_dataset.sh musique all --model "$strategy" --queries full --clear-graph
