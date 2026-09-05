#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <campaign-id> [--check]" >&2
    exit 2
}

[ "$#" -eq 1 ] || [ "$#" -eq 2 ] || usage
campaign_id=$1
check_arg=()
if [ "$#" -eq 2 ]; then
    [ "$2" = "--check" ] || usage
    check_arg=(--check)
fi
case "$campaign_id" in
    ""|*[!A-Za-z0-9._-]*)
        echo "Campaign ID may contain only letters, digits, dot, underscore, and hyphen." >&2
        exit 2
        ;;
esac

repo_root=$(cd "$(dirname "$0")/.." && pwd)
runner="$repo_root/scripts/run_paper_target.sh"
datasets=(multihoprag musique)
strategies=(prehop naive hoprag ms_graphrag browsenet proprag)

for dataset in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
        run_id="${campaign_id}-${dataset}-${strategy}"
        echo ">>> paper target: dataset=$dataset strategy=$strategy run_id=$run_id"
        if "$runner" "$dataset" "$strategy" "$run_id" "${check_arg[@]}"; then
            :
        else
            rc=$?
            echo "Paper matrix stopped: dataset=$dataset strategy=$strategy exit_code=$rc" >&2
            exit "$rc"
        fi
    done
done
