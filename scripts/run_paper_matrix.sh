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
. "$repo_root/scripts/lib.sh"
PYTHON_BIN=$(resolve_python "$repo_root") || exit 1
export PYTHON_BIN
runner="$repo_root/scripts/run_paper_target.sh"
datasets=(multihoprag musique)
# PropRAG is excluded from this campaign because its generation route remained
# non-responsive even with serial requests, bounded output, and backoff.
# Keeping strategy as the outer loop completes both datasets for one baseline
# before moving on.
strategy_lines=$(python3 "$repo_root/core/strategy_registry.py" --primary-lines) || {
    echo "strategy registry query failed" >&2
    exit 1
}
mapfile -t strategies <<< "$strategy_lines"
[ "${#strategies[@]}" -gt 0 ] || { echo "strategy registry returned no primary methods" >&2; exit 1; }
failed_targets=()

if [ "${#check_arg[@]}" -eq 0 ]; then
    "$PYTHON_BIN" "$repo_root/scripts/paper_gate_ledger.py" verify \
        --ledger "data/results/$campaign_id/gate_ledger.json" --campaign "$campaign_id"
fi

for strategy in "${strategies[@]}"; do
    for dataset in "${datasets[@]}"; do
        run_id="${campaign_id}-${dataset}-${strategy}"
        echo ">>> paper target: dataset=$dataset strategy=$strategy run_id=$run_id"
        if "$runner" "$dataset" "$strategy" "$run_id" "${check_arg[@]}"; then
            # run_paper_target.sh returns success only after exact-target
            # verify_paper_target.py admission in its effective environment.
            :
        else
            rc=$?
            echo "Paper target failed; continuing matrix: dataset=$dataset strategy=$strategy exit_code=$rc" >&2
            failed_targets+=("$dataset/$strategy:$rc")
        fi
    done
done

if [ "${#failed_targets[@]}" -gt 0 ]; then
    printf 'Paper matrix completed with %d failed target(s):' "${#failed_targets[@]}" >&2
    printf ' %s' "${failed_targets[@]}" >&2
    printf '\n' >&2
    exit 1
fi
