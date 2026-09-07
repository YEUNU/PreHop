#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <campaign-id> <init|setup|preflight|gateway|record> [stage evidence.json]" >&2
    exit 2
}

[ "$#" -ge 2 ] || usage
campaign=$1
action=$2
repo_root=$(cd "$(dirname "$0")/.." && pwd)
. "$repo_root/scripts/lib.sh"
load_project_env "$repo_root/.env"
PYTHON_BIN=$(resolve_python "$repo_root") || exit 1
export PYTHON_BIN
ledger="data/results/$campaign/gate_ledger.json"
case "$campaign" in ""|*[!A-Za-z0-9._-]*) usage ;; esac
cd "$repo_root"

record_gate() {
    "$PYTHON_BIN" scripts/paper_gate_ledger.py record --ledger "$ledger" --campaign "$campaign" \
        --stage "$1" --evidence "$2"
}

case "$action" in
    init)
        [ "$#" -eq 2 ] || usage
        "$PYTHON_BIN" scripts/paper_gate_ledger.py init --ledger "$ledger" --campaign "$campaign"
        ;;
    setup|preflight|gateway)
        [ "$#" -eq 2 ] || usage
        case "$action" in
            setup) stages=(runtime_setup) ;;
            preflight) stages=(preflight_16) ;;
            gateway) stages=(chat_probe embedding_probe bisection_probe) ;;
        esac
        for stage in "${stages[@]}"; do
            "$PYTHON_BIN" scripts/paper_gate_ledger.py execute --ledger "$ledger" --campaign "$campaign" --stage "$stage"
        done
        ;;
    record)
        [ "$#" -eq 4 ] || usage
        record_gate "$3" "$4"
        ;;
    *) usage ;;
esac
