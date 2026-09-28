#!/bin/bash

# Shared shell helpers for service validation and Neo4j orchestration.

# Load a dotenv file while preserving variables explicitly exported by the
# caller. Plain `. .env` silently overwrites shell overrides; python-dotenv in
# main.py does the opposite, so the two launch paths previously disagreed.
load_project_env() {
    local env_file="$1"
    local exported_snapshot
    if [ "${RAG_SKIP_PROJECT_ENV:-false}" = "true" ]; then
        return 0
    fi
    [ -f "$env_file" ] || return 0
    exported_snapshot="$(export -p)"
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
    # ``export -p`` emits ``declare -x`` statements. Evaluating those inside
    # this function creates function-local variables, so values loaded from
    # .env remain active after return instead of caller-provided overrides.
    # Restore with ``export`` assignments, which update the shell environment
    # visible to subsequently launched commands.
    eval "${exported_snapshot//declare -x/export}"
}

canonicalize_inference_transport() {
    local repo_root python assignments
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    python=$(resolve_python "$repo_root") || return 1
    assignments=$(cd "$repo_root" && "$python" -c \
        'from scripts.runner_environment import export_runner_environment; export_runner_environment()' "$@") || return 1
    # Python uses shlex.quote for every value, including whitespace and newlines.
    eval "$assignments"
}

# Resolve the Python interpreter. Prefers the project-local .venv (created with
# `uv sync --locked --python 3.12 --extra dev`), so run scripts work for a fresh clone
# without the user having to `source .venv/bin/activate` first. Override with
# the PYTHON_BIN env var. Falls back to system python3/python.
# Usage: PYTHON_BIN="$(resolve_python "$SCRIPT_DIR")" || exit 1
resolve_python() {
    local script_dir="${1:-$(pwd)}"
    local selected="${PYTHON_BIN:-}"
    local environment="${UV_PROJECT_ENVIRONMENT:-}"
    if [ -n "$environment" ]; then
        [[ "$environment" = /* ]] || environment="$script_dir/$environment"
        if [ -n "$selected" ] && [ "$(realpath -s "$selected")" != "$(realpath -s "$environment/bin/python")" ]; then
            echo "ERROR: PYTHON_BIN and UV_PROJECT_ENVIRONMENT select different runtimes." >&2
            return 1
        fi
        selected="$environment/bin/python"
    fi
    if [ -n "$selected" ]; then
        [ -x "$selected" ] || { echo "ERROR: selected Python does not exist." >&2; return 1; }
        realpath -s "$selected"; return 0
    fi
    if [ -x "${script_dir}/.venv/bin/python" ]; then
        echo "${script_dir}/.venv/bin/python"; return 0
    fi
    if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
    if command -v python  >/dev/null 2>&1; then command -v python;  return 0; fi
    echo "ERROR: no Python interpreter found. Create the env first:" >&2
    echo "  uv sync --locked --python 3.12 --extra dev" >&2
    return 1
}

resolve_method_python() {
    local method_repo="$1"
    local method_name="$2"
    if [ "$method_name" = "hoprag" ]; then
        PYTHONPATH="$method_repo${PYTHONPATH:+:$PYTHONPATH}" python3 -c 'from core.runtime_requirements import method_main_python; print(method_main_python("hoprag"))'
    else
        resolve_python "$method_repo"
    fi
}
