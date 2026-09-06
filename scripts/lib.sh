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
    local generation_base="${RAG_INFERENCE_BASE_URL:-}"
    local legacy_name
    for legacy_name in VLLM_API_BASE VLLM_URL VLLM_EMBED_API_BASE VLLM_EMBED_URL VLLM_API_KEY VLLM_SERVED_MODEL_NAME VLLM_SERVED_EMBED_MODEL_NAME OPENAI_BASE_URL OPENAI_API_BASE OPENAI_PROVIDER AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_KEY; do
        if [ -n "${!legacy_name:-}" ]; then
            echo "ERROR: $legacy_name is not a public input; configure only the canonical RAG_* LiteLLM contract." >&2
            return 1
        fi
    done
    generation_base="${generation_base%/}"
    if [ -z "$generation_base" ]; then
        echo "ERROR: paper inference requires one canonical OpenAI-compatible LiteLLM base URL." >&2
        return 1
    fi
    export RAG_INFERENCE_BASE_URL="$generation_base"
    export RAG_INFERENCE_API_KEY="${RAG_INFERENCE_API_KEY:-}"
    export RAG_GENERATION_MODEL="${RAG_GENERATION_MODEL:-}"
    export RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-}"
    if [ -z "$RAG_INFERENCE_API_KEY" ] || [ -z "$RAG_GENERATION_MODEL" ] || [ -z "$RAG_EMBEDDING_MODEL" ]; then
        echo "ERROR: canonical LiteLLM key and registered generation/embedding model names are required." >&2
        return 1
    fi
    local registry defaults name value
    registry="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/core/strategy_registry.py"
    defaults=$(python3 "$registry" --paper-defaults-tsv) || return 1
    while IFS=$'\t' read -r name value; do
        export "$name=${!name:-$value}"
    done <<< "$defaults"
    local repo_root python method_defaults
    repo_root="$(dirname "$(dirname "$registry")")"
    python=$(resolve_python "$repo_root") || return 1
    method_defaults=$(cd "$repo_root" && "$python" -c 'from core.paper_policy import method_environment_defaults; [print(k+"\t"+v) for k,v in method_environment_defaults().items()]') || return 1
    while IFS=$'\t' read -r name value; do
        if [ -z "${!name+x}" ]; then export "$name=$value"; fi
    done <<< "$method_defaults"

}

# Resolve the Python interpreter. Prefers the project-local .venv (created with
# `uv venv --python 3.12 .venv`), so the run scripts work for a fresh clone
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
    echo "  uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e ." >&2
    return 1
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local success_codes="${3:-200|401|405}"
    local max_attempts="${4:-300}"
    local attempt=0
    local check_url="$url"
    local curl_args=(-s -o /dev/null -w "%{http_code}" --max-time 2)

    if [ -n "${RAG_INFERENCE_API_KEY:-}" ] && [ "$check_url" = "$url" ]; then
        curl_args+=(-H "Authorization: Bearer ${RAG_INFERENCE_API_KEY}")
    fi

    echo "Wait for $name ($check_url)..."
    while [ "$attempt" -lt "$max_attempts" ]; do
        if curl "${curl_args[@]}" "$check_url" | grep -qE "$success_codes"; then
            echo " ✅ $name is Ready!"
            return 0
        fi
        printf "."
        sleep 5
        attempt=$((attempt + 1))

        if [ $((attempt % 12)) -eq 0 ]; then
            echo " (Waiting for ${name}... $((attempt * 5))s elapsed)"
        fi
    done

    echo ""
    echo " ❌ ERROR: $name failed to start after $((max_attempts * 5 / 60)) minutes."
    return 1
}
