#!/bin/bash
#
# run_servers.sh - Centralized service manager for Prehop
# Usage: ./run_servers.sh {neo4j|gen|embed|all}
#
# Model inference is external. This script validates configured endpoints and
# never launches local model processes.

set -e

# Environment
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

. "$SCRIPT_DIR/scripts/lib.sh"
load_project_env "$SCRIPT_DIR/.env"
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD must be set in .env}"

SERVICE=$1

# Optional Java setup (for local/custom Neo4j distributions)
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

resolve_neo4j_cmd() {
    if [ -n "${NEO4J_BIN:-}" ] && [ -x "${NEO4J_BIN}" ]; then
        echo "${NEO4J_BIN}"
        return 0
    fi
    if [ -n "${NEO4J_HOME:-}" ] && [ -x "${NEO4J_HOME}/bin/neo4j" ]; then
        echo "${NEO4J_HOME}/bin/neo4j"
        return 0
    fi
    if command -v neo4j >/dev/null 2>&1; then
        command -v neo4j
        return 0
    fi
    return 1
}

curl_with_auth() {
    local url="$1"
    if [ -n "$VLLM_API_KEY" ] && [ "$VLLM_API_KEY" != "EMPTY" ]; then
        curl -fsS --max-time 5 -H "Authorization: Bearer ${VLLM_API_KEY}" "$url"
    else
        curl -fsS --max-time 5 "$url"
    fi
}

endpoint_has_model() {
    local base_url="${1%/}"
    local model_id="$2"
    local payload
    if ! payload="$(curl_with_auth "${base_url}/models")"; then
        return 1
    fi
    MODEL_ID="$model_id" PAYLOAD="$payload" "$SCRIPT_DIR/.venv/bin/python" -c '
import json, os
payload = json.loads(os.environ["PAYLOAD"])
raise SystemExit(0 if os.environ["MODEL_ID"] in {str(row.get("id", "")) for row in payload.get("data", [])} else 1)
' >/dev/null 2>&1
}

apply_neo4j_docker_limits() {
    local container_name="$1"
    local neo4j_docker_cpus="${NEO4J_DOCKER_CPUS:-12}"
    local neo4j_docker_cpuset="${NEO4J_DOCKER_CPUSET:-}"
    local update_args=()

    if [ -n "${neo4j_docker_cpus}" ]; then
        update_args+=(--cpus "${neo4j_docker_cpus}")
    fi

    if [ -n "${neo4j_docker_cpuset}" ]; then
        update_args+=(--cpuset-cpus "${neo4j_docker_cpuset}")
    fi

    if [ ${#update_args[@]} -gt 0 ]; then
        docker update "${update_args[@]}" "${container_name}" > /dev/null
    fi

    if [ -n "${neo4j_docker_cpus}" ]; then
        echo "Applied Neo4j Docker CPU limit: ${neo4j_docker_cpus}"
    fi

    if [ -n "${neo4j_docker_cpuset}" ]; then
        echo "Applied Neo4j Docker CPU pinning: ${neo4j_docker_cpuset}"
    fi
}

start_neo4j_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        return 1
    fi

    local container_name="${NEO4J_CONTAINER_NAME:-prehop-neo4j}"
    local neo4j_user="${NEO4J_USER:-neo4j}"
    local neo4j_password="${NEO4J_PASSWORD}"
    local neo4j_docker_cpus="${NEO4J_DOCKER_CPUS:-12}"
    local neo4j_docker_cpuset="${NEO4J_DOCKER_CPUSET:-}"
    local docker_args=()

    if [ -n "${neo4j_docker_cpus}" ]; then
        docker_args+=(--cpus "${neo4j_docker_cpus}")
    fi

    if [ -n "${neo4j_docker_cpuset}" ]; then
        docker_args+=(--cpuset-cpus "${neo4j_docker_cpuset}")
    fi

    if docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
        apply_neo4j_docker_limits "${container_name}"
        echo "✅ Neo4j Docker container is already UP (${container_name})"
        return 0
    fi

    if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
        echo "Starting Neo4j Docker container (${container_name})..."
        docker start "${container_name}" > /dev/null
        apply_neo4j_docker_limits "${container_name}"
        return 0
    fi

    local neo4j_data_dir="${NEO4J_DATA_DIR:-${SCRIPT_DIR:-.}/neo4j_data}"
    mkdir -p "${neo4j_data_dir}"

    echo "Starting Neo4j Docker container (${container_name})..."
    docker run -d \
        --name "${container_name}" \
        "${docker_args[@]}" \
        -p 7474:7474 \
        -p 7687:7687 \
        -v "${neo4j_data_dir}:/data" \
        -e NEO4J_AUTH="${neo4j_user}/${neo4j_password}" \
        -e NEO4J_server_memory_pagecache_size=8g \
        -e NEO4J_server_memory_heap_initial__size=8g \
        -e NEO4J_server_memory_heap_max__size=16g \
        -e NEO4J_dbms_memory_transaction_total_max=10g \
        neo4j:5-community > /dev/null
}

start_neo4j() {
    if ! curl -s --max-time 1 http://localhost:7474 > /dev/null 2>&1; then
        local neo4j_cmd
        if ! neo4j_cmd="$(resolve_neo4j_cmd)"; then
            echo "Neo4j local binary not found. Trying Docker fallback..."
            if start_neo4j_docker; then
                return 0
            fi
            echo "❌ Neo4j not found. Set NEO4J_BIN/NEO4J_HOME, install neo4j on PATH, or install Docker."
            return 1
        fi
        echo "Starting Neo4j..."
        mkdir -p logs
        nohup "${neo4j_cmd}" start > logs/neo4j.log 2>&1 &
    else
        echo "✅ Neo4j is already UP"
    fi
}

start_gen() {
    local configured_url="${VLLM_URL:-}"
    local configured_model="${VLLM_SERVED_MODEL_NAME:-generation-model}"
    if [ -z "$configured_url" ]; then
        echo "❌ VLLM_URL must point to an external inference endpoint." >&2
        return 1
    fi
    if endpoint_has_model "$configured_url" "$configured_model"; then
        echo "✅ Generation endpoint/model is available (${configured_url}, ${configured_model})"
        return 0
    fi
    echo "❌ Generation endpoint is unreachable or missing model '${configured_model}': ${configured_url}" >&2
    return 1
}

start_embed() {
    local configured_url="${VLLM_EMBED_URL:-}"
    local configured_model="${VLLM_SERVED_EMBED_MODEL_NAME:-embedding-model}"
    if [ -z "$configured_url" ]; then
        echo "❌ VLLM_EMBED_URL must point to an external inference endpoint." >&2
        return 1
    fi
    if endpoint_has_model "$configured_url" "$configured_model"; then
        echo "✅ Embedding endpoint/model is available (${configured_url}, ${configured_model})"
        return 0
    fi
    echo "❌ Embedding endpoint is unreachable or missing model '${configured_model}': ${configured_url}" >&2
    return 1
}

case $SERVICE in
    neo4j)  start_neo4j ;;
    gen)    start_gen ;;
    embed)  start_embed ;;
    all)
        start_neo4j
        start_gen
        start_embed
        ;;
    *)
        echo "Usage: $0 {neo4j|gen|embed|all}"
        exit 1
        ;;
esac
