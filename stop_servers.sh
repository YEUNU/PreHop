#!/bin/bash

# Optional Java setup (for local/custom Neo4j distributions)
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

SERVICE="${1:-all}"
NEO4J_CONTAINER_NAME="${NEO4J_CONTAINER_NAME:-prehop-neo4j}"

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

is_docker_available() {
    command -v docker >/dev/null 2>&1
}

is_container_running() {
    local name="$1"
    docker ps --format '{{.Names}}' 2>/dev/null | grep -Fxq "${name}"
}

show_port_status() {
    local port
    echo ""
    echo "========================================="
    echo "Port Status Check:"
    echo "========================================="
    for port in "$@"; do
        if fuser "${port}/tcp" >/dev/null 2>&1; then
            echo "⚠️  Port ${port} still in use!"
        else
            echo "✅ Port ${port} is free"
        fi
    done
}

stop_neo4j() {
    local neo4j_cmd
    echo "[Neo4j] Stopping..."

    neo4j_cmd="$(resolve_neo4j_cmd || true)"
    if [ -n "${neo4j_cmd}" ]; then
        "${neo4j_cmd}" stop >/dev/null 2>&1 || true
    fi

    if is_docker_available && is_container_running "${NEO4J_CONTAINER_NAME}"; then
        docker stop "${NEO4J_CONTAINER_NAME}" >/dev/null || true
    fi

    sleep 1
}

echo "========================================="
echo "     Neo4j Service Shutdown              "
echo "========================================="

case "${SERVICE}" in
    neo4j)
        stop_neo4j
        show_port_status 7474 7687
        ;;
    all)
        stop_neo4j
        show_port_status 7474 7687
        ;;
    *)
        echo "Usage: $0 {neo4j|all}"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "     Requested shutdown completed.       "
echo "========================================="
