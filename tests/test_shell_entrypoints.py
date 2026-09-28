import os
import subprocess
import sys
from pathlib import Path

import pytest

from models.ms_graphrag import official_indexer

ROOT = Path(__file__).resolve().parents[1]


def test_service_and_dataset_help_need_no_configuration(tmp_path):
    env = {"PATH": os.environ["PATH"], "RAG_SKIP_PROJECT_ENV": "true"}
    for script in ("run_servers.sh", "run_multihoprag.sh"):
        result = subprocess.run(["bash", str(ROOT / script), "--help"], cwd=tmp_path,
                                env=env, capture_output=True, text=True, check=True)
        assert "Usage:" in result.stdout
        assert "browsenet" not in result.stdout
        assert "proprag" not in result.stdout


def _fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "python"
    executable.write_text('#!/bin/sh\nif [ "$1" = -c ]; then exec '+sys.executable+' "$@"; fi\nprintf "<%s>\\n" "$@"\n', encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _entrypoint_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    # This test deliberately supplies a fake interpreter, independent of the
    # reviewer's selected project environment.
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    for name in tuple(env):
        if name.startswith(("VLLM_", "AZURE_OPENAI")) or name in {
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "OPENAI_PROVIDER",
        }:
            env.pop(name, None)
    env.update(
        {
            "PYTHON_BIN": str(_fake_python(tmp_path)),
            "RAG_LOG_ROOT": str(tmp_path / "logs"),
            "RAG_RUN_ID": "shared-run",
            "RAG_SKIP_PROJECT_ENV": "true",
            "RAG_INFERENCE_BASE_URL": "http://litellm.test/v1",
            "RAG_INFERENCE_API_KEY": "test-key",
            "RAG_GENERATION_MODEL": "gemma-4-31b-it",
            "RAG_EMBEDDING_MODEL": "qwen3-embedding-4b",
        }
    )
    return env


@pytest.mark.parametrize("arguments", [["--model", "lightrag"], ["--all"]])
def test_benchmark_preflight_reaches_dispatch_for_current_methods(tmp_path, arguments):
    env = _entrypoint_env(tmp_path)
    python = Path(env["PYTHON_BIN"])
    python.write_text(
        '#!/bin/sh\n'
        f'if [ "$1" = - ] || [ "$1" = -c ]; then exec "{sys.executable}" "$@"; fi\n'
        'printf "dispatch:<%s>\\n" "$@"\n'
    )
    result = subprocess.run(
        ["bash", str(ROOT / "run_benchmark.sh"), "--skip-server", *arguments],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True,
    )
    assert "Dependency preflight: OK" in result.stdout
    assert "dispatch:<main.py>" in result.stdout


def _remote_service_env(tmp_path):
    env = _entrypoint_env(tmp_path)
    python = Path(env["PYTHON_BIN"])
    python.write_text(
        '#!/bin/sh\n'
        f'if [ "$1" = - ] || [ "$1" = -c ]; then exec "{sys.executable}" "$@"; fi\n'
        'printf "dispatch:<%s>\\n" "$@"\n'
    )
    # This module checks the actual arguments at the database service boundary.
    (tmp_path / "neo4j.py").write_text('''
import os
class GraphDatabase:
    @staticmethod
    def driver(uri, *, auth, **kwargs):
        assert uri == "neo4j+s://remote.example:17687"
        assert auth == ("reviewer", "secret-test-password")
        if os.environ.get("TEST_NEO4J_FAIL"):
            raise RuntimeError("secret-test-password")
        return Connection()
class Connection:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def session(self, *, database):
        assert database == "review"
        return self
    def run(self, query):
        assert query == "RETURN 1 AS ready"
        return self
    def consume(self): pass
''')
    curl = tmp_path / "curl"
    curl.write_text(
        '#!/bin/sh\n'
        'for arg; do case "$arg" in http*) [ "$arg" = "http://litellm.test/v1/models" ] || exit 2;; esac; done\n'
        'echo \'{"data":[{"id":"gemma-4-31b-it"},{"id":"qwen3-embedding-4b"}]}\'\n'
    )
    curl.chmod(0o755)
    for executable in ("docker", "neo4j"):
        path = tmp_path / executable
        path.write_text('#!/bin/sh\ntouch "$TEST_START_ATTEMPT"\nexit 1\n')
        path.chmod(0o755)
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}", PYTHONPATH=str(tmp_path),
        NEO4J_URI="neo4j+s://remote.example:17687", NEO4J_USER="reviewer",
        NEO4J_PASSWORD="secret-test-password", NEO4J_DATABASE="review",
        TEST_START_ATTEMPT=str(tmp_path / "start-attempt"),
    )
    return env


@pytest.mark.parametrize("script", ["run_index.sh", "run_benchmark.sh"])
def test_launchers_check_configured_database_without_local_http_port(tmp_path, script):
    env = _remote_service_env(tmp_path)
    result = subprocess.run(
        ["bash", str(ROOT / script), "--model", "prehop"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=True,
    )
    assert "Configured Neo4j connection is ready" in result.stdout
    assert "dispatch:<main.py>" in result.stdout
    assert not Path(env["TEST_START_ATTEMPT"]).exists()


def test_failed_remote_database_never_starts_local_services_or_leaks_credentials(tmp_path):
    env = _remote_service_env(tmp_path)
    env["TEST_NEO4J_FAIL"] = "1"
    result = subprocess.run(
        ["bash", str(ROOT / "run_servers.sh"), "neo4j"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode != 0
    assert "Neo4j connection check failed" in result.stderr
    assert "secret-test-password" not in result.stdout + result.stderr
    assert not Path(env["TEST_START_ATTEMPT"]).exists()


def test_paper_matrix_dispatches_all_registered_targets_without_gate_verification(tmp_path):
    env = _entrypoint_env(tmp_path)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_paper_matrix.sh"), "reviewer-fixture"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=True,
    )
    from core.strategy_registry import PRIMARY_STRATEGIES

    assert result.stdout.count("<reuse-target>") == 2 * len(PRIMARY_STRATEGIES)
    assert "<verify>" not in result.stdout


def test_index_logs_are_separated_by_dataset_and_strategy(tmp_path):
    env = _entrypoint_env(tmp_path)

    first = subprocess.run(
        [
            "./run_index.sh",
            "--skip-server",
            "--model",
            "prehop",
            "--dataset",
            "data/corpus with spaces",
            "--corpus-tag",
            "multihoprag",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        [
            "./run_index.sh",
            "--skip-server",
            "--model",
            "naive",
            "--dataset",
            "data/other",
            "--corpus-tag",
            "hotpotqa",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "<data/corpus with spaces>" in first.stdout
    assert (tmp_path / "logs/index/shared-run/multihoprag/prehop.log").is_file()
    assert (tmp_path / "logs/index/shared-run/hotpotqa/naive.log").is_file()
    assert "multihoprag/prehop.log" in first.stdout
    assert "hotpotqa/naive.log" in second.stdout


def test_benchmark_logs_are_separated_by_dataset_and_strategy(tmp_path):
    env = _entrypoint_env(tmp_path)

    completed = subprocess.run(
        [
            "./run_benchmark.sh",
            "--skip-server",
            "--model",
            "prehop",
            "--queries",
            "data/multihoprag queries.json",
            "--corpus-tag",
            "multihoprag",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "<data/multihoprag queries.json>" in completed.stdout
    assert (tmp_path / "logs/benchmark/shared-run/multihoprag/prehop.log").is_file()


def test_shell_preflight_reports_fixed_protocol_values(tmp_path):
    env = _entrypoint_env(tmp_path)
    env["RAG_CHUNK_SENTENCES"] = "99"
    env["RAG_DEFAULT_TOP_K"] = "99"

    indexing = subprocess.run(
        [
            "./run_index.sh",
            "--skip-server",
            "--model",
            "prehop",
            "--dataset",
            "data/corpus",
            "--corpus-tag",
            "test",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    benchmark = subprocess.run(
        [
            "./run_benchmark.sh",
            "--skip-server",
            "--model",
            "prehop",
            "--queries",
            "data/queries.json",
            "--corpus-tag",
            "test",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "chunk_sentences=6" in indexing.stdout
    assert "top_k=12" in benchmark.stdout
    assert "=99" not in indexing.stdout + benchmark.stdout


def test_shell_preflight_reports_naive_controlled_protocol(tmp_path):
    env = _entrypoint_env(tmp_path)

    indexing = subprocess.run(
        [
            "./run_index.sh",
            "--skip-server",
            "--model",
            "naive",
            "--dataset",
            "data/corpus",
            "--corpus-tag",
            "test",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    benchmark = subprocess.run(
        [
            "./run_benchmark.sh",
            "--skip-server",
            "--model",
            "naive",
            "--queries",
            "data/queries.json",
            "--corpus-tag",
            "test",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "chunk_sentences=6" in indexing.stdout
    assert "top_k=12" in benchmark.stdout


def test_ms_graphrag_internal_log_is_dataset_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(official_indexer, "_OUTPUT_ROOT", tmp_path / "ms-output")
    monkeypatch.setattr(official_indexer, "_register_external_models_with_litellm", lambda: None)
    monkeypatch.delenv("RAG_INDEX_LOG_DIR", raising=False)
    monkeypatch.setenv("VLLM_API_BASE", "http://generation/v1")
    monkeypatch.setenv("VLLM_EMBED_API_BASE", "http://embedding/v1")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "generation")
    monkeypatch.setenv("VLLM_SERVED_EMBED_MODEL_NAME", "embedding")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "generation")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding")
    monkeypatch.setattr(official_indexer, "_GEN_API_BASE", "http://generation/v1")
    monkeypatch.setattr(official_indexer, "_GEN_MODEL_NAME", "generation")
    monkeypatch.setattr(official_indexer, "_EMBED_API_BASE", "http://embedding/v1")
    monkeypatch.setattr(official_indexer, "_EMBED_MODEL_NAME", "embedding")
    monkeypatch.setattr(official_indexer, "_GEN_API_KEY", "test-key")

    config = official_indexer.build_config("hotpotqa", tmp_path / "input")

    assert Path(config.reporting.base_dir) == tmp_path / "ms-output/hotpotqa/_logs/internal"


def test_multihoprag_wrapper_runs_exactly_one_strategy(tmp_path):
    env = _entrypoint_env(tmp_path)
    completed = subprocess.run(
        ["./run_multihoprag.sh", "index", "--model", "naive", "--skip-server"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.count(">>> [MultiHop-RAG index]") == 1
    assert "<naive>" in completed.stdout
    assert "<prehop>" not in completed.stdout


def test_dataset_wrapper_rejects_multi_strategy_mode(tmp_path):
    env = _entrypoint_env(tmp_path)
    completed = subprocess.run(
        ["./run_dataset.sh", "hotpotqa", "index", "--model", "all", "--skip-server"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Unknown --model 'all'" in completed.stderr


def test_dataset_wrappers_never_launch_an_implicit_second_strategy():
    multihop = (ROOT / "run_multihoprag.sh").read_text(encoding="utf-8")
    dataset = (ROOT / "run_dataset.sh").read_text(encoding="utf-8")

    assert "./run_multihoprag.sh all --model hoprag" not in multihop
    assert "./run_multihoprag.sh all --model hoprag" not in dataset


def test_shell_and_python_share_target_configuration(tmp_path, monkeypatch):
    import json

    from core.inference_transport import InferenceTransport
    from core.paper_policy import configure_target_environment

    env = _entrypoint_env(tmp_path)
    env.update(RAG_EXECUTION_PROFILE=str(ROOT / 'configs/execution_profiles/direct-8.json'),
               RAG_GENERATION_CONCURRENCY='123', RAG_MAX_PARALLEL_FILES='123')
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    keys = ['RAG_RUN_ID', 'RAG_INDEX_NAMESPACE', 'RAG_INDEX_STATS_PATH', 'RAG_CHUNK_CACHE_DIR',
            'RAG_LIGHTRAG_OUTPUT_ROOT', 'RAG_LLM_SEED', 'EMBEDDING_QUERY_INSTRUCTION',
            'RAG_BENCHMARK_CONCURRENCY', 'RAG_GENERATION_CONCURRENCY', 'RAG_MAX_PARALLEL_FILES']
    result = subprocess.run(['bash', '-c', '''
source scripts/lib.sh
canonicalize_inference_transport lightrag hotpotqa fixture-run || exit 1
"$PYTHON_BIN" -c 'import json, os, sys; print(json.dumps({k: os.environ[k] for k in sys.argv[1:]}))' "$@"
''', 'test', *keys], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout)
    configure_target_environment('lightrag', 'hotpotqa', 'fixture-run')
    assert observed == {key: os.environ[key] for key in keys}
    assert observed['RAG_INDEX_NAMESPACE'] == 'hotpotqa_fixture-run'
    assert observed['RAG_MAX_PARALLEL_FILES'] == observed['RAG_BENCHMARK_CONCURRENCY'] == '8'
    assert int(observed['RAG_GENERATION_CONCURRENCY']) == InferenceTransport.resolve('lightrag').generation_concurrency


def test_paper_runner_uses_shared_conservative_embedding_load():
    paper_runner = (ROOT / "scripts/run_paper_target.sh").read_text(encoding="utf-8")

    from core.strategy_registry import paper_environment_defaults

    assert "canonicalize_inference_transport" in paper_runner
    assert paper_environment_defaults()["RAG_EMBEDDING_BATCH_SIZE"] == "16"
    assert paper_environment_defaults()["RAG_MAX_CONCURRENT_EMBEDDING_REQUESTS"] == "1"
    assert 'if [ "$strategy" = browsenet ]' not in paper_runner
