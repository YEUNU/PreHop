import os
import subprocess
from pathlib import Path

from models.ms_graphrag import official_indexer

ROOT = Path(__file__).resolve().parents[1]


def _fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "python"
    executable.write_text('#!/bin/sh\nprintf "<%s>\\n" "$@"\n', encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _entrypoint_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(_fake_python(tmp_path)),
            "RAG_LOG_ROOT": str(tmp_path / "logs"),
            "RAG_RUN_ID": "shared-run",
        }
    )
    return env


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
            "musique",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "<data/corpus with spaces>" in first.stdout
    assert (tmp_path / "logs/index/shared-run/multihoprag/prehop.log").is_file()
    assert (tmp_path / "logs/index/shared-run/musique/naive.log").is_file()
    assert "multihoprag/prehop.log" in first.stdout
    assert "musique/naive.log" in second.stdout


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
    monkeypatch.setattr(official_indexer, "_install_litellm_router_for_gen", lambda: None)
    monkeypatch.delenv("RAG_INDEX_LOG_DIR", raising=False)

    config = official_indexer.build_config("musique", tmp_path / "input")

    assert Path(config.reporting.base_dir) == tmp_path / "ms-output/musique/_logs/internal"


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
        ["./run_dataset.sh", "musique", "index", "--model", "all", "--skip-server"],
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


def test_paper_runner_uses_run_scoped_neo4j_namespace():
    paper_runner = (ROOT / "scripts/run_paper_target.sh").read_text(encoding="utf-8")

    assert 'export RAG_INDEX_NAMESPACE="${dataset}_${run_id}"' in paper_runner
