from __future__ import annotations

import json
import sys

import pytest

from models.official_baseline_runtime import OfficialQueryWorker, run_index_worker, stage_corpus


def test_stage_corpus_removes_only_transport_headers(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "source.one.txt").write_text(
        "Title: Human title\nParagraph-ID: hotpotqa:abc\n\nBody sentence.\nSecond line.", encoding="utf-8"
    )
    monkeypatch.setenv("RAG_LIGHTRAG_OUTPUT_ROOT", str(tmp_path / "output"))

    rows, target = stage_corpus("lightrag", corpus, "hotpotqa")

    assert rows == [
        {
            "source_id": "source.one",
            "title": "Human title",
            "text": "Body sentence.\nSecond line.",
        }
    ]
    assert json.loads((target / "input" / "corpus.json").read_text(encoding="utf-8")) == rows


def test_official_python_preserves_virtualenv_symlink_path(tmp_path, monkeypatch):
    from models.official_baseline_runtime import _runtime_env, official_python

    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    monkeypatch.setenv("RAG_LIGHTRAG_PYTHON", str(venv_python))
    monkeypatch.setenv("VLLM_API_BASE", "http://generation/v1")
    monkeypatch.setenv("VLLM_EMBED_API_BASE", "http://embedding/v1")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "generation")
    monkeypatch.setenv("VLLM_SERVED_EMBED_MODEL_NAME", "embedding")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "generation")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding")

    assert official_python("lightrag") == venv_python
    assert _runtime_env("lightrag")["PATH"].split(":", 1)[0] == str(venv_python.parent)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["lightrag"])
async def test_external_capacity_excludes_staged_input(tmp_path, monkeypatch, strategy):
    from cli.index import _collect_index_capacity

    root = tmp_path / strategy
    monkeypatch.setenv(f"RAG_{strategy.upper()}_OUTPUT_ROOT", str(root))
    target = root / "hotpotqa"
    (target / "input").mkdir(parents=True)
    (target / "artifacts").mkdir()
    (target / "input" / "corpus.json").write_bytes(b"x" * 100)
    (target / "artifacts" / "graph.bin").write_bytes(b"y" * 17)

    capacity = await _collect_index_capacity(strategy, "hotpotqa")

    assert capacity["bytes"] == 17


def test_persistent_worker_protocol_ignores_upstream_stdout(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(
        """import json, sys
prefix = '__PREHOP_OFFICIAL_RESULT__='
for line in sys.stdin:
    row = json.loads(line)
    print('upstream progress noise', flush=True)
    if row['operation'] == 'shutdown':
        print(prefix + json.dumps({'ok': True}), flush=True)
        break
    if row['operation'] == 'ready':
        print(prefix + json.dumps({'ok': True, 'ready': True}), flush=True)
    else:
        print(prefix + json.dumps({'ok': True, 'documents': [{'text': row['query']}]}), flush=True)
""",
        encoding="utf-8",
    )
    import models.official_baseline_runtime as runtime

    monkeypatch.setattr(runtime, "_command", lambda *_args: [sys.executable, str(script)])
    monkeypatch.setenv("RAG_LIGHTRAG_ROOT", str(tmp_path / "lightrag/source"))
    monkeypatch.setenv("VLLM_API_BASE", "http://generation/v1")
    monkeypatch.setenv("VLLM_EMBED_API_BASE", "http://embedding/v1")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "generation")
    monkeypatch.setenv("VLLM_SERVED_EMBED_MODEL_NAME", "embedding")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "generation")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding")

    worker = OfficialQueryWorker("lightrag", "test")
    try:
        assert worker.request({"operation": "query", "query": "evidence"})["documents"] == [
            {"text": "evidence"}
        ]
    finally:
        worker.close()


def test_index_worker_streams_noise_and_reads_structured_result(tmp_path, monkeypatch, capsys):
    script = tmp_path / "fake_index_worker.py"
    script.write_text(
        """import json, sys
prefix = '__PREHOP_OFFICIAL_RESULT__='
request = json.loads(sys.stdin.readline())
print('official index progress', flush=True)
print(prefix + json.dumps({'ok': request['operation'] == 'index', 'stats': {'documents': 2}}), flush=True)
""",
        encoding="utf-8",
    )
    import models.official_baseline_runtime as runtime

    monkeypatch.setattr(runtime, "_command", lambda *_args: [sys.executable, str(script)])
    monkeypatch.setenv("RAG_LIGHTRAG_ROOT", str(tmp_path / "lightrag/source"))
    monkeypatch.setenv("VLLM_API_BASE", "http://generation/v1")
    monkeypatch.setenv("VLLM_EMBED_API_BASE", "http://embedding/v1")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "generation")
    monkeypatch.setenv("VLLM_SERVED_EMBED_MODEL_NAME", "embedding")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    monkeypatch.setenv("RAG_INFERENCE_BASE_URL", "http://litellm/v1")
    monkeypatch.setenv("RAG_INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("RAG_GENERATION_MODEL", "generation")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding")

    result = run_index_worker("lightrag", "test", {"operation": "index"})

    assert result["stats"] == {"documents": 2}
    assert "official index progress" in capsys.readouterr().err
