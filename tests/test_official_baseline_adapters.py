from __future__ import annotations

import json
import sys

import pytest

from models.official_baseline_runtime import (
    OFFICIAL_REVISIONS,
    OfficialQueryWorker,
    corpus_records_sha256,
    run_index_worker,
    source_set_sha256,
    stage_corpus,
    verify_snapshot,
)


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


def test_external_snapshot_verification_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_LIGHTRAG_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("RAG_EMBEDDING_REVISION", "test-revision")
    target = tmp_path / "output" / "hotpotqa"
    target.mkdir(parents=True)
    (target / "artifacts").mkdir()
    (target / "artifacts/index.bin").write_bytes(b"index")
    metadata = {
        "status": "complete",
        "strategy": "lightrag",
        "official_revision": OFFICIAL_REVISIONS["lightrag"],
        "embedding_model": "test-embedding",
        "embedding_revision": "test-revision",
        "source_count": 2,
        "source_set_sha256": source_set_sha256(["a", "b"]),
        "corpus_records_sha256": corpus_records_sha256(
            [
                {"source_id": "a", "title": "A", "text": "one"},
                {"source_id": "b", "title": "B", "text": "two"},
            ]
        ),
        "corpus_manifest_fingerprint": "fingerprint",
    }
    from core.semantic_config import semantic_config_sha256
    from models.official_baseline_runtime import artifact_inventory
    semantic = {"embedding_model": "test-embedding", "embedding_revision": "test-revision"}
    metadata.update(semantic_config_id="lightrag-paper-v1", semantic_config=semantic,
                    semantic_config_sha256=semantic_config_sha256(semantic))
    metadata["artifact_inventory"] = artifact_inventory(target)
    (target / "input").mkdir()
    (target / "input" / "corpus.json").write_text(
        json.dumps(
            [
                {"source_id": "a", "title": "A", "text": "one"},
                {"source_id": "b", "title": "B", "text": "two"},
            ]
        ),
        encoding="utf-8",
    )
    (target / "index_snapshot_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert verify_snapshot("lightrag", "hotpotqa", ["b", "a"], {"fingerprint": "fingerprint"}) == metadata
    with pytest.raises(RuntimeError, match="source set"):
        verify_snapshot("lightrag", "hotpotqa", ["a", "c"], {"fingerprint": "fingerprint"})
    with pytest.raises(RuntimeError, match="fingerprint"):
        verify_snapshot("lightrag", "hotpotqa", ["a", "b"], {"fingerprint": "changed"})
    metadata["embedding_revision"] = "changed-revision"
    (target / "index_snapshot_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="embedding revision"):
        verify_snapshot("lightrag", "hotpotqa", ["a", "b"], {"fingerprint": "fingerprint"})


def test_research_snapshot_requires_valid_semantic_config_hash(tmp_path, monkeypatch):
    from core.semantic_config import semantic_config_sha256

    monkeypatch.setenv("RAG_LIGHTRAG_OUTPUT_ROOT", str(tmp_path / "output"))
    target = tmp_path / "output" / "hotpotqa"
    (target / "input").mkdir(parents=True)
    (target / "artifacts").mkdir()
    (target / "artifacts/index.bin").write_bytes(b"index")
    records = [{"source_id": "a", "title": "A", "text": "one"}]
    (target / "input/corpus.json").write_text(json.dumps(records), encoding="utf-8")
    semantic = {"embedding_model": "test-embedding", "embedding_revision": None, "query_mode": "mix"}
    metadata = {
        "status": "complete", "strategy": "lightrag",
        "official_revision": OFFICIAL_REVISIONS["lightrag"],
        "embedding_model": "test-embedding", "embedding_revision": None,
        "semantic_config_id": "lightrag-paper-v1", "semantic_config": semantic,
        "semantic_config_sha256": semantic_config_sha256(semantic),
        "source_count": 1, "source_set_sha256": source_set_sha256(["a"]),
        "corpus_records_sha256": corpus_records_sha256(records),
        "corpus_manifest_fingerprint": "fingerprint",
    }
    from models.official_baseline_runtime import artifact_inventory
    metadata["artifact_inventory"] = artifact_inventory(target)
    snapshot = target / "index_snapshot_metadata.json"
    snapshot.write_text(json.dumps(metadata), encoding="utf-8")
    assert verify_snapshot("lightrag", "hotpotqa", ["a"], {"fingerprint": "fingerprint"}) == metadata
    metadata["semantic_config_sha256"] = "0" * 64
    snapshot.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="semantic config"):
        verify_snapshot("lightrag", "hotpotqa", ["a"], {"fingerprint": "fingerprint"})


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

    monkeypatch.setattr(runtime, "validate_runtime", lambda _strategy: None)
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

    monkeypatch.setattr(runtime, "validate_runtime", lambda _strategy: None)
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
