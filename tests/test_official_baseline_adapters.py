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
        "Title: Human title\nParagraph-ID: musique:abc\n\nBody sentence.\nSecond line.", encoding="utf-8"
    )
    monkeypatch.setenv("RAG_BROWSENET_OUTPUT_ROOT", str(tmp_path / "output"))

    rows, target = stage_corpus("browsenet", corpus, "musique")

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
    monkeypatch.setenv("RAG_BROWSENET_PYTHON", str(venv_python))

    assert official_python("browsenet") == venv_python
    assert _runtime_env("browsenet")["PATH"].split(":", 1)[0] == str(venv_python.parent)


def test_external_snapshot_verification_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_PROPRAG_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("VLLM_SERVED_EMBED_MODEL_NAME", "test-embedding")
    monkeypatch.setenv("RAG_EMBEDDING_REVISION", "test-revision")
    target = tmp_path / "output" / "musique"
    target.mkdir(parents=True)
    metadata = {
        "status": "complete",
        "strategy": "proprag",
        "official_revision": OFFICIAL_REVISIONS["proprag"],
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

    assert verify_snapshot("proprag", "musique", ["b", "a"], {"fingerprint": "fingerprint"}) == metadata
    with pytest.raises(RuntimeError, match="source set"):
        verify_snapshot("proprag", "musique", ["a", "c"], {"fingerprint": "fingerprint"})
    with pytest.raises(RuntimeError, match="fingerprint"):
        verify_snapshot("proprag", "musique", ["a", "b"], {"fingerprint": "changed"})
    monkeypatch.setenv("RAG_EMBEDDING_REVISION", "changed-revision")
    with pytest.raises(RuntimeError, match="embedding revision"):
        verify_snapshot("proprag", "musique", ["a", "b"], {"fingerprint": "fingerprint"})


class _FakeWorker:
    def __init__(self, strategy, corpus_tag):
        self.strategy = strategy
        self.corpus_tag = corpus_tag

    def request(self, payload):
        assert payload == {"operation": "query", "query": "question"}
        return {
            "ok": True,
            "documents": [
                {"source_id": "doc-1", "title": "First", "text": "Evidence one", "score": 0.9},
                {"source_id": "doc-2", "title": "Second", "text": "Evidence two", "score": 0.8},
            ],
        }

    def close(self):
        return None


class _FakeLLM:
    async def generate_response(self, messages, **kwargs):
        assert "Evidence one" in messages[0]["content"]
        return "answer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "adapter_name", "strategy"),
    [
        ("models.browsenet.browsenet_adapter", "BrowseNetAdapter", "browsenet"),
        ("models.proprag.proprag_adapter", "PropRAGAdapter", "proprag"),
    ],
)
async def test_official_adapter_preserves_retrieval_order_and_common_answer_boundary(
    monkeypatch, module_name, adapter_name, strategy
):
    module = __import__(module_name, fromlist=[adapter_name])
    monkeypatch.setattr(module, "OfficialQueryWorker", _FakeWorker)
    monkeypatch.setattr(module, "get_llm_client", lambda _model_id: _FakeLLM())
    adapter = getattr(module, adapter_name)(corpus_tag="musique")

    answer, sources, trace = await adapter.run_workflow("question")

    assert answer == "@@ANSWER: answer"
    assert [row["source"] for row in sources] == ["doc-1", "doc-2"]
    assert trace[0]["retrieved"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["browsenet", "proprag"])
async def test_external_capacity_excludes_staged_input(tmp_path, monkeypatch, strategy):
    from cli.index import _collect_index_capacity

    root = tmp_path / strategy
    monkeypatch.setenv(f"RAG_{strategy.upper()}_OUTPUT_ROOT", str(root))
    target = root / "musique"
    (target / "input").mkdir(parents=True)
    (target / "artifacts").mkdir()
    (target / "input" / "corpus.json").write_bytes(b"x" * 100)
    (target / "artifacts" / "graph.bin").write_bytes(b"y" * 17)

    capacity = await _collect_index_capacity(strategy, "musique")

    assert capacity["bytes"] == 17


@pytest.mark.parametrize("strategy", ["browsenet", "proprag"])
def test_external_index_policy_records_litellm_embedding(monkeypatch, strategy):
    from cli.index import _resolved_index_policy
    from core.config import RAGConfig

    policy = _resolved_index_policy(strategy, "default")

    assert policy["embedding_model"] == RAGConfig.EMBEDDING_MODEL
    assert policy["embedding_dimensions"] == RAGConfig.EMBEDDING_DIMENSIONS
    assert policy["embedding_transport"] == "litellm"


def test_proprag_index_policy_records_its_query_instruction():
    from cli.index import _resolved_index_policy

    policy = _resolved_index_policy("proprag", "default")

    assert policy["embedding_query_instruction"].startswith("Given a question")
    assert "shared entity" in policy["embedding_query_instruction"]


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

    worker = OfficialQueryWorker("browsenet", "test")
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

    result = run_index_worker("browsenet", "test", {"operation": "index"})

    assert result["stats"] == {"documents": 2}
    assert "official index progress" in capsys.readouterr().err
