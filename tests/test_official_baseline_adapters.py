from __future__ import annotations

import io
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.benchmark_failures import BenchmarkIntegrityError
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


@pytest.mark.parametrize('mode', ['index', 'ready'])
def test_worker_initialization_and_protocol_failure_release_resources(tmp_path, monkeypatch, mode):
    import models.official_baseline_runtime as runtime

    script = tmp_path / 'failing_worker.py'
    script.write_text("import sys, time\nsys.stdin.readline()\n" + (
        "print('__PREHOP_OFFICIAL_RESULT__=invalid-json', flush=True)\ntime.sleep(60)\n"
        if mode == 'index' else 'sys.exit(3)\n'))
    lock = io.StringIO()
    processes = []
    popen = runtime.subprocess.Popen

    def start(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runtime, '_acquire_runtime_lock', lambda strategy: lock)
    monkeypatch.setattr(runtime, '_runtime_env', lambda strategy: {})
    monkeypatch.setattr(runtime, '_command', lambda *args: [sys.executable, str(script)])
    monkeypatch.setattr(runtime.subprocess, 'Popen', start)
    if mode == 'index':
        with pytest.raises(json.JSONDecodeError):
            run_index_worker('lightrag', 'test', {'operation': 'index'})
    else:
        with pytest.raises(BenchmarkIntegrityError, match='closed stdout'):
            OfficialQueryWorker('lightrag', 'test')
    assert lock.closed
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


@pytest.mark.parametrize('failure', ['pipe', 'eof', 'logs'])
def test_query_worker_transport_failure_is_target_fatal(monkeypatch, failure):
    import models.official_baseline_runtime as runtime

    # Model only the transport boundary; no native retrieval is involved.
    worker = OfficialQueryWorker.__new__(OfficialQueryWorker)
    worker.strategy = 'lightrag'
    worker._lock = threading.Lock()
    worker._runtime_lock = None
    stdin = Mock()
    process = SimpleNamespace(stdin=stdin, terminate=Mock(), poll=lambda: 3)
    worker._process = process
    worker._stdout_queue = Mock()
    if failure == 'pipe':
        stdin.write.side_effect = BrokenPipeError('closed')
    elif failure == 'eof':
        worker._stdout_queue.get.return_value = None
    else:
        worker._stdout_queue.get.return_value = 'upstream progress noise'
        monkeypatch.setenv('RAG_OFFICIAL_QUERY_TIMEOUT', '1')
        # The queue stays nonempty across the absolute deadline.
        times = iter([0., .5, 2.])
        monkeypatch.setattr(runtime.time, 'monotonic', lambda: next(times))
    try:
        with pytest.raises(BenchmarkIntegrityError):
            worker.request({'operation': 'query', 'query': 'test'})
        if failure == 'logs':
            process.terminate.assert_called_once()
            worker._stdout_queue.get.assert_called_once()
    finally:
        worker._process = None
