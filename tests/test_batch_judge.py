import json

import pytest

from cli import benchmark
from core.config import RAGConfig
from utils import batch_judge
from utils.batch_judge import OpenAIBatchJudge
from utils.metrics import _resolve_judge_fields


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.asyncio
async def test_submit_persists_manifest_at_batch_creation(monkeypatch):
    events = []

    class _Files:
        def create(self, **_kwargs):
            return _Obj(id="file-in")

    class _Batches:
        def create(self, **_kwargs):
            return _Obj(id="batch-1")

    class _Client:
        def __init__(self, **_kwargs):
            self.files = _Files()
            self.batches = _Batches()

    monkeypatch.setattr("openai.OpenAI", _Client)
    judge = OpenAIBatchJudge("gpt-test", "sk-test")
    judge.register("row-1", "prompt")
    batch_id = await judge.submit(on_submitted=events.append)

    assert batch_id == "batch-1"
    assert events == ["batch-1"]
    request = json.loads(judge._build_jsonl().decode())
    assert request["custom_id"] == "row-1"
    assert request["url"] == "/v1/chat/completions"


def test_resolve_judge_fields_never_fabricates_missing_score():
    fields = _resolve_judge_fields(None, "answer", "gpt-test")
    assert fields["llm_judge_score"] == -1.0
    assert fields["hallucination"] == -1.0


def test_resolve_judge_fields_does_not_derive_hallucination_from_score():
    fields = _resolve_judge_fields({"score": 0.0, "reason": "incorrect"}, "substantive answer", "gpt-test")
    assert fields["llm_judge_score"] == 0.0
    assert fields["groundedness"] == -1.0
    assert fields["hallucination"] == -1.0
    assert fields["hallucination_source"] == "unjudged"


def _write_pending_run(tmp_path, payload_count=2):
    run_dir = tmp_path / "run"
    result_dir = run_dir / "naive" / "corpus"
    result_dir.mkdir(parents=True)
    result_file = result_dir / "naive_corpus.json"
    rows = [
        {
            "category": "test",
            "answer": f"answer-{idx}",
            "judge_custom_id": str(idx),
            "llm_judge_score": -1.0,
            "hallucination": -1.0,
        }
        for idx in range(2)
    ]
    result_file.write_text(
        json.dumps({"details": rows, "total_queries": 2, "status": "pending_judge"}),
        encoding="utf-8",
    )
    manifest = result_dir / "naive_corpus.pending_judge.json"
    manifest.write_text(
        json.dumps(
            {
                "batch_id": "batch-1",
                "judge_model": "gpt-test",
                "result_file": str(result_file.resolve()),
                "submitted": payload_count,
            }
        ),
        encoding="utf-8",
    )
    return run_dir, result_file, manifest


@pytest.mark.asyncio
async def test_reconcile_requires_complete_batch_before_patch(tmp_path, monkeypatch):
    run_dir, result_file, manifest = _write_pending_run(tmp_path)
    before = result_file.read_text(encoding="utf-8")
    monkeypatch.setattr(RAGConfig, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        batch_judge,
        "resolve_batches",
        lambda *_args: {
            "batch-1": {"0": {"score": 1, "groundedness": 1, "hallucination": 0, "reason": "ok"}}
        },
    )

    with pytest.raises(RuntimeError, match="missing 1/2"):
        await benchmark.reconcile_pending_judges(run_dir)

    assert result_file.read_text(encoding="utf-8") == before
    assert manifest.exists()


@pytest.mark.asyncio
async def test_reconcile_preserves_manifest_when_hallucination_field_is_missing(tmp_path, monkeypatch):
    run_dir, result_file, manifest = _write_pending_run(tmp_path)
    before = result_file.read_text(encoding="utf-8")
    monkeypatch.setattr(RAGConfig, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        batch_judge,
        "resolve_batches",
        lambda *_args: {
            "batch-1": {
                "0": {"score": 1, "groundedness": 1, "hallucination": 0, "reason": "ok"},
                "1": {"score": 0, "reason": "missing field"},
            }
        },
    )

    with pytest.raises(RuntimeError, match="missing valid score, groundedness, or hallucination"):
        await benchmark.reconcile_pending_judges(run_dir)

    assert result_file.read_text(encoding="utf-8") == before
    assert manifest.exists()


@pytest.mark.asyncio
async def test_reconcile_patches_complete_batch_and_removes_manifest(tmp_path, monkeypatch):
    run_dir, result_file, manifest = _write_pending_run(tmp_path)
    monkeypatch.setattr(RAGConfig, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        batch_judge,
        "resolve_batches",
        lambda *_args: {
            "batch-1": {
                "0": {"score": 1, "groundedness": 1, "hallucination": 0, "reason": "ok"},
                "1": {"score": 0, "groundedness": 0, "hallucination": 1, "reason": "wrong"},
            }
        },
    )

    assert await benchmark.reconcile_pending_judges(run_dir) == 1
    output = json.loads(result_file.read_text(encoding="utf-8"))
    assert output["status"] == "completed"
    assert output["avg_llm_judge_score"] == pytest.approx(0.5)
    detail_rows = [
        json.loads(line)
        for line in result_file.with_name("naive_corpus.details.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["llm_judge_score"] for row in detail_rows] == [1.0, 0.0]
    assert not manifest.exists()
