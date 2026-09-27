import gzip
import json
from types import SimpleNamespace

import pytest

from scripts.compare_prehop_graph_selection import apply_ranking, evaluate, generate, read_inputs
from utils.prompts.graph_evidence import add_graph_evidence


def nodes():
    return [{"id": "a", "title": "First", "page": 1, "sent_id": 0, "text": "unrelated text"},
            {"id": "b", "title": "Second", "page": 1, "sent_id": 0, "text": "decisive fact",
             "retrieval_paths": [{"kind": "hop", "source_chunk_id": "a"}]}]


def test_graph_evidence_uses_candidate_ids_without_inventing_connections():
    first, second = nodes()
    second["retrieval_paths"] += [{"kind": "hop", "source_chunk_id": "missing"},
                                  {"kind": "hop", "source_chunk_id": "b"},
                                  {"kind": "hop", "source_chunk_id": "a"}]
    prompt = add_graph_evidence("original prompt", {"C003": second, "C008": first})
    assert prompt.count("HOP: C008 -> C003") == 1
    assert "missing" not in prompt
    assert "HOP: C003 -> C003" not in prompt
    assert "original prompt" in prompt


def test_native_short_rank_fill_and_duplicate_behavior():
    candidates = dict(zip(["C000", "C001"], nodes(), strict=True))
    short = apply_ranking({"ranking": ["C001"]}, candidates, ["a", "b"], 2)
    assert [s["chunk_id"] for s in short] == ["b", "a"]
    duplicated = apply_ranking({"ranking": ["C000", "C000"]}, candidates, ["a", "b"], 2)
    assert [s["chunk_id"] for s in duplicated] == ["a"]
    with pytest.raises(KeyError):
        apply_ranking({"ranking": ["C999"]}, candidates, ["a", "b"], 2)


def reference_with_traces(tmp_path):
    trace = tmp_path / "events.jsonl"
    start = {"ordered": nodes(), "query_text": "Which fact?", "top_k": 1}
    start["ordered"][0]["embedding"] = [0.1, 0.2]
    request = {"messages": [{"role": "user", "content": "Rank C000 unrelated text; C001 decisive fact."}],
               "response_format": {"json_schema": {"name": "prehop_ranking_v1"}}}
    events = []
    for name, data in [("SimilarityScoringMixin._role_body_list_ranking.start", start),
                       ("http.request", {"body": json.dumps(request)})]:
        payload = f"{len(events)}.json.gz"
        (tmp_path / payload).write_bytes(gzip.compress(json.dumps(data).encode()))
        events.append({"event": name, "identity": {"query_id": "q1"}, "payload": payload})
    trace.write_text("".join(json.dumps(e) + "\n" for e in events))
    return {"dataset": "multihoprag", "details": [{"query_id": "q1", "ground_truth": "private answer",
            "expected_sources": {"facts": ["decisive fact"]},
            "prehop_trace": {"events_path": str(trace)}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", ["MultiHop-RAG", "multihoprag"])
async def test_replay_runs_and_scores_without_private_inputs_or_a_graph(tmp_path, monkeypatch, dataset):
    from core import inference_telemetry
    from core.vllm_client import VLLMClient

    ref = reference_with_traces(tmp_path)
    ref["dataset"] = dataset
    inputs = read_inputs(ref)
    assert "ground_truth" not in inputs[0]
    assert "expected_sources" not in inputs[0]
    assert "embedding" not in inputs[0]["nodes"][0]
    seen = []

    async def fake_generate(self, messages, **kwargs):
        seen.append(messages)
        assert "private answer" not in str(messages)
        inference_telemetry.record("generation", SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)))
        return {"ranking": ["C001" if "Stored graph connections:" in messages[0]["content"] else "C000"]}

    monkeypatch.setattr(VLLMClient, "generate_json", fake_generate)
    output = tmp_path / "out"
    calls = await generate(inputs, output)
    evaluate(ref, calls, output)
    assert len(seen) == 2
    baseline = json.loads((output / "current.official.json").read_text())
    graph = json.loads((output / "graph_aware.official.json").read_text())
    assert baseline["metrics"]["MAP@10"] == 0
    assert graph["metrics"]["MAP@10"] == 1
    assert graph["answer"] is None
    assert all(r["usage"]["generation_calls"] == 1 for r in calls.values())
    assert all(r["usage"]["embedding_calls"] == 0 for r in calls.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "lookup"])
async def test_failed_selection_stays_in_metric_population(tmp_path, monkeypatch, failure):
    from core.vllm_client import VLLMClient

    ref = reference_with_traces(tmp_path)

    async def fail(self, messages, **kwargs):
        if failure == "transport":
            raise RuntimeError("generation failed")
        return {"ranking": ["C999"]}

    monkeypatch.setattr(VLLMClient, "generate_json", fail)
    output = tmp_path / "out"
    calls = await generate(read_inputs(ref), output)
    if failure == "lookup":
        assert calls["q1", "graph_aware"]["response"] == {"ranking": ["C999"]}
    evaluate(ref, calls, output)
    report = json.loads((output / "graph_aware.official.json").read_text())
    assert report["rows"] == report["metric_rows"] == report["failed_rows"] == 1
    assert report["metrics"]["MAP@10"] == 0

    # Questions without gold evidence have no defined retrieval score.
    ref["details"][0]["expected_sources"]["facts"] = []
    evaluate(ref, calls, output)
    null_report = json.loads((output / "graph_aware.official.json").read_text())
    assert null_report["rows"] == null_report["failed_rows"] == 1
    assert null_report["metric_rows"] == 0
    assert null_report["metrics"]["MAP@10"] is None
