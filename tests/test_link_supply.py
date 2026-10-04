"""Controls for candidate-count matching and pre-selector fact coverage."""

import json
from collections import Counter

import numpy as np
import pytest

import scripts.analyze_link_supply as supply
from scripts.analyze_link_supply import (
    additions,
    coupled_sample,
    digest,
    evaluate,
    rewire_edges,
    save_json,
)


def test_rewiring_preserves_degrees_source_exclusion_and_input():
    sources = {str(i): str(i // 2) for i in range(30)}
    edges = [(str(i), str((i + offset) % 30)) for i in range(30) for offset in (5, 11)]
    before = list(edges)
    changed, audit = rewire_edges(edges, sources, 7)
    assert edges == before
    assert len(changed) == len(set(changed)) == len(edges)
    assert Counter(s for s, _ in edges) == Counter(s for s, _ in changed)
    assert Counter(t for _, t in edges) == Counter(t for _, t in changed)
    assert all(sources[s] != sources[t] for s, t in changed)
    assert audit["accepted_swaps"] > 0
    assert changed != sorted(edges)
    assert rewire_edges(edges, sources, 7) == (changed, audit)
    with pytest.raises(ValueError, match="different sources"):
        rewire_edges([("0", "1")], sources, 7)


def test_only_direct_passages_expand_and_next_overlap_adds_no_slot():
    adjacency = {"direct": {"next", "new"}, "next": {"wrong"}, "new": {"wrong"}}
    assert additions({"direct"}, {"direct", "next"}, adjacency) == {"new"}


def test_shared_sampling_does_not_create_exclusivity_for_identical_pools():
    first = ["a", "b", "c", "d"]
    selected = coupled_sample([first, first[::-1], ["a", "c", "e"]], 2, "query:42")
    assert selected[0] == selected[1]
    assert all(len(set(s)) == 2 for s in selected)
    assert coupled_sample([[], first], 0, "query:42") == [[], []]
    with pytest.raises(ValueError, match="budget"):
        coupled_sample([first, []], 1, "query:42")






@pytest.mark.parametrize("recorded_arms", [None, ["shuffled", "body", "question"]])
def test_offline_evaluation_keeps_zero_budget_and_every_missing_fact(tmp_path, monkeypatch, recorded_arms):
    nodes = [{"id": str(i), "text": text} for i, text in enumerate(("Alpha", "Beta Gamma", "Beta"))]
    queries = [{"_id": "q1", "evidence_facts": ["Alpha", "Beta", "Gamma", "Never"]},
               {"_id": "q2", "evidence_facts": ["Alpha", "Never"]},
               {"_id": "null", "evidence_facts": []}]
    query_path = tmp_path / "queries.json"
    save_json(query_path, queries)
    protocol = {"analysis": "fixture", "sources": {"queries": {"path": str(query_path),
                                                             "sha256": digest(query_path)}},
                "population": {"eligible_fact_queries": 2},
                "inference": {"bootstrap_seed": 42, "resamples": 100}, "interpretation": "diagnostic"}
    if recorded_arms is not None:
        protocol["conditions"] = recorded_arms
    save_json(tmp_path / "protocol.json", protocol)
    rows = [{"query_id": q, "original_query_id": q, "base": ["0"], "budget": b,
             "question": ["1"], "body": ["2"]} for q, b in (("q1", 1), ("q2", 0), ("null", 0))]
    save_json(tmp_path / "snapshot.json", {"nodes": nodes, "rows": rows})
    samples = np.full((2, 3, 3, 1), -1, dtype=np.int32)
    samples[:, 0, :, 0] = [1, 2, 0]
    if recorded_arms is not None:
        original_arms = ["question", "body", "shuffled"]
        samples = samples[:, :, [original_arms.index(arm) for arm in recorded_arms], :]
    np.savez_compressed(tmp_path / "samples.npz", samples=samples)
    save_json(tmp_path / "prepared.json", {
        "protocol_sha256": digest(tmp_path / "protocol.json"),
        "files": {n: digest(tmp_path / n) for n in ("snapshot.json", "samples.npz")},
    })
    evaluate(tmp_path, protocol, tmp_path / "report")
    report = json.loads((tmp_path / "report/comparison.json").read_text())
    assert report["eligible_queries"] == 2
    assert report["null_query_ids"] == ["null"]
    assert report["budget"]["mean"] == .5
    assert report["conditions"]["question"]["added_fact_recall"]["mean"] == .25
    assert set(report["conditions"]) == {"question", "shuffled"}
    assert report["conditions"]["shuffled"]["added_fact_recall"]["mean"] == 0
    assert report["conditions"]["question"]["complete_pool_gain"]["mean"] == 0

    archived_hashes = {name: digest(tmp_path / name) for name in
                       ("protocol.json", "prepared.json", "snapshot.json", "samples.npz", "queries.json")}
    monkeypatch.setattr(supply, "coupled_sample", lambda *args, **kwargs: pytest.fail("Must reuse archived samples"))
    evaluate(tmp_path, protocol, tmp_path / "with-body", include_body=True)
    with_body = json.loads((tmp_path / "with-body/comparison.json").read_text())
    assert list(with_body["conditions"]) == ["question", "body", "shuffled"]
    assert with_body["conditions"]["body"]["added_fact_recall"]["mean"] == .125
    assert with_body["contrasts"]["body_minus_question"]["added_fact_recall"]["mean"] == -.125
    assert with_body["contrasts"]["body_minus_shuffled"]["added_fact_recall"]["mean"] == .125
    for key in ("baseline", "budget", "prepared_queries", "eligible_queries", "null_queries", "null_query_ids"):
        assert with_body[key] == report[key]
    assert with_body["contrasts"]["question_minus_shuffled"] == report["contrasts"]["question_minus_shuffled"]
    for old_column, arm in enumerate(("question", "shuffled")):
        assert with_body["conditions"][arm] == report["conditions"][arm]  # Includes unchanged bootstrap intervals.
        new_column = with_body["realization_variation"]["arms"].index(arm)
        assert with_body["realization_variation"]["sd"][new_column] == report["realization_variation"]["sd"][old_column]
        for old, new in zip(report["details"], with_body["details"], strict=True):
            assert new["matched"][arm] == old["matched"][arm]
    assert [line.split(",")[0] for line in (tmp_path / "with-body/summary.csv").read_text().splitlines()[1:]] == [
        "question", "body", "shuffled"]
    assert {name: digest(tmp_path / name) for name in archived_hashes} == archived_hashes
    with np.load(tmp_path / "samples.npz") as archive:
        np.testing.assert_array_equal(archive["samples"], samples)
    evaluate(tmp_path, protocol, tmp_path / "default-after-body")
    assert json.loads((tmp_path / "default-after-body/comparison.json").read_text()) == report
    assert supply.ARMS == ("question", "shuffled")

    monkeypatch.setattr("sys.argv", ["analyze_link_supply", "--prepared", str(tmp_path),
                                    "--include-body", "--output", str(tmp_path / "cli-with-body")])
    supply.main()
    assert json.loads((tmp_path / "cli-with-body/comparison.json").read_text()) == with_body
    save_json(tmp_path / "queries.json", [])
    with pytest.raises(ValueError, match="Source changed"):
        evaluate(tmp_path, protocol, tmp_path / "report")


def test_evaluation_cannot_overwrite_its_archived_inputs(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite"):
        evaluate(tmp_path, {}, tmp_path)


def test_include_body_rejects_archive_without_body_samples(tmp_path):
    protocol = {"conditions": ["question", "shuffled"]}
    save_json(tmp_path / "protocol.json", protocol)
    save_json(tmp_path / "snapshot.json", {})
    np.savez_compressed(tmp_path / "samples.npz", samples=np.full((2, 1, 2, 1), -1, dtype=np.int32))
    save_json(tmp_path / "prepared.json", {
        "protocol_sha256": digest(tmp_path / "protocol.json"),
        "files": {name: digest(tmp_path / name) for name in ("snapshot.json", "samples.npz")},
    })
    with pytest.raises(ValueError, match="candidate arms"):
        evaluate(tmp_path, protocol, tmp_path / "report", include_body=True)
    assert not (tmp_path / "report/comparison.json").exists()
