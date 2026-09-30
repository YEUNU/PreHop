"""Controls for candidate-count matching and pre-selector fact coverage."""

import json
from collections import Counter

import numpy as np
import pytest

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






def test_offline_evaluation_keeps_zero_budget_and_every_missing_fact(tmp_path):
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
    save_json(tmp_path / "protocol.json", protocol)
    rows = [{"query_id": q, "original_query_id": q, "base": ["0"], "budget": b,
             "question": ["1"], "body": ["2"]} for q, b in (("q1", 1), ("q2", 0), ("null", 0))]
    save_json(tmp_path / "snapshot.json", {"nodes": nodes, "rows": rows})
    samples = np.full((2, 3, 3, 1), -1, dtype=np.int32)
    samples[:, 0, :, 0] = [1, 2, 0]
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
    save_json(tmp_path / "queries.json", [])
    with pytest.raises(ValueError, match="Source changed"):
        evaluate(tmp_path, protocol, tmp_path / "report")


def test_evaluation_cannot_overwrite_its_archived_inputs(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite"):
        evaluate(tmp_path, {}, tmp_path)
