"""Link controls, frozen inputs, and selector failure/resume boundaries."""
import copy
from collections import Counter

import pytest

from scripts.analyze_link_supply import digest
from scripts.campaign_runtime import atomic_json
from scripts.compare_link_representations import (
    ARMS,
    BASE_ARM,
    adjacency,
    candidates_for_query,
    evaluate,
    evidence_funnel,
    funnel_summary,
    match_degrees,
    read,
    select,
    verify_prepared,
)


def test_degree_control_is_query_independent_and_matches_unique_destinations():
    nodes = [{"id": str(i), "source": str(i)} for i in range(12)]
    graphs = {arm: [(str(i), str((i + d) % 12)) for i in range(12) for d in distances]
              for arm, distances in zip(ARMS, [(1, 3, 5), (2, 4), (1,)], strict=False)}
    original = copy.deepcopy(graphs)
    matched, audit = match_degrees(graphs, nodes, 42)
    assert graphs == original
    assert all(Counter(a for a, _ in edges) == Counter({str(i): 1 for i in range(12)})
               for edges in matched.values())
    assert Counter(b for _, b in matched["question"]) == Counter(b for _, b in matched["shuffled"])
    assert all(a != b for edges in matched.values() for a, b in edges)
    assert match_degrees(graphs, nodes, 42) == (matched, audit)
    graphs["qplus_body"].append(("0", "0"))
    with pytest.raises(ValueError, match="different-source"):
        match_degrees(graphs, nodes, 42)


def test_expansion_matches_counts_keeps_direct_score_and_never_expands_next():
    nodes = {str(i): {"id": str(i), "text": str(i), "embedding": [1., 0.]} for i in range(7)}
    direct = [{**nodes["0"], "representation_score": 3.}]
    hops = {arm: adjacency([("0", "2"), ("0", "1"), ("1", "6")]) for arm in ARMS}
    hops["body"]["0"].add("3")
    pools, audit = candidates_for_query(direct, nodes, adjacency([("0", "1")], undirected=True), hops, 12, "q")
    assert audit["added_budget"] == 1
    assert all(len(pool) == 3 for pool in pools.values())
    assert all("6" not in {n["id"] for n in pool} for pool in pools.values())
    for pool in pools.values():
        assert next(n for n in pool if n["id"] == "0")["representation_score"] == 3.
        assert next(n for n in pool if n["id"] == "1")["representation_score"] == 1.5
    assert "representation_score" not in nodes["0"]
    hops["shuffled"]["0"] = {"1"}
    pools, audit = candidates_for_query(direct, nodes, {"0": {"1"}}, hops, 12, "q")
    assert audit["added_budget"] == 0
    assert all({n["id"] for n in pool} == {"0", "1"} for pool in pools.values())


def prepared_fixture(tmp_path, *, with_base=False):
    import scripts.compare_link_representations as runner
    from core.generation_profiles import request_settings

    queries = [{"_id": "q", "query": "query", "evidence_facts": ["fact"]},
               {"_id": "null", "query": "null query", "evidence_facts": []}]
    atomic_json(tmp_path / "queries.json", queries)
    arms = (*ARMS, BASE_ARM) if with_base else ARMS
    protocol = {"arms": arms, "query_ids": ["q", "null"], "queries": str(tmp_path / "queries.json"),
                "queries_sha256": digest(tmp_path / "queries.json"), "seed": 42, "generation_model": "test",
                "script_sha256": digest(runner.Path(runner.__file__)),
                "ranking_settings": request_settings("ranking"),
                "selector_prompt_sha256": digest(runner.Path("utils/prompts/evidence_ranking.py"))}
    atomic_json(tmp_path / "protocol.json", protocol)
    source = {"id": "node", "text": "fact", "source": "doc", "title": "title", "page": 0, "sent_id": 0}
    for q in queries:
        atomic_json(tmp_path / "inputs" / f"{q['_id']}.json", {
            "query_id": q["_id"], "query": q["query"],
            "conditions": {arm: {"ordered": [source], "fused_sources": [source]} for arm in arms},
        })
    atomic_json(tmp_path / "input-audit.json", {
        "details": [{"query_id": q["_id"], "base": ["node"] if with_base else []} for q in queries]})
    atomic_json(tmp_path / "prepared.json", {"files": {
        name: digest(tmp_path / name)
        for name in ("protocol.json", "input-audit.json", "inputs/q.json", "inputs/null.json")}})


@pytest.mark.asyncio
@pytest.mark.parametrize("with_base", [False, True])
async def test_selector_isolated_clients_and_terminal_failures_are_retained_on_resume(tmp_path, monkeypatch, with_base):
    from core.vllm_client import VLLMClient
    from models.prehop import tracing
    from models.prehop.graphrag import GraphRAG

    prepared_fixture(tmp_path, with_base=with_base)
    expected_calls = 10 if with_base else 8
    clients, calls, closes = [], [], []

    class Client:
        model_name = "test"
        _generation_max_context_tokens = 32768

        def _count_tokens(self, _messages):
            return 20

        @classmethod
        async def global_close(cls):
            closes.append(True)

    def attach(client, _recorder):
        clients.append(client)
        return client

    async def ranking(self, query, ordered, top_k):
        assert top_k == 12
        ident = tracing._IDENTITY.get()
        calls.append((ident["query_id"], ident["arm"]))
        self.trace_recorder.emit("mock_selection", ident)
        if ident == {"query_id": "q", "arm": "shuffled"}:
            raise RuntimeError("terminal model failure")
        return ordered

    monkeypatch.setattr("core.vllm_client.VLLMClient", Client)
    monkeypatch.setattr(tracing, "attach_client", attach)
    monkeypatch.setattr(GraphRAG, "_role_body_list_ranking", ranking)
    assert await select(tmp_path, 2, None) == 1
    assert len(calls) == len({id(c) for c in clients}) == expected_calls
    assert await select(tmp_path, 2, None) == 1
    assert len(calls) == expected_calls
    assert len(closes) == 2
    assert VLLMClient is not Client
    evaluate(tmp_path)
    result = read(tmp_path / "evaluation.json")
    assert result["eligible"] == 1
    assert result["null_query_ids"] == ["null"]
    assert result["failures"]["shuffled/llm"] == 1
    assert result["conditions"]["shuffled/llm"]["MAP@10"]["mean"] == 0
    assert result["conditions"]["question/llm"]["MAP@10"]["mean"] == 1
    assert result["candidate_coverage"]["question"]["added_fact_recall"]["mean"] == (0 if with_base else 1)
    assert result["selector_interactions"]["question_minus_shuffled:llm_minus_fused"]["MAP@10"]["mean"] == 1
    if with_base:
        assert result["contrasts"]["shuffled/llm_minus_base_only/llm"]["MAP@10"]["mean"] == -1
        assert result["funnels"]["shuffled/llm"]["all_complete_losses_to_base"] == 1
        assert result["funnels"]["shuffled/llm"]["retention_given_supply"] is None
        assert result["funnels"]["base_only/llm"]["supply_queries"] == 0


def test_prepared_inputs_cannot_change_between_selection_and_evaluation(tmp_path):
    prepared_fixture(tmp_path)
    verify_prepared(tmp_path)
    atomic_json(tmp_path / "inputs/q.json", {})
    with pytest.raises(ValueError, match="Frozen pilot input changed"):
        verify_prepared(tmp_path)


def test_funnel_tracks_same_new_fact_and_keeps_supply_distinct_from_complete_selection():
    base = [{"text": "Alpha"}]
    pool = [*base, {"text": "Beta Gamma"}]
    facts = ["Alpha", "Beta", "Gamma"]
    supplied = evidence_funnel(facts, base, pool, base, base)
    retained = evidence_funnel(facts, base, pool, [{"text": "Beta Gamma"}], base)
    complete = evidence_funnel(facts, base, pool, pool, base)
    assert supplied["supplied_fact_indices"] == [1, 2]
    assert supplied["survived_fact_indices"] == []
    assert retained["retained_new_fact"] and not retained["selected_complete"]
    assert complete["complete_after_retention"] and complete["complete_gain_over_base"]
    summary = funnel_summary([supplied, retained, complete])
    assert (summary["supply_queries"], summary["retention_queries"], summary["complete_after_retention_queries"]) == (3, 2, 1)
    assert summary["retention_given_supply"] == pytest.approx(2 / 3)
    assert summary["completion_given_retention"] == .5
    # Repeated selected passages never create additional recovered facts.
    duplicate = evidence_funnel(facts, base, pool, [*pool, *pool], base)
    assert duplicate == complete
    outside_top_ten = evidence_funnel(facts, base, pool, base * 10 + [pool[1]], base)
    assert not outside_top_ten["retained_new_fact"]


def test_completeness_can_change_without_new_evidence_and_zero_supply_has_no_retention_rate():
    pool = [{"text": "Alpha"}, {"text": "Beta"}]
    changed = evidence_funnel(["Alpha", "Beta"], pool, pool, pool, pool[:1])
    assert changed["complete_gain_over_base"]
    assert not changed["supplied_new_fact"]
    assert not changed["complete_after_retention"]
    summary = funnel_summary([changed])
    assert summary["retention_given_supply"] is None
    assert summary["all_complete_gains_over_base"] == 1
    assert funnel_summary([])["retention_given_supply"] is None
