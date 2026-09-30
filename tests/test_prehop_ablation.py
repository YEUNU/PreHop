import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config import RAGConfig
from core.prehop_ablation import COMMON, PROFILES, ablation_identity
from models.prehop.graphrag import GraphRAG
from scripts.prehop_ablation import plan


def configure(monkeypatch, profile, reference=""):
    for key, value in {
        **COMMON,
        **PROFILES[profile],
        "PREHOP_ABLATION_PROFILE": profile,
        "ABLATION_Q_MINUS": True,
        "ABLATION_Q_PLUS": True,
    }.items():
        monkeypatch.setattr(RAGConfig, key, value)
    monkeypatch.setenv("RAG_PAPER_MODE", "false")
    monkeypatch.setenv("RAG_INDEX_NAMESPACE", "ablation_body")


@pytest.mark.parametrize("policy,expected", [("all", {"body": set()})])
@pytest.mark.asyncio
async def test_body_seed_traversal_and_inherited_score(monkeypatch, policy, expected):
    monkeypatch.setattr(RAGConfig, "HOP_SEED_POLICY", policy)
    monkeypatch.setattr(RAGConfig, "GRAPH_PATH_DECAY", 0.5)
    rag = GraphRAG(strategy="prehop")
    seed = {
        "id": "body",
        "title": "Body",
        "sent_id": 0,
        "text": "start",
        "representation_score": 0.8,
        "representation_scores": {"body": 0.8},
    }
    target = {"id": "target", "text": "follow-up", "source_id": "body", "path_type": "hop"}
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    rag._retrieve_with_candidate_pool = AsyncMock(return_value=([], [seed]))
    rag._expand_frontier = AsyncMock(side_effect=lambda *args: [target] if args[2] else [])
    rag._score_and_select = AsyncMock(side_effect=lambda _q, candidates, _k, **kwargs: (candidates, candidates))
    await rag.graph_search(["query"], depth=1, top_k=12)
    assert rag._expand_frontier.await_args.args[2] == expected
    candidates = rag._score_and_select.await_args.args[1]
    if policy == "all":
        assert next(x for x in candidates if x["id"] == "target")["representation_score"] == 0.4
    else:
        assert [x["id"] for x in candidates] == ["body"]


def test_ablation_metadata_does_not_relabel_primary(monkeypatch):
    monkeypatch.setattr(RAGConfig, "PREHOP_ABLATION_PROFILE", "")
    assert ablation_identity() == {}
    configure(monkeypatch, "question_full")
    full = ablation_identity()
    configure(monkeypatch, "question_body")
    assert full != ablation_identity()
    assert full["comparison_scope"] == "ablation_only"


def test_launcher_plans_distinct_outputs_without_execution(tmp_path):
    args = SimpleNamespace(
        namespace="ablation_question",
        run_id="query_a",
        profile="question_full",
        mode="benchmark",
        reference=None,
        index_stats=tmp_path / "stats.json",
        dataset=tmp_path / "corpus",
        queries=tmp_path / "queries.json",
        corpus_tag="multihoprag",
    )
    task = plan(args)
    assert "--clear-graph" not in task["command"]
    assert task["environment"]["RAG_HOP_SEED_POLICY"] == "all"
    assert task["output"].endswith("ablations/query_a/question_full")
    args.mode = "index"
    args.profile = "question_body"


@pytest.mark.asyncio
async def test_body_search_uses_only_original_query(monkeypatch):
    configure(monkeypatch, "question_body")
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0, 0.0]])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])
    await rag._retrieve_with_candidate_pool(
        "query",
        12,
        query_embedding=[1.0, 0.0],
        select_final=False,
    )
    assert rag._hybrid_rrf_candidates.await_count == 1
    assert rag._hybrid_rrf_candidates.await_args.kwargs["channel"] == "body"
    assert rag._hybrid_rrf_candidates.await_args.args[0] == "query"
    rag.llm.get_embeddings.assert_not_awaited()


def test_existing_index_plan_preserves_source_run_and_separate_output(tmp_path):
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps({'status': 'complete', 'strategy': 'prehop',
        'corpus_tag': 'multihoprag', 'run_id': 'original_index',
        'index_policy': {'index_namespace': 'existing_primary'}}))
    args = SimpleNamespace(namespace='existing_primary', run_id='new_query',
        profile='question_body', mode='benchmark', reference=None,
        index_stats=stats, dataset=tmp_path, queries=tmp_path/'queries.json',
        corpus_tag='multihoprag', reuse_existing_index=True)
    task = plan(args)
    assert task['environment']['RAG_RUN_ID'] == 'original_index'
    assert task['environment']['RAG_LLM_SEED'] == ''
    assert task['output'].endswith('/ablations/new_query/question_body')
    args.namespace = 'wrong_namespace'


def test_channel_replay_can_preserve_historical_scoring(tmp_path):
    args = SimpleNamespace(namespace='original', run_id='channel-replay',
        profile='question_body', mode='benchmark', index_stats=None,
        dataset=tmp_path, queries=tmp_path/'queries.json', corpus_tag='multihoprag',
        hop_semantic_variant='body_bridge_min')
    task = plan(args)
    assert task['environment']['RAG_HOP_SEMANTIC_VARIANT'] == 'body_bridge_min'
    assert task['environment']['RAG_HYPO_CHANNEL_VARIANT'] == 'body_only'
