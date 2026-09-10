import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config import RAGConfig
from core.prehop_ablation import COMMON, PROFILES, ablation_identity, validate_profile
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing.body_links import build_body_links, load_reference, make_reference
from scripts.prehop_ablation import plan


def configure(monkeypatch, profile, reference=""):
    for key, value in {
        **COMMON,
        **PROFILES[profile],
        "PREHOP_ABLATION_PROFILE": profile,
        "ABLATION_Q_MINUS": profile != "body_body",
        "ABLATION_Q_PLUS": profile != "body_body",
        "BODY_LINK_REFERENCE": str(reference),
    }.items():
        monkeypatch.setattr(RAGConfig, key, value)
    monkeypatch.setenv("RAG_PAPER_MODE", "false")
    monkeypatch.setenv("RAG_INDEX_NAMESPACE", "ablation_body")


def node(node_id, source, degree):
    return {
        "id": node_id,
        "source": source,
        "title": source,
        "text": node_id,
        "sent_id": 0,
        "embedding": [1.0, 0.0],
        "degree": degree,
    }


@pytest.fixture
def reference(tmp_path):
    rows = [node("a", "doc1", 1), node("b", "doc2", 0)]
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(make_reference("ablation_question", rows)))
    return path, rows


@pytest.mark.parametrize("policy,expected", [("all", {"body": set()}), ("qplus", {})])
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


@pytest.mark.asyncio
async def test_body_builder_matches_degree_without_questions(monkeypatch, reference):
    path, rows = reference
    configure(monkeypatch, "body_body", path)
    writes = []

    async def query(cypher, params=None):
        if cypher.startswith("MATCH (c:"):
            return rows if params["after"] == "" else []
        if cypher.startswith("CALL db.index.vector"):
            assert params["embedding"] == rows[0]["embedding"]
            assert params["degree"] == 1
            assert "node.source <> $source" in cypher
            return [{"id": "b", "score": 1.0}]
        if cypher.startswith("UNWIND"):
            assert "HAS_Q" not in cypher
            assert "body_to_body" in cypher
            writes.append(params)
        return []

    engine = SimpleNamespace(
        _safe_corpus="ablation_body",
        chunk_label="PR_ablation_body_Chunk",
        body_vector_index="body_index",
        retry_query=AsyncMock(side_effect=query),
    )
    await build_body_links(engine)
    assert writes == [{"id": "a", "edges": [{"id": "b", "score": 1.0}]}]


@pytest.mark.asyncio
async def test_body_builder_rejects_changed_snapshot_before_writing(monkeypatch, reference):
    path, rows = reference
    configure(monkeypatch, "body_body", path)
    rows[0]["text"] = "changed"
    query = AsyncMock(return_value=rows)
    engine = SimpleNamespace(_safe_corpus="ablation_body", chunk_label="Body", retry_query=query)
    with pytest.raises(ValueError, match="differs from reference"):
        await build_body_links(engine)
    assert query.await_count == 1


@pytest.mark.asyncio
async def test_body_only_index_never_calls_question_generator(monkeypatch, reference):
    path, _ = reference
    configure(monkeypatch, "body_body", path)
    rag = GraphRAG(strategy="prehop")
    rag.llm = SimpleNamespace()
    assert await rag.extract_hoprag_queries("passage", "title") == {"q_minus": [], "q_plus": []}


def test_profile_refuses_mixed_conditions_and_primary_namespace(monkeypatch):
    configure(monkeypatch, "question_body")
    validate_profile(RAGConfig)
    monkeypatch.setattr(RAGConfig, "HOP_SEED_POLICY", "qplus")
    with pytest.raises(ValueError, match="mismatch"):
        validate_profile(RAGConfig)
    monkeypatch.setattr(RAGConfig, "HOP_SEED_POLICY", "all")
    monkeypatch.setenv("RAG_INDEX_NAMESPACE", "live_campaign")
    with pytest.raises(ValueError, match="isolated"):
        validate_profile(RAGConfig)


def test_body_reference_cannot_be_destination(monkeypatch, reference):
    path, _ = reference
    configure(monkeypatch, "body_body", path)
    monkeypatch.setenv("RAG_INDEX_NAMESPACE", "ablation_question")
    with pytest.raises(ValueError, match="separate"):
        validate_profile(RAGConfig)


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
    with pytest.raises(ValueError, match="reuse"):
        plan(args)


def test_reference_rejects_invalid_degree(reference):
    path, _ = reference
    payload = json.loads(path.read_text())
    payload["nodes"]["a"]["degree"] = -1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="degree"):
        load_reference(path)


def test_index_policy_checks_reject_cross_profile_reuse(monkeypatch, reference):
    from cli.index import _resolved_index_policy
    from core.prehop_ablation import validate_ablation_index_policy
    from core.semantic_config import semantic_config_sha256

    path, _ = reference
    configure(monkeypatch, "question_full")
    policy = _resolved_index_policy("prehop", "default", "multihoprag")
    digest = semantic_config_sha256(policy)
    configure(monkeypatch, "question_body")
    validate_ablation_index_policy(policy, digest, "multihoprag")
    configure(monkeypatch, "body_body", path)
    with pytest.raises(RuntimeError, match="settings differ"):
        validate_ablation_index_policy(policy, digest, "multihoprag")
    body_policy = _resolved_index_policy("prehop", "default", "multihoprag")
    validate_ablation_index_policy(body_policy, semantic_config_sha256(body_policy), "multihoprag")
    with pytest.raises(RuntimeError, match="digest"):
        validate_ablation_index_policy(body_policy, digest, "multihoprag")


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


@pytest.mark.asyncio
async def test_occupied_namespace_is_never_reindexed(monkeypatch):
    from core.neo4j_service import Neo4jService
    from core.prehop_ablation import require_empty_ablation_namespace

    configure(monkeypatch, "question_full")
    query = AsyncMock(return_value=[{"count": 1}])
    monkeypatch.setattr(Neo4jService, "execute_query", query)
    with pytest.raises(RuntimeError, match="existing data is preserved"):
        await require_empty_ablation_namespace()
    assert "DELETE" not in query.await_args.args[0]


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
    with pytest.raises(ValueError, match='Namespace'):
        plan(args)


@pytest.mark.asyncio
async def test_existing_ablation_index_cannot_be_written(monkeypatch):
    from core.prehop_ablation import require_empty_ablation_namespace
    configure(monkeypatch, 'question_full')
    monkeypatch.setenv('RAG_ABLATION_REUSE_EXISTING_INDEX', 'true')
    monkeypatch.setenv('RAG_INDEX_NAMESPACE', 'existing_primary')
    validate_profile(RAGConfig)
    with pytest.raises(ValueError, match='read-only'):
        await require_empty_ablation_namespace()


def test_body_clone_plan_uses_no_indexing_pipeline(tmp_path):
    args = SimpleNamespace(namespace='ablation_cloned_body', run_id='body_clone',
        profile='body_body', mode='index', reference=tmp_path/'new_reference.json',
        index_stats=None, dataset=tmp_path, queries=tmp_path/'queries.json',
        corpus_tag='multihoprag', clone_body_from=tmp_path/'source_stats.json')
    task = plan(args)
    assert task['command'][1].endswith('scripts/clone_prehop_body.py')
    assert not any(part.endswith('main.py') for part in task['command'])
    args.profile='question_full'
    with pytest.raises(ValueError, match='body_body'):
        plan(args)


@pytest.mark.asyncio
async def test_clone_refuses_source_as_destination():
    from scripts.clone_prehop_body import copy_body_graph
    engine=SimpleNamespace(_safe_corpus='ablation_source', retry_query=AsyncMock())
    with pytest.raises(ValueError, match='separate'):
        await copy_body_graph(engine, 'ablation_source')
    engine.retry_query.assert_not_called()
