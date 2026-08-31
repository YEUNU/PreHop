import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing.answer_links import build_continuation_index


class _AsyncRecords:
    def __init__(self, rows):
        self._rows = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._rows)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_retrieval_has_no_dataset_specific_metadata_gate():
    rag = GraphRAG(strategy="prehop")
    assert not hasattr(rag, "_extract_query_metadata")
    assert not hasattr(rag, "_apply_retrieval_calibration")
    assert not hasattr(RAGConfig, "COMPANY_ANCHORING")
    assert not hasattr(RAGConfig, "RERANKER_THRESHOLD")


@pytest.mark.asyncio
async def test_hop_page_collection_bounds_concurrency_and_preserves_order(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HOP_GATHER_WAVE", 2)
    monkeypatch.setattr(RAGConfig, "HOP_BUILD_CONCURRENCY", 2)
    rag = GraphRAG(strategy="prehop")
    active = 0
    peak = 0

    async def resolve(wave):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [[{"id": item["id"]}] for item in wave]

    rag._collect_hop_wave_candidates = resolve  # type: ignore[method-assign]
    rows = await rag._collect_hop_page_candidates([{"id": str(index)} for index in range(7)])

    assert [row[0]["id"] for row in rows] == [str(index) for index in range(7)]
    assert peak == 2


def test_primary_source_selection_uses_complete_list_ranking():
    assert RAGConfig.SOURCE_SELECTION_VARIANT == "role_body_list_ranking"


def test_continuation_index_requires_exact_foreign_document_mentions():
    index = build_continuation_index(
        [
            {
                "source_chunk_id": "source-a",
                "source": "a.txt",
                "question_id": "qm-a",
                "continuation_anchor": "Taxi Girl",
            },
            {
                "source_chunk_id": "source-b",
                "source": "b.txt",
                "question_id": "qm-b",
                "continuation_anchor": "Kim Park",
            },
        ],
        [
            {"id": "source-a", "source": "a.txt", "title": "Album", "text": "Taxi Girl performed it."},
            {"id": "target-a", "source": "c.txt", "title": "Singer", "text": "A member of TAXI GIRL."},
            {"id": "partial", "source": "d.txt", "title": "Taxi", "text": "A girl performed."},
            {"id": "source-b", "source": "b.txt", "title": "Acme", "text": "Kim Park founded it."},
        ],
    )

    assert len(index["anchors"]) == 1
    anchor = index["anchors"][0]
    assert anchor["text"] == "Taxi Girl"
    assert anchor["normalized_text"] == "taxi girl"
    assert index["question_links"] == [{"question_id": "qm-a", "anchor_id": anchor["id"]}]
    assert index["mention_links"] == [{"anchor_id": anchor["id"], "chunk_id": "target-a"}]


def test_continuation_index_shares_common_anchor_mentions_without_cartesian_edges():
    index = build_continuation_index(
        [
            {
                "source": "a.txt",
                "question_id": "qm-a",
                "continuation_anchor": "United States",
            },
            {
                "source": "b.txt",
                "question_id": "qm-b",
                "continuation_anchor": "UNITED STATES",
            },
        ],
        [
            {"id": "a", "source": "a.txt", "title": "A", "text": "United States"},
            {"id": "b", "source": "b.txt", "title": "B", "text": "United States"},
            {"id": "c", "source": "c.txt", "title": "C", "text": "United States"},
        ],
    )

    assert len(index["anchors"]) == 1
    assert len(index["question_links"]) == 2
    assert len(index["mention_links"]) == 3
    # A source-by-target representation would require six continuation edges.
    assert sum(len(index[key]) for key in ("question_links", "mention_links")) == 5


def test_all_grounded_continuation_policy_uses_answers_without_optional_marker():
    records = [
        {
            "source": "a.txt",
            "question_id": "qm-a",
            "answer": "Kim Park",
            "continuation_anchor": "",
        },
        {
            "source": "b.txt",
            "question_id": "qm-b",
            "answer": "Taxi Girl",
            "continuation_anchor": "Taxi Girl",
        },
    ]
    chunks = [
        {"id": "source-a", "source": "a.txt", "title": "Acme", "text": "Kim Park founded it."},
        {"id": "target-a", "source": "c.txt", "title": "Founder", "text": "Kim Park was born here."},
        {"id": "source-b", "source": "b.txt", "title": "Band", "text": "Taxi Girl performed."},
    ]

    named_only = build_continuation_index(records, chunks)
    all_grounded = build_continuation_index(
        records,
        chunks,
        anchor_policy="all_grounded",
    )

    assert not named_only["anchors"]
    assert [anchor["text"] for anchor in all_grounded["anchors"]] == ["Kim Park"]
    assert all_grounded["question_links"] == [{"question_id": "qm-a", "anchor_id": all_grounded["anchors"][0]["id"]}]
    assert all_grounded["mention_links"] == [{"anchor_id": all_grounded["anchors"][0]["id"], "chunk_id": "target-a"}]


@pytest.mark.asyncio
async def test_similarity_ordering_has_no_score_gate():
    rag = GraphRAG(strategy="prehop")

    selected, ordered = await rag._score_and_select(
        [1.0, 0.0],
        [
            {"id": "low", "text": "a", "embedding": [-0.8, 0.6]},
            {"id": "high", "text": "b", "embedding": [-0.2, 0.98]},
        ],
        top_k=2,
        selection_variant="global",
    )

    assert [node["id"] for node in selected] == ["high", "low"]
    assert selected == ordered


@pytest.mark.asyncio
async def test_final_order_preserves_representation_rank_without_fitted_weight():
    rag = GraphRAG(strategy="prehop")

    selected, _ = await rag._score_and_select(
        [1.0, 0.0],
        [
            {"id": "semantic-first", "embedding": [1.0, 0.0], "representation_score": 0.5},
            {"id": "representation-first", "embedding": [0.8, 0.6], "representation_score": 1.0},
            {"id": "middle", "embedding": [0.9, 0.43589], "representation_score": 0.75},
        ],
        top_k=3,
        selection_variant="global",
    )

    assert [node["id"] for node in selected] == ["semantic-first", "representation-first", "middle"]
    assert selected[0]["rank_fusion_score"] == pytest.approx(1.0 + 1.0 / 3.0)
    assert selected[1]["rank_fusion_score"] == pytest.approx(1.0 / 3.0 + 1.0)


@pytest.mark.asyncio
async def test_hop_candidate_scoring_preserves_bridge_question_semantics():
    rag = GraphRAG(strategy="prehop")
    candidate = {
        "id": "target",
        "title": "2023 manual",
        "text": "The target body.",
        "embedding": [0.6, 0.8],
        "bridge_embeddings": [[0.8, 0.6], [0.2, 0.98]],
    }

    selected, _ = await rag._score_and_select([1.0, 0.0], [candidate], top_k=1, selection_variant="global")

    assert selected[0]["similarity_score"] == 0.6
    assert selected[0]["bridge_similarity_score"] == 0.8
    assert selected[0]["final_score"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_bridge_only_hop_scoring_does_not_require_direct_query_body_match(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HOP_SEMANTIC_VARIANT", "bridge_only")
    rag = GraphRAG(strategy="prehop")
    candidate = {
        "id": "target",
        "embedding": [0.6, 0.8],
        "bridge_embeddings": [[0.8, 0.6]],
    }

    selected, _ = await rag._score_and_select([1.0, 0.0], [candidate], top_k=1, selection_variant="global")

    assert selected[0]["similarity_score"] == pytest.approx(0.6)
    assert selected[0]["bridge_similarity_score"] == pytest.approx(0.8)
    assert selected[0]["final_score"] == pytest.approx(0.8)


def test_final_selection_round_robins_sources_without_fraction_parameter():
    rag = GraphRAG(strategy="prehop")
    ordered = [
        {"id": "a1", "source": "a"},
        {"id": "a2", "source": "a"},
        {"id": "b1", "source": "b"},
        {"id": "a3", "source": "a"},
        {"id": "b2", "source": "b"},
    ]

    selected = rag._source_round_robin(ordered, top_k=4)

    assert [node["id"] for node in selected] == ["a1", "b1", "a2", "b2"]
    assert not hasattr(RAGConfig, "MAX_CHUNKS_PER_SOURCE_FRACTION")


def test_source_balanced_selection_can_revisit_a_strong_source():
    ordered = [
        {"id": "a1", "source": "a", "rank_fusion_score": 2.0, "final_score": 1.0},
        {"id": "a2", "source": "a", "rank_fusion_score": 1.8, "final_score": 0.9},
        {"id": "b1", "source": "b", "rank_fusion_score": 0.8, "final_score": 0.8},
        {"id": "c1", "source": "c", "rank_fusion_score": 0.7, "final_score": 0.7},
    ]

    selected = GraphRAG._source_balanced(ordered, top_k=3)

    assert [node["id"] for node in selected] == ["a1", "a2", "b1"]


def test_source_balancing_uses_document_title_before_paragraph_file():
    ordered = [
        {
            "id": "a1",
            "title": "A",
            "source": "paragraph-a1",
            "rank_fusion_score": 1.0,
            "final_score": 1.0,
        },
        {
            "id": "a2",
            "title": "A",
            "source": "paragraph-a2",
            "rank_fusion_score": 0.9,
            "final_score": 0.9,
        },
        {
            "id": "b1",
            "title": "B",
            "source": "paragraph-b1",
            "rank_fusion_score": 0.6,
            "final_score": 0.6,
        },
    ]

    round_robin = GraphRAG._source_round_robin(ordered, top_k=2)
    balanced = GraphRAG._source_balanced(ordered, top_k=2)

    assert [row["id"] for row in round_robin] == ["a1", "b1"]
    assert [row["id"] for row in balanced] == ["a1", "b1"]


def test_graph_pair_selection_keeps_hop_source_with_high_ranked_target():
    ordered = [
        {
            "id": "target",
            "retrieval_paths": [{"kind": "hop", "source_chunk_id": "source", "depth": 1}],
        },
        {"id": "other-1"},
        {"id": "other-2"},
        {"id": "source"},
    ]

    selected = GraphRAG._graph_pairs(ordered, top_k=3)

    assert [row["id"] for row in selected] == ["target", "source", "other-1"]


@pytest.mark.asyncio
async def test_document_balance_and_graph_pair_policies_compose(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "source_balanced_graph_pairs")
    rag = GraphRAG(strategy="prehop")
    candidates = [
        {
            "id": "target",
            "title": "A",
            "source": "a-target",
            "embedding": [1.0, 0.0],
            "retrieval_paths": [{"kind": "hop", "source_chunk_id": "source", "depth": 1}],
        },
        {"id": "a-repeat", "title": "A", "source": "a-repeat", "embedding": [0.99, 0.01]},
        {"id": "other", "title": "B", "source": "b", "embedding": [0.8, 0.2]},
        {"id": "source", "title": "C", "source": "c", "embedding": [0.7, 0.3]},
    ]

    selected, _ = await rag._score_and_select([1.0, 0.0], candidates, top_k=3)

    assert [row["id"] for row in selected] == ["target", "source", "other"]


@pytest.mark.asyncio
async def test_global_source_selection_is_an_explicit_ablation(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    candidates = [
        {"id": "a1", "source": "a", "embedding": [1.0, 0.0]},
        {"id": "a2", "source": "a", "embedding": [0.9, 0.1]},
        {"id": "b1", "source": "b", "embedding": [0.8, 0.2]},
    ]

    selected, _ = await rag._score_and_select([1.0, 0.0], candidates, top_k=2)

    assert [node["id"] for node in selected] == ["a1", "a2"]


@pytest.mark.asyncio
async def test_role_body_owner_selection_searches_bodies_without_representation_vote(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "single_combined")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "role_body_owners")
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0], [0.9], [0.8]])

    async def candidates(view, *, query_embedding, limit, channel):
        _ = query_embedding, limit
        if channel == "body":
            return [{"id": f"body-{view}", "embedding": [1.0]}]
        return [{"id": f"regular-{channel}", "embedding": [1.0]}]

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=candidates)  # type: ignore[method-assign]

    _selected, pool = await rag._retrieve_with_candidate_pool(
        "original",
        top_k=12,
        channel_queries={"q_minus": ["minus"], "q_plus": ["plus"]},
        select_final=False,
    )

    assert [(call.args[0], call.kwargs["channel"]) for call in rag._hybrid_rrf_candidates.await_args_list] == [
        ("minus", "q_minus"),
        ("plus", "q_plus"),
        ("minus", "body"),
        ("plus", "body"),
    ]
    minus_owner = next(node for node in pool if node["id"] == "body-minus")
    plus_owner = next(node for node in pool if node["id"] == "body-plus")
    assert minus_owner["role_body_owner_orders"] == [0]
    assert plus_owner["role_body_owner_orders"] == [1]
    assert minus_owner["role_body_owner_only"] is True
    assert minus_owner["representation_score"] == 0.0


def test_role_body_owner_selection_deduplicates_and_fills_global_order():
    ordered = [
        {"id": "global-first"},
        {"id": "shared", "role_body_owner_orders": [0, 2]},
        {"id": "second-owner", "role_body_owner_orders": [1]},
        {"id": "global-last"},
    ]

    selected = GraphRAG._role_body_owners(ordered, top_k=4)

    assert [node["id"] for node in selected] == [
        "shared",
        "second-owner",
        "global-first",
        "global-last",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection_variant",
    ["role_body_rounds", "role_body_list_ranking"],
)
async def test_role_body_rank_variants_retain_all_auxiliary_body_ranks(
    monkeypatch,
    selection_variant,
):
    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "single_combined")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", selection_variant)
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0], [0.9]])

    async def candidates(view, *, query_embedding, limit, channel):
        _ = query_embedding, limit
        if channel == "body":
            return [
                {"id": f"body-{view}-1", "embedding": [1.0], "rrf_score": 1.0},
                {"id": f"body-{view}-2", "embedding": [1.0], "rrf_score": 0.5},
            ]
        return [{"id": f"regular-{channel}", "embedding": [1.0]}]

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=candidates)  # type: ignore[method-assign]

    _selected, pool = await rag._retrieve_with_candidate_pool(
        "original",
        top_k=12,
        channel_queries={"q_minus": ["minus"], "q_plus": []},
        select_final=False,
    )

    first = next(node for node in pool if node["id"] == "body-minus-1")
    second = next(node for node in pool if node["id"] == "body-minus-2")
    assert first["role_body_round_entries"] == [{"view_order": 0, "rank": 0, "score": 1.0}]
    assert second["role_body_round_entries"] == [{"view_order": 0, "rank": 1, "score": 0.5}]
    assert first["role_body_owner_only"] is True


@pytest.mark.asyncio
async def test_role_body_list_ranking_completes_known_omissions(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "role_body_list_ranking")
    rag = GraphRAG(strategy="prehop")
    rag.llm.generate_json = AsyncMock(return_value={"ranking": ["C999", "C001", "C001"]})
    candidates = [
        {
            "id": "body-first",
            "title": "First",
            "text": "first evidence",
            "embedding": [1.0, 0.0],
            "role_body_round_entries": [{"view_order": 0, "rank": 0, "score": 1.0}],
        },
        {
            "id": "body-second",
            "title": "Second",
            "text": "second evidence",
            "publisher": "Example News",
            "published_at": "2026-08-30T00:00:00Z",
            "embedding": [0.9, 0.1],
            "role_body_round_entries": [{"view_order": 1, "rank": 0, "score": 0.9}],
        },
        {"id": "global", "title": "Global", "text": "other", "embedding": [0.8, 0.2]},
    ]

    selected, _ = await rag._score_and_select(
        [1.0, 0.0],
        candidates,
        top_k=2,
        query_text="Which evidence is needed?",
    )

    assert [node["id"] for node in selected] == ["body-second", "body-first"]
    prompt = rag.llm.generate_json.await_args.args[0][0]["content"]
    assert "Which evidence is needed?" in prompt
    assert all(candidate_id in prompt for candidate_id in ("C000", "C001"))
    assert "C002" in prompt
    assert "Publisher: Example News" in prompt
    assert "Published: 2026-08-30T00:00:00Z" in prompt
    assert "exactly 2 candidate IDs" in prompt
    assert "Do not include more than 2 IDs" in prompt


def test_role_body_round_selection_completes_each_rank_before_the_next():
    ordered = [
        {
            "id": "rank-zero-low",
            "role_body_round_entries": [{"view_order": 0, "rank": 0, "score": 0.5}],
        },
        {
            "id": "rank-one-high",
            "role_body_round_entries": [{"view_order": 0, "rank": 1, "score": 1.0}],
        },
        {
            "id": "rank-zero-high",
            "role_body_round_entries": [{"view_order": 1, "rank": 0, "score": 1.0}],
        },
        {"id": "global"},
    ]

    selected = GraphRAG._role_body_rounds(ordered, top_k=4)

    assert [node["id"] for node in selected] == [
        "rank-zero-high",
        "rank-zero-low",
        "rank-one-high",
        "global",
    ]


def test_transient_index_embeddings_are_not_returned_publicly():
    rag = GraphRAG(strategy="prehop")
    output = rag._without_transient_retrieval_scores(
        {
            "id": "chunk",
            "text": "evidence",
            "embedding": [1.0],
            "bridge_embeddings": [[1.0]],
            "role_body_owner_orders": [0],
            "role_body_round_entries": [{"view_order": 0, "rank": 0, "score": 1.0}],
            "role_body_owner_only": True,
        }
    )

    assert output == {"id": "chunk", "text": "evidence"}


@pytest.mark.asyncio
async def test_retrieve_searches_each_role_channel_once_in_full_mode(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")

    stage1_node = {
        "id": "n1",
        "title": "AMD_2022_10K",
        "sent_id": 1,
        "page": 10,
        "text": "Operating cash flow was 3.6 billion.",
        "rrf_score": 1.0,
        "embedding": [1.0, 0.0],
    }
    stage2_node = {
        "id": "n2",
        "title": "AMD_2022_10K",
        "sent_id": 2,
        "page": 11,
        "text": "Capital expenditures were 1.1 billion in FY2022.",
        "rrf_score": 0.9,
        "embedding": [1.0, 0.0],
    }

    async def fake_candidates(_q_text: str, query_embedding, limit: int, channel: str = "body"):
        assert query_embedding == [1.0, 0.0]
        if channel == "q_minus":
            return [dict(stage1_node)]
        if channel == "q_plus":
            return [dict(stage2_node)]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])

    _, nodes = await rag.retrieve(
        "What was AMD FY2022 free cash flow?",
        top_k=2,
    )

    ids = {n.get("id") for n in nodes}
    assert "n1" in ids
    assert "n2" in ids


@pytest.mark.asyncio
async def test_retrieval_embeds_query_once_before_parallel_channels():
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._retrieve_with_candidate_pool("one query", top_k=12)

    rag.llm.get_embedding.assert_awaited_once_with("one query")
    assert rag._hybrid_rrf_candidates.await_count == 3
    assert [call.kwargs["channel"] for call in rag._hybrid_rrf_candidates.await_args_list] == [
        "q_minus",
        "body",
        "q_plus",
    ]
    assert all(call.kwargs["query_embedding"] == [1.0, 0.0] for call in rag._hybrid_rrf_candidates.await_args_list)


@pytest.mark.asyncio
async def test_sentence_channel_uses_original_query_and_collapses_to_chunk_role(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SENTENCE_CHANNEL_ENABLED", True)
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0], [0.9], [0.8]])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._retrieve_with_candidate_pool(
        "original",
        top_k=12,
        channel_queries={"q_minus": ["minus"], "q_plus": ["plus"]},
    )

    assert [(call.args[0], call.kwargs["channel"]) for call in rag._hybrid_rrf_candidates.await_args_list] == [
        ("minus", "q_minus"),
        ("original", "body"),
        ("original", "sentence"),
        ("plus", "q_plus"),
    ]


@pytest.mark.asyncio
async def test_sentence_and_chunk_results_fuse_once_as_body_role(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SENTENCE_CHANNEL_ENABLED", True)
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])

    async def candidates(_query, *, query_embedding, limit, channel):
        _ = query_embedding, limit
        if channel in {"body", "sentence"}:
            return [{"id": "shared", "text": "evidence", "embedding": [1.0]}]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=candidates)  # type: ignore[method-assign]

    _selected, pool = await rag._retrieve_with_candidate_pool("original", top_k=1, select_final=False)

    assert pool[0]["representation_scores"] == {"body": 1.0}
    assert pool[0]["representation_score"] == 1.0
    assert [path["channel"] for path in pool[0]["retrieval_paths"]] == ["body", "sentence"]


@pytest.mark.asyncio
async def test_candidate_pool_multiplier_widens_search_not_final_evidence(monkeypatch):
    monkeypatch.setattr(RAGConfig, "CANDIDATE_POOL_MULTIPLIER", 2)
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    selected, _pool = await rag._retrieve_with_candidate_pool("one query", top_k=12)

    assert selected == []
    assert all(call.kwargs["limit"] == 24 for call in rag._hybrid_rrf_candidates.await_args_list)


@pytest.mark.asyncio
async def test_body_only_variant_searches_only_body_without_rebuilding_index(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "body_only")
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0, 0.0])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._retrieve_with_candidate_pool("one query", top_k=12)

    rag._hybrid_rrf_candidates.assert_awaited_once()
    assert rag._hybrid_rrf_candidates.await_args.kwargs["channel"] == "body"


@pytest.mark.asyncio
async def test_retrieval_records_every_direct_representation_path():
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])

    async def candidates(_query, *, query_embedding, limit, channel):
        _ = query_embedding, limit
        if channel == "q_minus":
            return [
                {
                    "id": "shared",
                    "text": "evidence",
                    "embedding": [1.0],
                    "matched_qminus_ids": ["qm-1"],
                }
            ]
        if channel == "q_plus":
            return [
                {
                    "id": "shared",
                    "text": "evidence",
                    "embedding": [1.0],
                    "matched_qplus_ids": ["qp-1"],
                }
            ]
        return []

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=candidates)  # type: ignore[method-assign]

    _, pool = await rag._retrieve_with_candidate_pool("query", top_k=1, select_final=False)

    assert pool[0]["retrieval_paths"] == [
        {"kind": "direct", "channel": "q_minus", "query_view": "query", "depth": 0},
        {"kind": "direct", "channel": "q_plus", "query_view": "query", "depth": 0},
    ]
    assert pool[0]["representation_scores"] == {"q_minus": 1.0, "q_plus": 1.0}
    assert pool[0]["representation_score"] == 2.0
    assert pool[0]["matched_qplus_ids"] == ["qp-1"]
    assert pool[0]["dependency_seed"] is True
    assert pool[0]["matched_qminus_ids"] == ["qm-1"]
    assert pool[0]["continuation_seed"] is True


@pytest.mark.asyncio
async def test_hop_target_inherits_only_the_qplus_seed_rank():
    rag = GraphRAG(strategy="prehop")
    seed = {
        "id": "seed",
        "title": "Seed",
        "sent_id": 0,
        "text": "dependency",
        "embedding": [1.0],
        "dependency_seed": True,
        "representation_score": 1.5,
        "representation_scores": {"q_minus": 1.0, "q_plus": 0.5},
    }
    rag._retrieve_with_candidate_pool = AsyncMock(return_value=([seed], [seed]))  # type: ignore[method-assign]
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._expand_frontier = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {
                "source_id": "seed",
                "id": "target",
                "title": "Target",
                "sent_id": 0,
                "text": "answer",
                "embedding": [1.0],
                "path_type": "hop",
                "bridge_embeddings": [[1.0]],
            }
        ]
    )
    captured: list[dict] = []

    async def capture_scores(_query_embedding, candidates, _top_k, **_kwargs):
        captured.extend(candidates)
        return candidates, candidates

    rag._score_and_select = AsyncMock(side_effect=capture_scores)  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=2)

    target = next(candidate for candidate in captured if candidate["id"] == "target")
    assert target["representation_score"] == 0.25


@pytest.mark.asyncio
async def test_role_body_owner_only_candidate_does_not_seed_graph_expansion(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    regular = {
        "id": "regular",
        "title": "Regular",
        "sent_id": 0,
        "text": "regular evidence",
        "embedding": [1.0],
        "representation_score": 1.0,
    }
    owner = {
        "id": "owner",
        "title": "Owner",
        "sent_id": 0,
        "text": "owned evidence",
        "embedding": [1.0],
        "representation_score": 0.0,
        "role_body_owner_only": True,
        "role_body_owner_orders": [0],
    }
    rag._retrieve_with_candidate_pool = AsyncMock(  # type: ignore[method-assign]
        return_value=([], [regular, owner])
    )
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._expand_frontier = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=2)

    assert rag._expand_frontier.await_args.args[0] == ["regular"]


@pytest.mark.asyncio
async def test_direct_target_keeps_score_and_records_graph_path():
    rag = GraphRAG(strategy="prehop")
    source = {
        "id": "source",
        "title": "Source",
        "sent_id": 0,
        "text": "dependency",
        "embedding": [1.0],
        "dependency_seed": True,
        "matched_qplus_ids": ["qp"],
        "representation_score": 1.0,
        "representation_scores": {"q_plus": 1.0},
    }
    target = {
        "id": "target",
        "title": "Target",
        "sent_id": 0,
        "text": "answer",
        "embedding": [1.0],
        "dependency_seed": False,
        "representation_score": 0.8,
    }
    rag._retrieve_with_candidate_pool = AsyncMock(  # type: ignore[method-assign]
        return_value=([], [source, target])
    )
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._expand_frontier = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {
                "source_id": "source",
                "id": "target",
                "title": "Target",
                "sent_id": 0,
                "text": "answer",
                "embedding": [1.0],
                "path_type": "hop",
                "bridge_embeddings": [[0.5]],
                "activated_question_ids": ["qp"],
            }
        ]
    )
    captured: list[dict] = []

    async def capture_scores(_query_embedding, candidates, _top_k, **_kwargs):
        captured.extend(candidates)
        return candidates, candidates

    rag._score_and_select = AsyncMock(side_effect=capture_scores)  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=2)

    enriched = next(candidate for candidate in captured if candidate["id"] == "target")
    assert enriched["representation_score"] == 0.8
    assert "bridge_embeddings" not in enriched
    assert enriched["retrieval_paths"] == [
        {
            "kind": "hop",
            "source_chunk_id": "source",
            "source_question_ids": ["qp"],
            "depth": 1,
            "edge_rank": 0,
        }
    ]


@pytest.mark.asyncio
async def test_role_aligned_views_search_only_their_matching_channels(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "single_combined")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0], [0.9], [0.8], [0.7]])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._retrieve_with_candidate_pool(
        "original",
        top_k=12,
        channel_queries={"q_minus": ["minus one", "minus two"], "q_plus": ["plus one"]},
    )

    assert rag.llm.get_embeddings.await_args.args[0] == ["original", "minus one", "minus two", "plus one"]
    assert [(call.args[0], call.kwargs["channel"]) for call in rag._hybrid_rrf_candidates.await_args_list] == [
        ("minus one", "q_minus"),
        ("minus two", "q_minus"),
        ("plus one", "q_plus"),
    ]


@pytest.mark.asyncio
async def test_query_views_fuse_once_inside_each_role(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HYPO_CHANNEL_VARIANT", "single_combined")
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    rag.llm.get_embeddings = AsyncMock(return_value=[[1.0], [0.9], [0.8], [0.7]])

    async def candidates(view, *, query_embedding, limit, channel):
        _ = query_embedding, limit
        if channel == "q_minus":
            return [{"id": "shared", "text": view, "embedding": [1.0]}]
        return [
            {
                "id": "shared",
                "text": view,
                "embedding": [1.0],
                "matched_qplus_ids": ["qp"],
            }
        ]

    rag._hybrid_rrf_candidates = AsyncMock(side_effect=candidates)  # type: ignore[method-assign]

    _selected, pool = await rag._retrieve_with_candidate_pool(
        "original",
        top_k=12,
        channel_queries={"q_minus": ["minus one", "minus two"], "q_plus": ["plus one"]},
    )

    assert pool[0]["representation_scores"] == {"q_minus": 1.0, "q_plus": 1.0}
    assert pool[0]["representation_score"] == 2.0
    assert [path["query_view"] for path in pool[0]["retrieval_paths"]] == [
        "minus one",
        "minus two",
        "plus one",
    ]


@pytest.mark.asyncio
async def test_hybrid_search_derives_modality_width_from_owner_budget():
    rag = GraphRAG(strategy="prehop")
    rag._run_channel_query = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._hybrid_rrf_candidates("query", [1.0], limit=7, channel="q_minus")
    qminus_limits = [call.args[1]["limit"] for call in rag._run_channel_query.await_args_list]
    assert qminus_limits == [21]
    assert "CALL () {" in rag._run_channel_query.await_args.args[0]

    rag._run_channel_query.reset_mock()
    await rag._hybrid_rrf_candidates("query", [1.0], limit=7, channel="body")
    body_limits = [call.args[1]["limit"] for call in rag._run_channel_query.await_args_list]
    assert body_limits == [7]
    assert "CALL () {" in rag._run_channel_query.await_args.args[0]
    assert "node.publisher AS publisher" in rag._run_channel_query.await_args.args[0]

    rag._run_channel_query.reset_mock()
    await rag._hybrid_rrf_candidates("query", [1.0], limit=7, channel="sentence")
    sentence_limits = [call.args[1]["limit"] for call in rag._run_channel_query.await_args_list]
    assert sentence_limits == [7 * RAGConfig.CHUNK_SENTENCES]
    sentence_query = rag._run_channel_query.await_args.args[0]
    assert sentence_query.count("HAS_SENTENCE") == 2


@pytest.mark.asyncio
async def test_hybrid_rrf_restores_each_modality_order_from_raw_score():
    rag = GraphRAG(strategy="prehop")
    # UNION/aggregation result order is intentionally scrambled.  Raw scores
    # establish ranks only within each modality before equal RRF.
    rag._run_channel_query = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {"id": "vector-low", "embedding": [1.0], "score": 0.1, "type": "vector"},
            {"id": "text-low", "embedding": [1.0], "score": 1.0, "type": "text"},
            {"id": "vector-high", "embedding": [1.0], "score": 0.9, "type": "vector"},
            {"id": "text-high", "embedding": [1.0], "score": 9.0, "type": "text"},
        ]
    )

    nodes = await rag._hybrid_rrf_candidates("query", [1.0], limit=4, channel="body")

    assert [node["id"] for node in nodes] == [
        "text-high",
        "vector-high",
        "text-low",
        "vector-low",
    ]
    assert [node["rrf_score"] for node in nodes] == [1.0, 1.0, 0.5, 0.5]


@pytest.mark.asyncio
async def test_qplus_hybrid_preserves_exact_question_ids_across_modalities():
    rag = GraphRAG(strategy="prehop")
    rag._run_channel_query = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {
                "id": "owner",
                "embedding": [1.0],
                "score": 0.9,
                "type": "vector",
                "matched_question_ids": ["qp-vector", "qp-shared"],
            },
            {
                "id": "owner",
                "embedding": [1.0],
                "score": 9.0,
                "type": "text",
                "matched_question_ids": ["qp-text", "qp-shared"],
            },
        ]
    )

    nodes = await rag._hybrid_rrf_candidates("query", [1.0], limit=1, channel="q_plus")

    assert nodes[0]["matched_qplus_ids"] == ["qp-shared", "qp-text", "qp-vector"]
    assert "matched_question_ids" not in nodes[0]
    query = rag._run_channel_query.await_args.args[0]
    assert query.count("collect(DISTINCT node.id) AS matched_question_ids") == 2


@pytest.mark.asyncio
async def test_only_query_matched_dependency_seeds_can_initiate_hop_traversal():
    rag = GraphRAG(strategy="prehop")
    direct = {
        "id": "direct",
        "title": "Direct",
        "sent_id": 0,
        "text": "direct",
        "dependency_seed": False,
    }
    dependency = {
        "id": "dependency",
        "title": "Dependency",
        "sent_id": 0,
        "text": "dependency",
        "dependency_seed": True,
        "matched_qplus_ids": ["qp-exact"],
    }
    rag._retrieve_with_candidate_pool = AsyncMock(  # type: ignore[method-assign]
        return_value=([direct], [direct, dependency])
    )
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._expand_frontier = AsyncMock(return_value=[])  # type: ignore[method-assign]
    rag._score_and_select = AsyncMock(return_value=([direct], [direct]))  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=1)

    assert rag._retrieve_with_candidate_pool.await_args.kwargs["select_final"] is False
    rag._expand_frontier.assert_awaited_once()
    assert set(rag._expand_frontier.await_args.args[0]) == {"direct", "dependency"}
    assert rag._expand_frontier.await_args.args[2] == {"dependency": {"qp-exact"}}


@pytest.mark.asyncio
async def test_exact_qplus_activation_passes_only_matched_question_ids(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HOP_EDGE_FILTER", "none")
    monkeypatch.setattr(RAGConfig, "QPLUS_HOP_ACTIVATION", "exact")
    rag = GraphRAG(strategy="prehop")
    session = AsyncMock()
    session.run = AsyncMock(return_value=_AsyncRecords([]))
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None
    rag.neo4j.driver.session = MagicMock(return_value=session_context)

    await rag._expand_frontier(
        ["source", "direct"],
        set(),
        {"source": {"qp-matched-b", "qp-matched-a"}},
    )

    query, parameters = session.run.await_args.args
    assert "question_id IN coalesce($hop_source_question_ids[src.id], [])" in query
    assert "q.id IN coalesce($hop_source_question_ids[src.id], [])" in query
    assert parameters["hop_source_ids"] == ["source"]
    assert parameters["hop_source_question_ids"] == {"source": ["qp-matched-a", "qp-matched-b"]}


@pytest.mark.asyncio
async def test_linked_continuation_uses_only_matched_qminus_ids(monkeypatch):
    monkeypatch.setattr(RAGConfig, "QUESTION_SCHEMA", "linked_v2")
    monkeypatch.setattr(RAGConfig, "CONTINUATION_EDGES_ENABLED", True)
    monkeypatch.setattr(RAGConfig, "HOP_EDGE_FILTER", "none")
    rag = GraphRAG(strategy="prehop")
    session = AsyncMock()
    session.run = AsyncMock(return_value=_AsyncRecords([]))
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None
    rag.neo4j.driver.session = MagicMock(return_value=session_context)

    await rag._expand_frontier(
        ["source"],
        set(),
        {},
        {"source": {"qm-b", "qm-a"}},
        [1.0],
        12,
    )

    query, parameters = session.run.await_args.args
    assert "ANSWER_ANCHOR" in query
    assert "MENTIONED_IN" in query
    assert "matched_q.id IN coalesce($continuation_source_question_ids[src.id], [])" in query
    assert "related.source <> src.source" in query
    assert "vector.similarity.cosine" in query
    assert parameters["continuation_source_ids"] == ["source"]
    assert parameters["continuation_source_question_ids"] == {"source": ["qm-a", "qm-b"]}


@pytest.mark.asyncio
async def test_reciprocal_hop_filter_is_read_only_and_uses_reverse_qplus_ann(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HOP_EDGE_FILTER", "reciprocal")
    rag = GraphRAG(strategy="prehop")
    session = AsyncMock()
    session.run = AsyncMock(return_value=_AsyncRecords([]))
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None
    rag.neo4j.driver.session = MagicMock(return_value=session_context)

    await rag._expand_frontier(["source"], set(), {"source": {"qp-exact"}})

    query, parameters = session.run.await_args.args
    assert "ANSWERED_BY" in query
    assert "db.index.vector.queryNodes" in query
    assert "reverse_best.id = edge_qplus.id" in query
    assert "edge_qplus.id IN coalesce($hop_source_question_ids[src.id], [])" in query
    assert all(keyword not in query for keyword in ("CREATE", "MERGE", "SET ", "DELETE"))
    assert parameters["qplus_vector_index"] == rag.q_plus_vector_index
    assert "reciprocal_source_question_ids" not in query


@pytest.mark.asyncio
async def test_offline_reciprocal_filter_uses_materialized_ids_without_ann(monkeypatch):
    monkeypatch.setattr(RAGConfig, "HOP_EDGE_FILTER", "reciprocal_offline")
    monkeypatch.setattr(RAGConfig, "QPLUS_HOP_ACTIVATION", "exact")
    rag = GraphRAG(strategy="prehop")
    session = AsyncMock()
    session.run = AsyncMock(return_value=_AsyncRecords([]))
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None
    rag.neo4j.driver.session = MagicMock(return_value=session_context)

    await rag._expand_frontier(["source"], set(), {"source": {"qp-exact"}})

    query, parameters = session.run.await_args.args
    assert "reciprocal_source_question_ids" in query
    assert "db.index.vector.queryNodes" not in query
    assert parameters["qplus_hop_activation"] == "exact"


@pytest.mark.asyncio
async def test_graph_search_scores_candidates_only_after_expansion(monkeypatch):
    monkeypatch.setattr(RAGConfig, "SOURCE_SELECTION_VARIANT", "global")
    rag = GraphRAG(strategy="prehop")
    seed = {
        "id": "seed",
        "title": "Seed",
        "sent_id": 0,
        "page": 0,
        "text": "seed",
        "embedding": [1.0],
        "dependency_seed": False,
        "representation_score": 1.0,
    }
    rag.llm.get_embedding = AsyncMock(return_value=[1.0])
    rag._hybrid_rrf_candidates = AsyncMock(return_value=[seed])  # type: ignore[method-assign]
    rag._expand_frontier = AsyncMock(return_value=[])  # type: ignore[method-assign]
    original = rag._score_and_select
    rag._score_and_select = AsyncMock(side_effect=original)  # type: ignore[method-assign]

    await rag.graph_search(["query"], depth=1, top_k=1)

    assert rag._score_and_select.await_count == 1


@pytest.mark.asyncio
async def test_build_graph_stores_generated_q_plus_without_heuristic_filter(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "off")
    rag = GraphRAG(strategy="prehop")
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    rag.vector_dimensions = 1
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts, encoding_type="document": [[0.1] for _ in texts])

    # Storage keeps generated non-empty Q+ values without applying a
    # domain/keyword heuristic after the model's schema-validated response.
    knowledge = {
        "chunks": [
            {
                "text": "Consolidated Statements of Cash Flows for fiscal year 2022.",
                "title": "AMD_2022_10K",
                "sent_id": 0,
                "page": 1,
                "q_minus": ["For AMD FY2022, what was operating cash flow?"],
                "q_plus": [
                    "For FY2022, what happened?",
                    "For AMD FY2022 cash flow statement, what was operating cash flow?",
                    "For FY2022, what happened?",
                    "",
                ],
            }
        ]
    }

    await rag.build_graph(knowledge, source="unit_test", document_filename="AMD_2022_10K.txt")

    assert rag._pending_batch, "Expected build_graph to enqueue at least one batch item."
    payload = rag._pending_batch[-1]["data"][0]
    assert [question["text"] for question in payload["q_plus"]] == [
        "For FY2022, what happened?",
        "For AMD FY2022 cash flow statement, what was operating cash flow?",
    ]
    assert len({question["id"] for question in payload["q_plus"]}) == 2
    assert all(question["query_embedding"] == [0.1] for question in payload["q_plus"])


@pytest.mark.asyncio
async def test_build_graph_stores_optional_source_metadata(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "off")
    rag = GraphRAG(strategy="prehop")
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    rag.vector_dimensions = 1
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts, encoding_type="document": [[0.1] for _ in texts])
    knowledge = {
        "chunks": [
            {
                "text": "Evidence.",
                "title": "Article",
                "sent_id": 0,
                "page": 1,
                "q_minus": [],
                "q_plus": [],
                "publisher": "Example News",
                "published_at": "2026-08-30T00:00:00Z",
                "author": "Author",
                "category": "news",
                "url": "https://example.com/article",
            }
        ]
    }

    await rag.build_graph(knowledge, source="Article.txt", document_filename="Article.txt")

    payload = rag._pending_batch[-1]["data"][0]
    assert payload["publisher"] == "Example News"
    assert payload["published_at"] == "2026-08-30T00:00:00Z"
    assert payload["url"] == "https://example.com/article"


@pytest.mark.asyncio
async def test_build_graph_stores_grounded_question_metadata(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "off")
    rag = GraphRAG(strategy="prehop")
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    rag.vector_dimensions = 1
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts, encoding_type="document": [[0.1] for _ in texts])
    knowledge = {
        "chunks": [
            {
                "text": "Acme was founded by Kim.",
                "title": "Acme",
                "sent_id": 0,
                "page": 1,
                "q_minus": [
                    {
                        "text": "Who founded Acme?",
                        "answer": "Kim",
                        "grounding_quote": "Acme was founded by Kim.",
                        "anchor_entities": ["Acme", "Kim"],
                        "question_schema": "grounded_v1",
                    }
                ],
                "q_plus": [
                    {
                        "text": "Where was Acme incorporated?",
                        "grounding_quote": "Acme was founded by Kim.",
                        "anchor_entities": ["Acme"],
                        "missing_information": "The jurisdiction.",
                        "question_schema": "grounded_v1",
                    }
                ],
            }
        ]
    }

    await rag.build_graph(knowledge, source="unit_test", document_filename="Acme.txt")

    payload = rag._pending_batch[-1]["data"][0]
    assert payload["q_minus"][0]["answer"] == "Kim"
    assert payload["q_plus"][0]["missing_information"] == "The jurisdiction."
    assert payload["q_plus"][0]["question_schema"] == "grounded_v1"


@pytest.mark.asyncio
async def test_build_graph_materializes_sentence_children_only_when_enabled(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_CACHE", "off")
    monkeypatch.setattr(RAGConfig, "SENTENCE_CHANNEL_ENABLED", True)
    rag = GraphRAG(strategy="prehop")
    rag._ensure_index_ready = AsyncMock(return_value=None)  # type: ignore[method-assign]
    rag.vector_dimensions = 1
    rag.llm.get_embeddings = AsyncMock(side_effect=lambda texts, encoding_type="document": [[0.1] for _ in texts])
    knowledge = {
        "chunks": [
            {
                "text": "Acme was founded by Kim. It is based in Seoul.",
                "title": "Acme",
                "sent_id": 0,
                "page": 1,
                "q_minus": [],
                "q_plus": [],
            }
        ]
    }

    await rag.build_graph(knowledge, source="unit_test", document_filename="Acme.txt")

    sentences = rag._pending_batch[-1]["data"][0]["sentences"]
    assert [sentence["text"] for sentence in sentences] == [
        "Acme was founded by Kim.",
        "It is based in Seoul.",
    ]
    assert [sentence["ordinal"] for sentence in sentences] == [0, 1]
    assert len({sentence["id"] for sentence in sentences}) == 2
    assert all(sentence["embedding"] == [0.1] for sentence in sentences)


@pytest.mark.asyncio
async def test_qplus_similarity_is_not_an_indexing_candidate_channel():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        if channel == "q_plus":
            return [
                {
                    "source_chunk_id": wave[0]["id"],
                    "source_question_id": wave[0]["questions"][0]["id"],
                    "target_id": "same-need-only",
                    "target_question_id": "qp-target",
                    "score": 0.99,
                }
            ]
        return []

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "manual-2022.txt",
                "questions": [{"id": "qp-source", "query_embedding": [1.0]}],
            }
        ],
    )

    assert edges == []


@pytest.mark.asyncio
async def test_hop_ann_pool_is_sized_per_source_not_by_corpus_max(monkeypatch):
    rag = GraphRAG(strategy="prehop")
    rag.retry_query = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rag._find_hop_candidates_batch(
        [
            {
                "id": "short-source-chunk",
                "source": "short.txt",
                "ann_pools": {"body": 22},
                "questions": [{"id": "q-short", "query_embedding": [1.0]}],
            },
            {
                "id": "long-source-chunk",
                "source": "long.txt",
                "ann_pools": {"body": 180},
                "questions": [{"id": "q-long", "query_embedding": [1.0]}],
            },
        ],
        "body",
    )

    parameters = rag.retry_query.await_args.args[1]
    assert [item["ann_pool"] for item in parameters["source_questions"]] == [22, 180]
    assert "candidate_limit" not in parameters


@pytest.mark.asyncio
async def test_full_hop_policy_follows_qminus_owner_without_second_body_search():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        source = {
            "source_chunk_id": wave[0]["id"],
            "source_question_id": wave[0]["questions"][0]["id"],
        }
        if channel == "q_minus":
            return [{**source, "target_id": "manual-2023", "target_question_id": "qm-2023", "score": 0.8}]
        return []

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "manual-2022.txt",
                "questions": [{"id": "qp-source", "query_embedding": [1.0]}],
            }
        ],
    )

    assert [edge["tgt_id"] for edge in edges] == ["manual-2023"]
    assert [call.args[1] for call in rag._find_hop_candidates_batch.await_args_list] == ["q_minus"]


@pytest.mark.asyncio
async def test_each_individual_qplus_keeps_one_direct_evidence_target():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        candidates = [
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-first",
                "target_id": "first-best",
                "target_question_id": "qm-first" if channel == "q_minus" else None,
                "target_text": "First target answers the first question.",
                "target_title": "First",
                "score": 0.99,
            },
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": "q-second",
                "target_id": "second-best",
                "target_question_id": "qm-second" if channel == "q_minus" else None,
                "target_text": "Second target answers the second question.",
                "target_title": "Second",
                "score": 0.70,
            },
        ]
        return candidates

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]
    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "source.txt",
                "questions": [
                    {"id": "q-first", "query_embedding": [1.0]},
                    {"id": "q-second", "query_embedding": [1.0]},
                ],
            }
        ],
    )

    assert {edge["tgt_id"] for edge in edges} == {"first-best", "second-best"}
    assert all(edge["direct_channels"] == ["q_minus"] for edge in edges)


@pytest.mark.asyncio
async def test_questions_sharing_one_target_keep_all_provenance():
    rag = GraphRAG(strategy="prehop")

    async def fake_candidates(wave, channel):
        return [
            {
                "source_chunk_id": wave[0]["id"],
                "source_question_id": question_id,
                "target_id": "shared-target",
                "target_question_id": f"qm-{question_id}" if channel == "q_minus" else None,
                "target_text": "One passage answers both questions.",
                "target_title": "Shared",
                "score": 0.9,
            }
            for question_id in ("q-first", "q-second")
        ]

    rag._find_hop_candidates_batch = AsyncMock(side_effect=fake_candidates)  # type: ignore[method-assign]

    edges = await rag._process_hop_wave(
        [
            {
                "id": "source-chunk",
                "source": "source.txt",
                "questions": [
                    {"id": "q-first", "text": "First question?", "query_embedding": [1.0]},
                    {"id": "q-second", "text": "Second question?", "query_embedding": [1.0]},
                ],
            }
        ]
    )

    assert len(edges) == 1
    assert edges[0]["source_question_ids"] == ["q-first", "q-second"]


@pytest.mark.asyncio
async def test_hop_and_provenance_edges_are_written_in_one_query():
    rag = GraphRAG(strategy="prehop")
    rag.retry_query = AsyncMock(return_value=[])
    edge = {
        "src_id": "source",
        "tgt_id": "target",
        "direct_channels": ["q_minus"],
        "q_minus_match": {"raw_score": 0.8},
        "body_match": None,
        "q_minus_matches": [{"source_question_id": "q-plus", "target_question_id": "q-minus", "raw_score": 0.8}],
        "body_matches": [],
        "source_question_ids": ["q-plus"],
        "source_question_texts": ["Which evidence is missing?"],
        "construction_mode": "qplus_to_qminus_owner",
    }

    await rag._flush_hop_edges([edge])

    rag.retry_query.assert_awaited_once()
    assert rag.retry_query.await_args.args[1] == {"edges": [edge]}


def test_indexing_has_no_tuned_hop_fusion_or_width_knobs():
    for name in (
        "RRF_K_CONSTANT",
        "HOP_LINK_LIMIT",
        "HOP_CANDIDATE_LIMIT",
        "HOP_ANN_POOL",
        "HOP_SAME_NEED_WEIGHT",
    ):
        assert not hasattr(RAGConfig, name)


def test_benchmark_does_not_reference_removed_hop_tuning_knobs():
    benchmark_source = Path("cli/benchmark.py").read_text(encoding="utf-8")
    for name in (
        "HOP_LINK_LIMIT",
        "HOP_CANDIDATE_LIMIT",
        "HOP_ANN_POOL",
        "HOP_SAME_NEED_WEIGHT",
    ):
        assert name not in benchmark_source
