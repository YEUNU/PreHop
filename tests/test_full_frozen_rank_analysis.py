from scripts.analyze_full_frozen_rank_variants import _orders


def test_frozen_rank_variants_reproduce_captured_fused_order():
    row = {
        "query": "Which evidence is needed?",
        "canonical_node_ids": ["a", "b", "c"],
        "candidates": [
            {
                "node_id": "a",
                "final_score": 0.9,
                "similarity_score": 0.9,
                "bridge_similarity_score": None,
                "representation_score": 0.3,
            },
            {
                "node_id": "b",
                "final_score": 0.8,
                "similarity_score": 0.8,
                "bridge_similarity_score": 0.95,
                "representation_score": 0.8,
            },
            {
                "node_id": "c",
                "final_score": 0.7,
                "similarity_score": 0.7,
                "bridge_similarity_score": None,
                "representation_score": 0.1,
            },
        ],
    }

    variants = _orders(row)

    assert [item["node_id"] for item in variants["fused"]] == ["a", "b", "c"]
    assert [item["node_id"] for item in variants["semantic_only"]] == ["a", "b", "c"]
    assert [item["node_id"] for item in variants["representation_only"]] == ["b", "a", "c"]
    assert [item["node_id"] for item in variants["bridge_only_fused"]] == ["b", "a", "c"]


def test_frozen_rank_analysis_rejects_nonreproducible_fused_order():
    row = {
        "query": "Which evidence is needed?",
        "canonical_node_ids": ["b", "a"],
        "candidates": [
            {
                "node_id": "a",
                "final_score": 0.9,
                "similarity_score": 0.9,
                "representation_score": 0.9,
            },
            {
                "node_id": "b",
                "final_score": 0.1,
                "similarity_score": 0.1,
                "representation_score": 0.1,
            },
        ],
    }

    try:
        _orders(row)
    except ValueError as exc:
        assert "cannot be reproduced" in str(exc)
    else:
        raise AssertionError("Expected a nonreproducible captured order to fail")


def test_frozen_rank_variants_reconstruct_graph_decay_from_source_channel_scores():
    row = {
        "query": "Which evidence is needed?",
        "canonical_node_ids": ["z-target", "a-source"],
        "candidates": [
            {
                "node_id": "z-target",
                "final_score": 0.9,
                "similarity_score": 0.9,
                "representation_score": 0.1,
                "representation_scores": {},
                "retrieval_paths": [
                    {
                        "kind": "hop",
                        "source_chunk_id": "a-source",
                        "depth": 1,
                    }
                ],
            },
            {
                "node_id": "a-source",
                "final_score": 0.1,
                "similarity_score": 0.1,
                "representation_score": 0.2,
                "representation_scores": {"q_plus": 0.2},
                "retrieval_paths": [{"kind": "direct", "channel": "q_plus", "depth": 0}],
            },
        ],
    }

    variants = _orders(row)

    assert [item["node_id"] for item in variants["graph_decay_zero_fused"]] == [
        "a-source",
        "z-target",
    ]
    assert [item["node_id"] for item in variants["graph_decay_one_fused"]] == [
        "z-target",
        "a-source",
    ]
