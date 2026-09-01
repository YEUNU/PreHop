from __future__ import annotations

import json

from scripts.frozen_trace_alignment import align_frozen_traces_to_gold
from scripts.replay_frozen_candidate_order import _support_metrics


def test_support_metrics_use_frozen_paragraph_ids() -> None:
    trace = {
        "candidates": [
            {"node_id": "a", "paragraph_id": "musique:p1", "text": "one"},
            {"node_id": "b", "paragraph_id": "musique:p2", "text": "two"},
        ]
    }
    metrics = _support_metrics(
        trace,
        ["a", "b"],
        {"evidence_paragraph_ids": ["musique:p1", "musique:p3"]},
    )

    assert metrics == {
        "paragraph_support_precision": 0.5,
        "paragraph_support_recall": 0.5,
        "paragraph_support_f1": 0.5,
    }


def test_duplicate_query_text_is_aligned_through_generated_query_views(tmp_path) -> None:
    query_path = tmp_path / "queries.json"
    query_path.write_text(
        json.dumps(
            [
                {"_id": "q2", "query": "Who is A?", "evidence_paragraph_ids": ["musique:p2"]},
                {"_id": "q1", "query": "Who is A?", "evidence_paragraph_ids": ["musique:p1"]},
            ]
        )
    )
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "evaluation_scope": "full_benchmark",
                "details": [
                    {"query_id": "q1", "query": "Who is A?"},
                    {"query_id": "q2", "query": "Who is A?"},
                ],
            }
        )
    )
    (tmp_path / "benchmark.traces.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "idx": 1,
                    "query": "Who is A?",
                    "interaction_trace": [
                        {"step": "query_rewrite", "output": {"q_minus": ["view one"], "q_plus": []}}
                    ],
                },
                {
                    "idx": 2,
                    "query": "Who is A?",
                    "interaction_trace": [
                        {"step": "query_rewrite", "output": {"q_minus": ["view two"], "q_plus": []}}
                    ],
                },
            ]
        )
        + "\n"
    )
    traces = [
        {
            "query": "WHO IS A",
            "candidates": [
                {"retrieval_paths": [{"kind": "direct", "channel": "q_minus", "query_view": "view two"}]}
            ],
        },
        {
            "query": "who is a?",
            "candidates": [
                {"retrieval_paths": [{"kind": "direct", "channel": "q_minus", "query_view": "view one"}]}
            ],
        },
    ]

    aligned = align_frozen_traces_to_gold(traces, benchmark_path, query_path)

    assert [row["_id"] for row in aligned] == ["q2", "q1"]
