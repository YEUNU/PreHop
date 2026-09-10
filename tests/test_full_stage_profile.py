from scripts.analyze_full_stage_profile import _point_estimates, _rows_with_trace_timers


def test_stage_profile_reads_separated_timers_from_interaction_trace() -> None:
    rows = {"q1": {"query_id": "q1", "latency": 9.0}}
    traces = {
        "q1": {
            "interaction_trace": [
                {
                    "step": "retrieve",
                    "retrieve_ms": 2000.0,
                    "graph_expand_ms": 300.0,
                    "deterministic_score_ms": 40.0,
                    "candidate_order_ms": 4000.0,
                },
                {"step": "synthesis", "synthesis_ms": 500.0},
            ]
        }
    }

    profiled = _rows_with_trace_timers(rows, traces)

    assert profiled["q1"] == {
        "query_id": "q1",
        "latency": 9.0,
        "retrieve_ms": 2000.0,
        "graph_expand_ms": 300.0,
        "deterministic_score_ms": 40.0,
        "candidate_order_ms": 4000.0,
        "synthesis_ms": 500.0,
    }


def test_stage_profile_rejects_missing_separated_timer() -> None:
    rows = {"q1": {"query_id": "q1"}}
    traces = {"q1": {"interaction_trace": [{"step": "retrieve", "retrieve_ms": 1.0}]}}

    try:
        _rows_with_trace_timers(rows, traces)
    except ValueError as exc:
        assert "Missing separated stage timers" in str(exc)
    else:
        raise AssertionError("Expected missing timers to fail")


def test_stage_profile_counts_current_generation_stages_once() -> None:
    rows = {"q1": {"query_id": "q1"}}
    traces = {
        "q1": {
            "interaction_trace": [
                {
                    "step": "retrieve",
                    "retrieve_ms": 2.0,
                    "graph_expand_ms": 3.0,
                    "deterministic_score_ms": 4.0,
                    "candidate_order_ms": 5.0,
                },
                {"step": "synthesis", "synthesis_ms": 6.0},
            ]
        }
    }

    profiled = _rows_with_trace_timers(rows, traces)

    profiled["q1"].update(answer_em=1.0, answer_f1=1.0, latency=0.02)
    estimates = _point_estimates(profiled, ["q1"])
    assert abs(estimates["generation_stage_seconds"]["mean"] - 0.011) < 1e-12
    assert abs(estimates["accounted_stage_seconds"]["mean"] - 0.020) < 1e-12
