import json

from scripts.merge_index_matrix_runs import merge
from scripts.run_index_matrix import (
    _aggregate_attempt_history,
    _estimate_matrix_eta,
    _log_progress,
    _read_attempt_journal,
)


def _result(run_id: str, status: str, elapsed: float, phase: float) -> dict:
    return {
        "target": "2wikimultihopqa__prehop",
        "dataset": "2wikimultihopqa",
        "strategy": "prehop",
        "run_id": run_id,
        "attempt": 1,
        "status": status,
        "input_file_count": 10,
        "elapsed_seconds": elapsed,
        "phase_timing_seconds": {"document_pipeline_seconds": phase},
        "finished_at": elapsed,
    }


def test_attempt_history_adds_phase_fragments():
    history = [
        {"elapsed_seconds": 10, "runtime_stage_timing_seconds": {"document_pipeline_seconds": 7}},
        {"elapsed_seconds": 4, "runtime_stage_timing_seconds": {"document_pipeline_seconds": 2, "hop_build_seconds": 1}},
    ]
    assert _aggregate_attempt_history(history) == {
        "document_pipeline_seconds": 9.0,
        "hop_build_seconds": 1.0,
    }


def test_merge_sums_interrupted_and_resumed_fragments(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "attempt_journal.jsonl").write_text(
        json.dumps({"event": "attempt_finished", "result": _result("first", "interrupted", 10, 7)}) + "\n",
        encoding="utf-8",
    )
    (second / "attempt_journal.jsonl").write_text(
        json.dumps({"event": "attempt_finished", "result": _result("second", "complete", 4, 3)}) + "\n",
        encoding="utf-8",
    )
    [merged] = merge([first, second])
    assert merged["status"] == "complete"
    assert merged["fragment_count"] == 2
    assert merged["elapsed_seconds"] == 14
    assert merged["phase_timing_seconds"] == {"document_pipeline_seconds": 10.0}


def test_live_log_progress_recognizes_ms_phase(tmp_path):
    log = tmp_path / "target.log"
    log.write_text("extract graph progress: 12/100\n", encoding="utf-8")
    assert _log_progress(log) == {
        "phase": "extract_graph",
        "completed": 12,
        "total": 100,
        "marker": "progress",
    }


def test_resume_journal_loads_attempt_fragments_and_highest_attempt(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = _result("run", "interrupted", 10, 7)
    row["target"] = "multihoprag__ms_graphrag"
    row["attempt"] = 2
    (run_dir / "attempt_journal.jsonl").write_text(
        json.dumps({"event": "attempt_started", "target": row["target"], "attempt": 2})
        + "\n"
        + json.dumps({"event": "attempt_finished", "result": row})
        + "\n",
        encoding="utf-8",
    )
    history, highest = _read_attempt_journal(run_dir)
    assert highest == {"multihoprag__ms_graphrag": 2}
    assert history["multihoprag__ms_graphrag"][0]["status"] == "interrupted"


def test_eta_includes_future_barrier_estimates():
    pending = type("Target", (), {"strategy": "ms_graphrag"})()
    future = type("Target", (), {"strategy": "hoprag"})()
    eta, components = _estimate_matrix_eta(
        {
            "parallel_limit": 3,
            "results": [
                {"strategy": "ms_graphrag", "status": "complete", "elapsed_seconds": 30},
                {"strategy": "hoprag", "status": "complete", "elapsed_seconds": 20},
            ],
            "pending": [pending],
            "future_targets": [future],
        },
        [10],
    )
    assert eta == 60
    assert components["pending_estimated_seconds"] == 50


def test_cold_eta_uses_strategy_specific_historical_samples():
    pending = type("Target", (), {"strategy": "ms_graphrag"})()
    future = type("Target", (), {"strategy": "hoprag"})()
    eta, components = _estimate_matrix_eta(
        {
            "parallel_limit": 3,
            "max_generation_parallel": 3,
            "results": [],
            "pending": [pending],
            "future_targets": [future],
            "historical_strategy_samples": {
                "ms_graphrag": [30],
                "hoprag": [90],
            },
        },
        [],
    )
    assert eta == 120.0
    assert components["historical_estimated_targets"] == 2
    assert components["unknown_targets"] == 0
