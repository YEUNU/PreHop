import json

import pytest

from core.strategy_registry import PRIMARY_STRATEGIES
from scripts import paper_gate_ledger


def _evidence(path, *, status="canary_passed", **values):
    if values.get("verdict") == "GO":
        from core.paper_compatibility import without_realized_runtime
        values["reviewed_configuration_sha256"] = paper_gate_ledger.identity_sha256(without_realized_runtime(paper_gate_ledger._context()))
    path.write_text(json.dumps({"status": status, **values}), encoding="utf-8")
    return path


def test_live_gate_ledger_is_ordered_content_bound_and_runtime_setup_aware(tmp_path, monkeypatch):
    context = {
        "code": {"source_tree_sha256": "code"},
        "verifier_sha256": "verifier",
        "runtime": {"phase": "before"},
        "targets": ["target"],
    }
    monkeypatch.setattr(paper_gate_ledger, "ROOT", tmp_path)
    monkeypatch.setattr(paper_gate_ledger, "_context", lambda: dict(context))
    ledger = tmp_path / "ledger.json"
    paper_gate_ledger.initialize(ledger, "campaign")

    report = _evidence(tmp_path / "review.json", verdict="GO")
    static = _evidence(tmp_path / "static.json", kind="independent_static_review", reviewer="independent",
                       report={"path": "review.json", "sha256": paper_gate_ledger.sha256_file(report)})
    paper_gate_ledger.record(ledger, "static_go", static)
    context["runtime"] = {"phase": "installed"}
    receipt = _evidence(tmp_path / "setup-receipt.json", stage="runtime_setup", argv=["setup", "--primary"],
                        exit_code=0, checks={name: "passed" for name in PRIMARY_STRATEGIES})
    setup = _evidence(tmp_path / "setup.json", schema_version=1, stage="runtime_setup",
                      receipt={"path": receipt.name, "sha256": paper_gate_ledger.sha256_file(receipt)})
    paper_gate_ledger.record(ledger, "runtime_setup", setup)
    assert json.loads(ledger.read_text())["context"]["runtime"] == {"phase": "installed"}
    with pytest.raises(RuntimeError, match="incomplete"):
        paper_gate_ledger.verify(ledger, "campaign")
    static.write_text('{"status":"failed"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or changed"):
        paper_gate_ledger.verify(ledger, "campaign")


@pytest.mark.parametrize("stage", ["full_target_admitted", "cold_canary_16", "one_query_matrix_16"])
def test_live_gate_rejects_self_asserted_status_or_target_names(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(paper_gate_ledger, "ROOT", tmp_path)
    evidence = {"status": "admitted" if stage == "full_target_admitted" else "canary_passed",
                "schema_version": 1, "stage": stage,
                "targets": [f"{dataset}/{strategy}" for strategy in PRIMARY_STRATEGIES for dataset in paper_gate_ledger.DATASETS]}
    with pytest.raises(RuntimeError):
        paper_gate_ledger._validate_evidence(stage, tmp_path / "evidence.json", evidence)


def test_live_gate_ledger_rejects_out_of_order_or_incomplete_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_gate_ledger, "ROOT", tmp_path)
    monkeypatch.setattr(paper_gate_ledger, "_context", lambda: {"stable": True})
    ledger = tmp_path / "ledger.json"
    paper_gate_ledger.initialize(ledger, "campaign")
    evidence = _evidence(tmp_path / "evidence.json")
    with pytest.raises(RuntimeError, match="prerequisite"):
        paper_gate_ledger.record(ledger, "preflight_16", evidence)


def test_static_no_go_never_unlocks_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_gate_ledger, "ROOT", tmp_path)
    report = _evidence(tmp_path / "review.json", verdict="NO-GO")
    with pytest.raises(RuntimeError, match="did not conclude GO"):
        paper_gate_ledger._validate_evidence("static_go", tmp_path / "static.json", {
            "status": "canary_passed", "kind": "independent_static_review", "reviewer": "independent",
            "report": {"path": report.name, "sha256": paper_gate_ledger.sha256_file(report)},
        })


def test_setup_producer_checks_readiness_before_process_and_emits_accepted_schema(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(paper_gate_ledger, "ROOT", tmp_path)
    monkeypatch.setattr(paper_gate_ledger, "_context", lambda: {"stable": True})
    ledger = tmp_path / "ledger.json"
    paper_gate_ledger.initialize(ledger, "campaign")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    with pytest.raises(RuntimeError, match="prerequisite"):
        paper_gate_ledger.execute_stage(ledger, "campaign", "runtime_setup")
    assert calls == []
    report = _evidence(tmp_path / "review.json", verdict="GO")
    static = _evidence(tmp_path / "static.json", kind="independent_static_review", reviewer="independent",
                       report={"path": report.name, "sha256": paper_gate_ledger.sha256_file(report)})
    paper_gate_ledger.record(ledger, "static_go", static)
    paper_gate_ledger.execute_stage(ledger, "campaign", "runtime_setup")
    assert calls == [["bash", "scripts/setup_official_baselines.sh", "--primary"]]
    payload = json.loads(ledger.read_text())
    assert payload["stages"]["runtime_setup"]["status"] == "canary_passed"
    paper_gate_ledger.execute_stage(ledger, "campaign", "preflight_16")
    assert len(calls) == 17
    payload = json.loads(ledger.read_text())
    assert payload["stages"]["preflight_16"]["status"] == "canary_passed"
