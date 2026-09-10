import json

from scripts import paper_gate_ledger as ledger


def test_recording_does_not_require_approval_or_matching_context(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    monkeypatch.setattr(ledger, "_context", lambda: {"revision": "old"})
    path = tmp_path / "ledger.json"
    ledger.initialize(path, "campaign")
    monkeypatch.setattr(ledger, "_context", lambda: {"revision": "new"})
    evidence = tmp_path / "result.json"
    evidence.write_text(json.dumps({"status": "completed", "queries": 2}))
    ledger.record(path, ledger.STAGES[-1], evidence)
    result = json.loads(path.read_text())
    assert result["status"] == "completed"
    assert result["stages"][ledger.STAGES[-1]]["verification"] == "not_checked"
