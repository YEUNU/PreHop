import bz2
import importlib.util
import io
import json
import random
import sqlite3
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.datasets.prepare_hotpotqa import build
from utils.hotpotqa import METRICS, project_sentences, score


def fixture_files(tmp_path):
    rows = [{"id": 1, "title": "Article A", "text": ["Alpha is here. ", "Beta is there."]},
            {"id": 2, "title": "Unrelated", "text": ["Not in any dev context."]}]
    archive = tmp_path / "wiki.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        data = bz2.compress("\n".join(json.dumps(r) for r in rows).encode())
        member = tarfile.TarInfo("../../never-extracted.bz2")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([{"_id": "q", "question": "Where?", "answer": "here", "type": "bridge",
                                 "supporting_facts": [["Article A", 0]],
                                 "context": [["LEAK", ["Must never be indexed."]]]}]))
    return archive, gold


def test_fullwiki_preparation_independent_of_gold_context(tmp_path):
    archive, gold = fixture_files(tmp_path)
    corpus = tmp_path / "hotpotqa_corpus"
    queries = tmp_path / "queries.json"
    manifest = build(archive, gold, corpus, queries, verify_official=False)
    assert manifest["paragraph_count"] == 2
    assert "Not in any dev context." in (corpus / "hotpotqa_2.txt").read_text()
    assert not (tmp_path.parent / "never-extracted.bz2").exists()
    assert project_sentences([{"source": "hotpotqa_1", "text": "Beta is there."}], corpus / "sentences.sqlite3") == [["Article A", 1]]
    assert project_sentences([{"title": "Article A", "text": "Beta is"}], corpus / "sentences.sqlite3") == []
    from cli.index import _load_corpus_manifest, _validate_staged_snapshot
    files = sorted(p.name for p in corpus.glob("*.txt"))
    _validate_staged_snapshot(files, _load_corpus_manifest(corpus), corpus)
    from cli.benchmark import _validate_benchmark_data
    _validate_benchmark_data(json.loads(queries.read_text()), str(queries))
    (corpus / files[0]).write_text("tampered")
    with pytest.raises(ValueError, match="content digest"):
        _validate_staged_snapshot(files, _load_corpus_manifest(corpus), corpus)


def test_missing_gold_sentence_is_reported_without_injecting_or_repairing_it(tmp_path):
    archive, gold = fixture_files(tmp_path)
    rows = json.loads(gold.read_text())
    rows[0]["supporting_facts"] = [["LEAK", 0]]
    gold.write_text(json.dumps(rows))
    manifest = build(archive, gold, tmp_path / "corpus", tmp_path / "queries.json", verify_official=False)
    assert manifest["gold_supporting_sentence_coverage"] == 0
    assert manifest["annotation_coverage"]["unavailable_facts"] == [
        {"query_id":"q", "title":"LEAK", "sentence_index":0, "corpus_sentences":None}]
    assert json.loads((tmp_path / "queries.json").read_text())[0]["supporting_facts"] == [["LEAK",0]]
    assert len(list((tmp_path / "corpus").glob("*.txt"))) == 2


def test_unavailable_official_fact_remains_in_scoring_denominator():
    metrics = score("answer", [["Article", 0]], "answer", [["Article", 0], ["Article", 902]])
    assert metrics["em"] == 1
    assert metrics["sp_recall"] == .5
    assert metrics["sp_f1"] == pytest.approx(2/3)
    assert metrics["joint_em"] == 0


def test_finalize_existing_rechecks_archive_and_does_not_rewrite_paragraphs(tmp_path):
    archive, gold = fixture_files(tmp_path)
    corpus, queries = tmp_path / "corpus", tmp_path / "queries.json"
    original = build(archive,gold,corpus,queries,verify_official=False)
    times = {p.name:p.stat().st_mtime_ns for p in corpus.glob("*.txt")}
    queries.unlink(); (corpus / "corpus_manifest.json").unlink()
    recovered = build(archive,gold,corpus,queries,verify_official=False,finalize_existing=True)
    assert recovered == original
    assert times == {p.name:p.stat().st_mtime_ns for p in corpus.glob("*.txt")}
    queries.unlink(); (corpus / "corpus_manifest.json").unlink()
    with sqlite3.connect(corpus / "sentences.sqlite3") as db:
        db.execute("UPDATE paragraphs SET title='tampered' WHERE source_id='hotpotqa_1'")
    with pytest.raises(ValueError, match="differs from source archive"):
        build(archive,gold,corpus,queries,verify_official=False,finalize_existing=True)
    assert not queries.exists()


def test_official_scorer_parity():
    path = Path("data/hotpotqa_raw/hotpot_evaluate_v1.py")
    if not path.exists():
        pytest.skip("Download official evaluator for external parity test")
    spec = importlib.util.spec_from_file_location("official_hotpot", path)
    module = importlib.util.module_from_spec(spec)
    # The official evaluator only needs JSON's public load API in these tests.
    sys.modules.setdefault("ujson", json)
    spec.loader.exec_module(module)
    rng = random.Random(42)
    words = ["yes", "no", "noanswer", "yes indeed", "The Cat!", "cat", "a", "", "New York"]
    for _ in range(250):
        pred, gold = rng.choice(words), rng.choice(words)
        pf = [["A", rng.randrange(3)] for _ in range(rng.randrange(5))]
        gf = [["A", rng.randrange(3)] for _ in range(rng.randrange(5))]
        expected = dict.fromkeys(METRICS, 0.0)
        em, p, r = module.update_answer(expected, pred, gold)
        sem, sp, sr = module.update_sp(expected, pf, gf)
        expected.update(joint_em=em*sem, joint_prec=p*sp, joint_recall=r*sr,
                        joint_f1=2*p*sp*r*sr/(p*sp+r*sr) if p*sp+r*sr else 0.0)
        assert score(pred, pf, gold, gf) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_benchmark_hotpot_metrics_and_failure_denominators(tmp_path):
    archive, gold = fixture_files(tmp_path)
    corpus = tmp_path / "corpus"
    build(archive, gold, corpus, tmp_path / "queries.json", verify_official=False)
    from utils.metrics import evaluate_multihoprag_response
    result = await evaluate_multihoprag_response("Where?", "@@ANSWER: here", "here",
        [{"source": "hotpotqa_1", "text": "Alpha is here."}], dataset="hotpotqa",
        supporting_facts=[["Article A", 0]], hotpot_sentence_store=str(corpus / "sentences.sqlite3"))
    assert result["hotpot_joint_em"] == 1.0
    assert result["official_qa_accuracy"] == -1
    from utils.reporting import compact_detail_row
    compact = compact_detail_row(result, 1)
    assert all(compact["hotpot_" + key] == result["hotpot_" + key] for key in METRICS)
    assert compact["predicted_supporting_facts"] == [["Article A", 0]]
    assert compact["support_prediction_policy"] == result["support_prediction_policy"]
    from cli.benchmark import _recompute_aggregates
    summary = {"details": [result, {"error": "timeout", "hotpot_joint_em": 1.0}]}
    _recompute_aggregates(summary)
    assert summary["avg_hotpot_joint_em"] == 0.5
    assert summary["eligible_hotpot_joint_em_count"] == 2


def test_hotpot_cold_fixture_has_its_own_sentence_store(tmp_path):
    from cli.benchmark import _load_benchmark_corpus_manifest, _validate_benchmark_data
    from scripts.cold_canary_fixture import stage_fixture
    base = tmp_path / "cold"
    corpus, manifest, row = stage_fixture(base, "hotpotqa")
    queries = base / "hotpotqa_queries.json"
    queries.write_text(json.dumps([row]))
    loaded = _load_benchmark_corpus_manifest("hotpotqa", queries)
    assert loaded["sentence_store_sha256"] == manifest["sentence_store_sha256"]
    _validate_benchmark_data([row], str(queries))
    files = sorted(corpus.glob("*.txt"))
    predictions = project_sentences([{"source": p.name, "text": p.read_text()} for p in files], corpus / "sentences.sqlite3")
    assert row["supporting_facts"][0] in predictions


def test_official_export_preserves_answer_and_accounts_for_failure():
    from scripts.export_hotpotqa_predictions import export
    result = {"status":"completed_unadmitted", "evaluation_scope":"full_benchmark", "dataset":"HotpotQA",
              "details":[{"query_id":"a", "answer":"@@ANSWER: Alpha", "predicted_supporting_facts":[["A",0]]},
                         {"query_id":"b", "error":"timeout"}]}
    gold = [{"_id":"a", "answer":"Alpha", "supporting_facts":[["A",0]]},
            {"_id":"b", "answer":"Beta", "supporting_facts":[["B",0]]}]
    predictions, metrics = export(result, gold)
    assert predictions == {"answer":{"a":"Alpha","b":""},"sp":{"a":[["A",0]],"b":[]}}
    assert metrics["joint_em"] == .5
    with pytest.raises(ValueError, match="IDs"):
        export(result, gold[:1])


def test_prepared_audit_checks_source_and_query_bytes(tmp_path, monkeypatch):
    import hashlib

    from scripts.datasets import prepare_hotpotqa as prep
    archive, gold = fixture_files(tmp_path)
    original = json.loads(gold.read_text())[0]
    rows = [original | {"_id":f"q{i}"} for i in range(7405)]
    gold.write_text(json.dumps(rows))
    monkeypatch.setattr(prep, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(prep, "ARCHIVE_MD5", prep.digest_file(archive, "md5"))
    monkeypatch.setattr(prep, "DEV_QUERY_IDS_SHA256", hashlib.sha256("\n".join(sorted(r["_id"] for r in rows)).encode()).hexdigest())
    (tmp_path / "download_provenance.json").write_text(json.dumps({"gold_sha256":prep.digest_file(gold), "archive_sha256":prep.digest_file(archive)}))
    corpus, queries = tmp_path / "hotpotqa_corpus", tmp_path / "queries.json"
    prep.build(archive,gold,corpus,queries)
    assert prep.verify_prepared(archive,gold,corpus,queries)["status"] == "ready"
    changed = json.loads(queries.read_text())
    changed[0]["ground_truth"] = "changed"
    queries.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Prepared queries"):
        prep.verify_prepared(archive,gold,corpus,queries)


@pytest.mark.parametrize("dataset", ["hotpotqa", "multihoprag"])
def test_paper_shell_dispatch_preserves_selected_dataset(tmp_path, dataset):
    import os
    import shutil
    import subprocess
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "core").mkdir()
    shutil.copyfile("scripts/run_paper_target.sh", scripts / "run_paper_target.sh")
    (tmp_path / ".env").write_text("")
    (tmp_path / "core/strategy_registry.py").write_text("import sys\nsys.exit(0)\n")
    (scripts / "lib.sh").write_text('''load_project_env() { :; }
resolve_method_python() { echo "$1/fake-python"; }
canonicalize_inference_transport() { export RAG_GENERATION_MODEL=test RAG_EMBEDDING_MODEL=test; }
''')
    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o700)
    for name in ("run_dataset.sh", "run_multihoprag.sh"):
        path = tmp_path / name
        path.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > dispatched-args\n')
        path.chmod(0o700)
    corpus = tmp_path / "data" / (dataset + "_corpus")
    corpus.mkdir(parents=True)
    (corpus / "corpus_manifest.json").write_text("{}")
    (tmp_path / "data" / (dataset + "_queries.json")).write_text("[]")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    environment = os.environ.copy()
    environment.pop("EMBEDDING_QUERY_INSTRUCTION", None)
    subprocess.run(["bash", str(scripts / "run_paper_target.sh"), dataset, "prehop", "fixture"],
                   cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    args = (tmp_path / "dispatched-args").read_text().splitlines()
    assert args == ([] if dataset == "multihoprag" else [dataset]) + ["all", "--model", "prehop", "--queries", "full"]
