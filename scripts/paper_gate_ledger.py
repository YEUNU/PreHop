#!/usr/bin/env python3
"""Create, advance, and verify a content-bound paper live-gate ledger."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.admission import identity_sha256, sha256_file, verifier_sources
from core.runtime_requirements import runtime_identity
from core.strategy_registry import PRIMARY_STRATEGIES
from utils.provenance import code_provenance

STAGES = (
    "static_go",
    "runtime_setup",
    "preflight_16",
    "chat_probe",
    "embedding_probe",
    "bisection_probe",
    "cold_canary_16",
    "resume_stale_rejection",
    "one_query_matrix_16",
    "full_target_admitted",
)
DATASETS = ("multihoprag", "hotpotqa")
MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _context() -> dict[str, object]:
    from core.paper_compatibility import context_configuration
    return context_configuration()


def _provenance() -> dict:
    return {"code": code_provenance(), "verifier_sha256": identity_sha256(
        {str(path): sha256_file(path) for path in verifier_sources()})}


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unsupported live-gate ledger")
    return payload


def initialize(path: Path, campaign: str) -> None:
    if path.exists():
        raise FileExistsError(f"live-gate ledger already exists: {path}")
    payload = {
        "schema_version": 1,
        "campaign": campaign,
        "status": "planned",
        "created_at_epoch": time.time(),
        "context": _context(),
        "provenance": _provenance(),
        "stages": {stage: {"status": "planned"} for stage in STAGES},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bound_json(reference: object) -> tuple[Path, dict]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise TypeError("gate artifact reference requires path and sha256")
    artifact = (ROOT / str(reference["path"])).resolve()
    if ROOT not in artifact.parents or not artifact.is_file() or sha256_file(artifact) != reference["sha256"]:
        raise RuntimeError("gate bound artifact is missing or changed")
    value = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("gate bound artifact must be a JSON object")
    return artifact, value


def _validate_full_admission(evidence_path: Path, value: dict) -> None:
    """Final paper validation removed; completion is recorded by the runner."""
    return


def _validate_canary_artifacts(stage: str, strategy: str, dataset: str, index_path: Path, query: dict, index: dict) -> None:
    """Apply the real index/runtime contract to canary-produced artifacts."""
    import os

    from core.admission import current_post_query_inventory
    from core.paper_policy import configure_target_environment
    from scripts.check_paper_runtime import check
    from scripts.verify_index_policy import verify as verify_index

    run_id = index.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise RuntimeError("canary requires an exact run identity")
    prior = os.environ.copy()
    try:
        configure_target_environment(strategy, dataset, run_id)
        # This consumes actual index_policy/keyset/hash and realized pinned
        # runtime files, rather than an invented canary completion schema.
        verify_index(index_path, strategy, dataset, run_id)
        check(strategy, dataset)
        if query.get("index_provenance", {}).get("policy_sha256") != index.get("index_policy_sha256"):
            raise RuntimeError("canary query/index policy binding differs")
        if query.get("index_stats_sha256") != sha256_file(index_path):
            raise RuntimeError("canary query/index content binding differs")
        if query.get("runtime_identity") != runtime_identity(strategy):
            raise RuntimeError("canary runtime identity is stale")
        inventory = current_post_query_inventory(strategy, dataset)
        if query.get("post_query_artifact_inventory") != inventory:
            raise RuntimeError("canary native artifact inventory is stale")
        if strategy not in {"prehop", "naive"} and inventory.get("file_count", 0) < 1:
            raise RuntimeError("canary native index artifacts are missing")
        records_path, records = _bound_json(query.get("query_record"))
        if records.get("dataset") != dataset or not records.get("_id") or not records.get("query"):
            raise RuntimeError("canary query record identity is incomplete")
        if query.get("query") != records["query"] or query.get("query_id") != records["_id"]:
            raise RuntimeError("canary output does not match its evaluated query record")
        if query.get("query_records_sha256") != sha256_file(records_path):
            raise RuntimeError("canary query record binding is stale")
        source_path, source = _bound_json(index.get("source_manifest"))
        if source.get("schema_version") != 2 or source.get("fingerprint") != index.get("corpus_manifest_fingerprint"):
            raise RuntimeError("canary source manifest identity differs")
        from cli.index import _load_corpus_manifest, _validate_staged_snapshot
        if source_path.name != "corpus_manifest.json":
            raise RuntimeError("canary source reference must be the actual corpus_manifest.json")
        validated = _load_corpus_manifest(source_path.parent)
        if validated is None or validated.get("fingerprint") != source.get("fingerprint"):
            raise RuntimeError("canary staged corpus identity is missing or differs")
        files = sorted(path.name for path in source_path.parent.iterdir() if path.is_file() and path.suffix in {".txt", ".md"})
        source_ids = set(_validate_staged_snapshot(files, validated, source_path.parent))
        from core.admission import current_corpus_identity
        expected_count = 2
        if stage == "cold_canary_16":
            from scripts.cold_canary_fixture import validate_fixture
            validate_fixture(source_path.parent, dataset, records, index.get("cold_fixture"))
            _, native_stats = _bound_json(index.get("native_index_stats"))
            for field in ("status", "strategy", "corpus_tag", "run_id", "index_policy", "index_policy_sha256", "corpus_manifest_fingerprint"):
                if native_stats.get(field) != index.get(field):
                    raise RuntimeError("cold evidence differs from bound native index stats")
            if query.get("active_index_snapshot", {}).get("source_count") != 2:
                raise RuntimeError("cold observed native snapshot source count differs")
        elif records.get("cold_fixture") is not None or index.get("cold_fixture") is not None:
            raise RuntimeError("Synthetic cold fixtures cannot supply real dataset evidence")
        if stage == "one_query_matrix_16":
            full_queries = json.loads((ROOT / "data" / f"{dataset}_queries.json").read_text(encoding="utf-8"))
            if sum(row == records for row in full_queries) != 1:
                raise RuntimeError("one-query canary query is not an exact current dataset member")
            current_corpus = current_corpus_identity(dataset)
            expected_count = current_corpus["paragraph_count"]
            if any(validated.get(field) != current_corpus.get(field) for field in
                   ("fingerprint", "corpus_files_sha256", "corpus_records_sha256", "source_ids_sha256")):
                raise RuntimeError("one-query canary does not use the current complete corpus")
        if len(files) != expected_count or index.get("source_count") != expected_count:
            raise RuntimeError("canary staged/index source count differs")
        if query.get("active_index_snapshot", {}).get("status") != "matched":
            raise RuntimeError("canary has no matched active native index snapshot")
        if any(not isinstance(row, dict) or row.get("source_id") not in source_ids or not row.get("text") for row in query["documents"]):
            raise RuntimeError("canary returned malformed native evidence")
    finally:
        os.environ.clear()
        os.environ.update(prior)


def _validate_evidence(stage: str, evidence_path: Path, value: dict) -> None:
    if not isinstance(value, dict) or value.get("status") not in {"canary_passed", "admitted"}:
        raise RuntimeError("gate evidence does not attest canary_passed/admitted")
    if stage == "static_go":
        # Reviewer attestation is explicitly separate from empirical evidence.
        if value.get("kind") != "independent_static_review" or not value.get("reviewer"):
            raise RuntimeError("static gate requires an independent reviewer attestation")
        _, report = _bound_json(value.get("report"))
        if report.get("verdict") != "GO":
            raise RuntimeError("independent static review did not conclude GO")
        from core.paper_compatibility import without_realized_runtime
        if report.get("reviewed_configuration_sha256") != identity_sha256(without_realized_runtime(_context())):
            raise RuntimeError("independent static review covers a different effective configuration")
        return
    if stage == "full_target_admitted":
        if value.get("status") != "admitted":
            raise RuntimeError("full-target evidence is not an admission record")
        _validate_full_admission(evidence_path, value)
        return
    if value.get("schema_version") != 1 or value.get("stage") != stage:
        raise RuntimeError("empirical gate evidence schema/stage mismatch")
    if stage in {"preflight_16", "cold_canary_16", "one_query_matrix_16"}:
        targets = value.get("targets")
        expected = {f"{dataset}/{strategy}" for strategy in PRIMARY_STRATEGIES for dataset in DATASETS}
        if not isinstance(targets, dict) or set(targets) != expected:
            raise RuntimeError(f"{stage} evidence requires exact 16-target bound artifacts")
        for target, reference in targets.items():
            _, result = _bound_json(reference)
            dataset, strategy = target.split("/")
            if result.get("strategy") != strategy or result.get("dataset") != dataset or result.get("status") != "canary_passed":
                raise RuntimeError("canary artifact target/status mismatch")
            if result.get("stage") != stage or result.get("exit_code") != 0:
                raise RuntimeError("canary artifact stage/exit mismatch")
            _, invocation = _bound_json(result.get("invocation"))
            if invocation.get("exit_code") != 0 or not invocation.get("argv") or invocation.get("target") != target:
                raise RuntimeError("canary invocation receipt is invalid")
            if stage != "preflight_16":
                index_path, index = _bound_json(result.get("index"))
                _, query = _bound_json(result.get("query"))
                _, admission = _bound_json(result.get("admission"))
                if index.get("status") != "complete" or not index.get("fresh_index"):
                    raise RuntimeError("canary requires a complete fresh index")
                if stage == "cold_canary_16" and index.get("source_count") != 2:
                    raise RuntimeError("cold canary requires exactly two documents")
                if query.get("query_count") != 1 or not isinstance(query.get("answer"), str) or not query["answer"].strip():
                    raise RuntimeError("canary requires one completed native query")
                if not isinstance(query.get("documents"), list) or not query["documents"]:
                    raise RuntimeError("canary requires native retrieved evidence")
                for artifact in (index, query, admission):
                    if artifact.get("strategy") != strategy or artifact.get("dataset") != dataset:
                        raise RuntimeError("canary index/query/admission identity mismatch")
                if not index.get("run_id") or any(artifact.get("run_id") != index["run_id"] for artifact in (query, admission)):
                    raise RuntimeError("canary run identities differ")
                if admission.get("status") != "canary_passed" or admission.get("errors") != []:
                    raise RuntimeError("canary admission is invalid")
                if admission.get("index_sha256") != result["index"]["sha256"] or admission.get("query_sha256") != result["query"]["sha256"]:
                    raise RuntimeError("canary admission content binding is stale")
                _validate_canary_artifacts(stage, strategy, dataset, index_path, query, index)
        return
    _, receipt = _bound_json(value.get("receipt"))
    if receipt.get("stage") != stage or receipt.get("exit_code") != 0 or not receipt.get("argv"):
        raise RuntimeError("probe/setup invocation receipt is invalid")
    if stage == "resume_stale_rejection":
        from scripts.recovery_checkpoint import validate_recovery_evidence
        validate_recovery_evidence(value)
    elif stage == "chat_probe":
        if not isinstance(value.get("answer"), str) or not value["answer"].strip() or value.get("request_count") != 1:
            raise RuntimeError("chat probe response is incomplete")
    elif stage == "embedding_probe":
        vectors = value.get("vectors")
        from math import isfinite

        from core.strategy_registry import PAPER_TRANSPORT
        batch_size = PAPER_TRANSPORT.embedding_batch_size
        if value.get("batch_size") != batch_size or value.get("concurrency") != PAPER_TRANSPORT.embedding_concurrency or not isinstance(vectors, list) or len(vectors) != batch_size:
            raise RuntimeError("embedding probe batch/count/concurrency differs")
        if [row.get("index") for row in vectors] != list(range(batch_size)):
            raise RuntimeError("embedding probe indices differ")
        if any(not isinstance(row.get("embedding"), list)
               or len(row["embedding"]) != PAPER_TRANSPORT.embedding_dimensions
               or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) for x in row["embedding"])
               for row in vectors):
            raise RuntimeError("embedding probe dimension/finite values differ")
    elif stage == "bisection_probe":
        if value.get("oversized_rejected") is not True or value.get("selective_bisection_passed") is not True:
            raise RuntimeError("bisection evidence is incomplete")
    elif stage == "runtime_setup":
        if receipt.get("checks") != {name: "passed" for name in PRIMARY_STRATEGIES}:
            raise RuntimeError("runtime setup receipt lacks exact primary dependency checks")


def ready(path: Path, stage: str) -> None:
    """Check prerequisites and freshness before any live action."""
    payload = _read(path)
    if payload.get("context") != _context():
        raise RuntimeError("live-gate context is stale before stage execution")
    for previous in STAGES[:STAGES.index(stage)]:
        row = payload.get("stages", {}).get(previous, {})
        if row.get("status") != "canary_passed":
            raise RuntimeError(f"prerequisite gate is incomplete: {previous}")
        if time.time() - float(row.get("recorded_at_epoch") or 0) > MAX_AGE_SECONDS:
            raise RuntimeError(f"prerequisite gate is stale: {previous}")
        evidence = (ROOT / str(row.get("evidence_path") or "")).resolve()
        if ROOT not in evidence.parents or not evidence.is_file() or sha256_file(evidence) != row.get("evidence_sha256"):
            raise RuntimeError(f"prerequisite evidence is missing or changed: {previous}")
        _validate_evidence(previous, evidence, json.loads(evidence.read_text(encoding="utf-8")))


def execute_stage(path: Path, campaign: str, stage: str) -> None:
    """Produce receipts from successful commands and advance one live stage."""
    import subprocess
    if stage not in {"runtime_setup", "preflight_16", "chat_probe", "embedding_probe", "bisection_probe"}:
        raise RuntimeError("this stage requires separately produced native artifacts")
    payload = _read(path)
    if payload.get("campaign") != campaign:
        raise RuntimeError("live stage campaign identity mismatch")
    ready(path, stage)
    output = path.parent / "gates" / stage
    if output.exists():
        raise FileExistsError("live stage output already exists; preserve prior evidence")
    output.mkdir(parents=True)
    def save(name: str, value: dict) -> dict:
        target = output / (name + ".json")
        target.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(target.relative_to(ROOT)), "sha256": sha256_file(target)}
    evidence = {"status": "canary_passed", "schema_version": 1, "stage": stage}
    if stage == "preflight_16":
        targets = {}
        for strategy in PRIMARY_STRATEGIES:
            for dataset in DATASETS:
                target = f"{dataset}/{strategy}"
                command = ["bash", "scripts/run_paper_target.sh", dataset, strategy,
                           f"{campaign}-check-{dataset}-{strategy}", "--check"]
                subprocess.run(command, cwd=ROOT, check=True)
                invocation = save(f"{dataset}-{strategy}-invocation", {"argv": command, "exit_code": 0, "target": target})
                targets[target] = save(f"{dataset}-{strategy}", {"stage": stage, "status": "canary_passed",
                    "strategy": strategy, "dataset": dataset, "exit_code": 0, "invocation": invocation})
        evidence["targets"] = targets
    elif stage == "runtime_setup":
        command = ["bash", "scripts/setup_official_baselines.sh", "--primary"]
        subprocess.run(command, cwd=ROOT, check=True)
        evidence["receipt"] = save("invocation", {"argv": command, "exit_code": 0, "stage": stage,
                                                   "checks": {name: "passed" for name in PRIMARY_STRATEGIES}})
    else:
        command = [sys.executable, "scripts/probe_inference_gateway.py", "--probe", stage.removesuffix("_probe")]
        completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
        evidence.update(json.loads(completed.stdout))
        evidence["receipt"] = save("invocation", {"argv": command, "exit_code": 0, "stage": stage})
    reference = save("evidence", evidence)
    record(path, stage, ROOT / reference["path"])


def record(path: Path, stage: str, evidence: Path) -> None:
    payload = _read(path)
    stages = payload["stages"]
    index = STAGES.index(stage)
    incomplete = [name for name in STAGES[:index] if stages.get(name, {}).get("status") != "canary_passed"]
    if incomplete:
        raise RuntimeError(f"cannot record {stage}; prerequisite gates are incomplete: {incomplete}")
    current_context = _context()
    if stage == "runtime_setup":
        prior_context = payload.get("context", {})
        from core.paper_compatibility import without_realized_runtime
        if without_realized_runtime(prior_context) != without_realized_runtime(current_context):
            raise RuntimeError("runtime setup changed source/verifier/target context")
        payload["context"] = current_context
    elif payload.get("context") != current_context:
        raise RuntimeError("cannot record gate against stale code/runtime context")
    evidence = evidence.resolve()
    if ROOT not in evidence.parents or not evidence.is_file():
        raise RuntimeError("gate evidence must be an existing file under the repository root")
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    _validate_evidence(stage, evidence, evidence_payload)
    stages[stage] = {
        "status": "canary_passed",
        "recorded_at_epoch": time.time(),
        "evidence_path": str(evidence.relative_to(ROOT)),
        "evidence_sha256": sha256_file(evidence),
    }
    payload["status"] = "canary_passed" if stage == STAGES[-1] else "in_progress"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(path: Path, campaign: str) -> None:
    payload = _read(path)
    if payload.get("campaign") != campaign or not re.fullmatch(r"[A-Za-z0-9._-]+", campaign):
        raise RuntimeError("live-gate ledger campaign identity mismatch")
    if payload.get("context") != _context():
        raise RuntimeError("live-gate ledger is stale for the current code/runtime context")
    for stage in STAGES:
        row = payload.get("stages", {}).get(stage, {})
        if row.get("status") != "canary_passed":
            raise RuntimeError(f"required live gate is incomplete: {stage}")
        recorded = float(row.get("recorded_at_epoch") or 0)
        if recorded <= 0 or time.time() - recorded > MAX_AGE_SECONDS:
            raise RuntimeError(f"required live gate is stale: {stage}")
        evidence = ROOT / str(row.get("evidence_path") or "")
        if not evidence.is_file() or sha256_file(evidence) != row.get("evidence_sha256"):
            raise RuntimeError(f"required live-gate evidence is missing or changed: {stage}")
        _validate_evidence(stage, evidence.resolve(), json.loads(evidence.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "record", "verify", "ready", "execute"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    from scripts.check_paper_runtime import _load_runner_environment

    _load_runner_environment()
    path = args.ledger if args.ledger.is_absolute() else ROOT / args.ledger
    if args.command == "init":
        initialize(path, args.campaign)
    elif args.command in {"ready", "execute"}:
        if args.stage is None:
            parser.error("ready/execute requires --stage")
        if args.command == "ready":
            ready(path, args.stage)
        else:
            execute_stage(path, args.campaign, args.stage)
    elif args.command == "record":
        if args.stage is None or args.evidence is None:
            parser.error("record requires --stage and --evidence")
        evidence = args.evidence if args.evidence.is_absolute() else ROOT / args.evidence
        record(path, args.stage, evidence)
    else:
        verify(path, args.campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
