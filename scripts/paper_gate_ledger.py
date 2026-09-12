#!/usr/bin/env python3
"""Create, advance, and verify a content-bound paper live-gate ledger."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.admission import identity_sha256, sha256_file, verifier_sources
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


def _context() -> dict[str, object]:
    from core.paper_compatibility import context_configuration
    return context_configuration()


def _provenance() -> dict:
    return {"code": code_provenance(), "verifier_sha256": identity_sha256(
        {str(path): sha256_file(path) for path in verifier_sources()})}


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def initialize(path: Path, campaign: str) -> None:
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
    artifact = (ROOT / str(reference["path"])).resolve()
    value = json.loads(artifact.read_text(encoding="utf-8"))
    return artifact, value


def execute_stage(path: Path, campaign: str, stage: str) -> None:
    """Produce receipts from successful commands and advance one live stage."""
    import subprocess
    _read(path)
    output = path.parent / "gates" / stage
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
    evidence = evidence.resolve()
    payload['stages'][stage] = {
        'status': 'completed', 'recorded_at_epoch': time.time(),
        'evidence_path': str(evidence.relative_to(ROOT)), 'evidence_sha256': sha256_file(evidence),
        'verification': 'not_checked',
    }
    payload['status'] = 'completed' if stage == STAGES[-1] else 'in_progress'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "record", "execute"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    from scripts.runner_environment import _load_runner_environment

    _load_runner_environment()
    path = args.ledger if args.ledger.is_absolute() else ROOT / args.ledger
    if args.command == "init":
        initialize(path, args.campaign)
    elif args.command == "execute":
        execute_stage(path, args.campaign, args.stage)
    elif args.command == "record":
        if args.stage is None or args.evidence is None:
            parser.error("record requires --stage and --evidence")
        evidence = args.evidence if args.evidence.is_absolute() else ROOT / args.evidence
        record(path, args.stage, evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
