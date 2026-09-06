#!/usr/bin/env python3
"""Read-only strict completion check for one paper target."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.admission import admission_bindings, verification_provenance
from core.strategy_registry import PRIMARY_STRATEGIES
from scripts.verify_submission_consistency import DATASETS, ROOT, _artifact_path, _load, _validate_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("strategy", choices=PRIMARY_STRATEGIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exact-run-id", action="store_true", help="Treat prefix as the exact target run ID")
    args = parser.parse_args()
    from scripts.check_paper_runtime import _load_runner_environment

    _load_runner_environment()
    path = _artifact_path(args.prefix, args.dataset, args.strategy, exact_run_id=args.exact_run_id)
    from core.paper_policy import configure_target_environment

    configure_target_environment(args.strategy, args.dataset, path.parts[2])
    absolute_path = ROOT / path
    details_path = absolute_path.with_name(absolute_path.stem + ".details.jsonl")
    payload = None
    try:
        payload = _load(path)
        errors = _validate_artifact(
            path,
            payload,
            dataset=args.dataset,
            strategy=args.strategy,
            expected_count=int(DATASETS[args.dataset]["count"]),
        )
    except Exception as exc:  # noqa: BLE001 - completion admission must fail closed
        errors = [f"{path}: {exc}"]
    bindings = admission_bindings(absolute_path, payload)
    required_bindings = {
        "result_sha256",
        "current_corpus_identity_sha256",
        "details_sha256",
        "evidence_contract",
        "configuration_sha256",
        "index_policy_sha256",
        "index_stats_path",
        "index_stats_sha256",
        "runtime_identity_sha256",
        "post_query_artifact_inventory_sha256",
        "recorded_post_query_artifact_inventory_sha256",
    }
    missing_bindings = sorted(key for key in required_bindings if not bindings.get(key))
    if missing_bindings:
        errors.append(f"{path}: admission content bindings are incomplete: {missing_bindings}")
    if (
        bindings.get("post_query_artifact_inventory_sha256")
        != bindings.get("recorded_post_query_artifact_inventory_sha256")
    ):
        errors.append(f"{path}: recorded post-query artifact inventory is stale")
    result = {
        "status": "admitted" if not errors else "failed",
        "path": str(absolute_path),
        "details_path": str(details_path),
        "strategy": args.strategy,
        "dataset": args.dataset,
        "bindings": bindings,
        "verification_provenance": verification_provenance(),
        "errors": errors,
    }
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        try:
            persist_admission(output, result)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            result['status'] = 'failed'
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


def persist_admission(output: Path, result: dict) -> None:
    """Called only after current artifact validation; never overwrite a ledger."""
    import os
    import time
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        fields = ('status', 'path', 'details_path', 'strategy', 'dataset', 'bindings', 'errors')
        if result['status'] == 'admitted' and all(previous.get(key) == result.get(key) for key in fields):
            return  # Keep original verifier/provenance and exact evidence bytes.
        raise RuntimeError('Existing admission has different bindings or failed current validation; preserve it and use a fresh namespace')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed checkpoint probe must not reserve the eventual successful ledger.
    # Preserve each failure separately so actual benchmark resume can complete.
    if result['status'] != 'admitted':
        output = output.with_name(output.stem + f'.failed-validation-{time.time_ns()}.json')
    pending = output.with_name(output.name + f'.{os.getpid()}.{time.time_ns()}.pending')
    try:
        with pending.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, output)  # Atomic publication without replacing historical bytes.
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
