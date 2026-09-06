#!/usr/bin/env python3
"""Read-only admission check for reusing a completed paper index artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.paper_policy import validate_canonical_index_policy
from core.strategy_registry import PRIMARY_STRATEGIES


def verify(path: Path, strategy: str, dataset: str, run_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"status": "complete", "strategy": strategy, "corpus_tag": dataset, "run_id": run_id}
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("index statistics identity/status does not match the requested target")
    digest = payload.get("index_policy_sha256")
    if not isinstance(digest, str):
        raise TypeError("completed index statistics are missing index_policy_sha256")
    validate_canonical_index_policy(strategy, dataset, payload.get("index_policy"), digest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("strategy", choices=PRIMARY_STRATEGIES)
    parser.add_argument("dataset", choices=("multihoprag", "musique"))
    parser.add_argument("run_id")
    args = parser.parse_args()
    try:
        verify(args.path, args.strategy, args.dataset, args.run_id)
    except Exception as exc:  # noqa: BLE001 - reuse must fail closed
        print(f"index policy verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
