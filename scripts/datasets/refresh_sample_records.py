"""Refresh a fixed sample's records from the current prepared full query file.

The sample's immutable query IDs and order are preserved. Only the records
associated with those IDs are replaced, which keeps newly prepared evaluation
annotations synchronized without drawing a new development sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def refresh_records(full_rows: list[dict], sample_rows: list[dict]) -> list[dict]:
    """Return current full records in the sample's immutable ID order."""
    full_by_id = {str(row.get("_id") or ""): row for row in full_rows}
    sample_ids = [str(row.get("_id") or "") for row in sample_rows]
    if not sample_ids or any(not query_id for query_id in sample_ids):
        raise ValueError("Sample contains a blank query ID")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Sample contains duplicate query IDs")
    missing = [query_id for query_id in sample_ids if query_id not in full_by_id]
    if missing:
        raise ValueError(f"Sample query IDs missing from full preparation: {missing[:5]}")
    return [full_by_id[query_id] for query_id in sample_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["multihoprag", "musique"])
    parser.add_argument("--sample", required=True, type=Path)
    args = parser.parse_args()

    full_path = ROOT / "data" / f"{args.dataset}_queries.json"
    sample_path = args.sample if args.sample.is_absolute() else ROOT / args.sample
    full_rows = json.loads(full_path.read_text(encoding="utf-8"))
    sample_rows = json.loads(sample_path.read_text(encoding="utf-8"))
    refreshed = refresh_records(full_rows, sample_rows)
    temporary = sample_path.with_name(f"{sample_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, sample_path)
    query_ids_sha256 = hashlib.sha256("\n".join(str(row["_id"]) for row in refreshed).encode()).hexdigest()
    print(f"refreshed={len(refreshed)} query_ids_sha256={query_ids_sha256} sample={sample_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
