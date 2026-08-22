#!/usr/bin/env python
"""Resolve pending OpenAI Batch judge manifests for an interrupted run."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.benchmark import reconcile_pending_judges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"run directory not found: {args.run_dir}")
    patched = asyncio.run(reconcile_pending_judges(args.run_dir))
    print(f"Reconciled {patched} result file(s) under {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
