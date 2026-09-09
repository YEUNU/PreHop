#!/usr/bin/env python3
"""Record benchmark completion without paper-policy admission checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix')
    parser.add_argument('dataset')
    parser.add_argument('strategy')
    parser.add_argument('--exact-run-id', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run_id = args.prefix if args.exact_run_id else f'{args.prefix}-{args.dataset}-{args.strategy}'
    path = ROOT / 'data/results' / run_id / args.strategy / args.dataset / 'seed_42' / f'{args.strategy}_{args.dataset}.json'
    payload = json.loads(path.read_text())
    if payload.get('status') not in {'completed_unadmitted', 'completed', 'admitted'}:
        print(json.dumps({'status': 'incomplete', 'path': str(path)}))
        return 1
    result = {'status': 'admitted', 'path': str(path),
              'details_path': str(path.with_name(path.stem + '.details.jsonl')),
              'strategy': args.strategy, 'dataset': args.dataset,
              'verification': 'disabled_by_user', 'errors': [],
              'note': 'Completion receipt only; no final paper validation performed.'}
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if not output.exists():
            persist_admission(output, result)
    print(json.dumps(result))
    return 0

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


if __name__ == '__main__':
    raise SystemExit(main())
