"""Inspect and verify a Prehop event log without contacting external services."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def iter_trace(path, *, source=None, query_id=None, errors=False, payloads=False):
    path = Path(path).resolve()
    root = path.parent
    verified = {}
    with path.open() as stream:
        yield from _read_events(stream, root, verified, source, query_id, errors, payloads)


def _read_events(stream, root, verified, source, query_id, errors, payloads):
    previous = 0
    for line in stream:
        event = json.loads(line)
        if event['sequence'] != previous + 1:
            raise ValueError('Trace sequence is missing or reordered')
        previous = event['sequence']
        target = (root / event['payload']).resolve()
        if root not in target.parents:
            raise ValueError('Trace payload escapes its session directory')
        if target not in verified:
            data = gzip.decompress(target.read_bytes())
            verified[target] = hashlib.sha256(data).hexdigest()
            json.loads(data)
        if verified[target] != event['payload_sha256']:
            raise ValueError('Trace payload hash mismatch')
        identity = event.get('identity', {})
        if source and source not in (identity.get('source'), identity.get('document_filename'),
                                      *identity.get('sources', [])):
            continue
        if query_id and identity.get('query_id') != query_id:
            continue
        data = json.loads(gzip.decompress(target.read_bytes())) if errors or payloads else None
        if errors and not (event['event'].endswith('.error')
                           or event['event'] == 'structured.attempt' and not data['valid']
                           or event['event'] == 'http.response' and data['status_code'] >= 400):
            continue
        if payloads:
            event['data'] = data
        yield event


def inspect_trace(path, **kwargs):
    return list(iter_trace(path, **kwargs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    parser.add_argument('--source')
    parser.add_argument('--query-id')
    parser.add_argument('--errors', action='store_true')
    parser.add_argument('--payloads', action='store_true', help='Include full request/response or stage payloads')
    args = parser.parse_args()
    for event in iter_trace(args.events, source=args.source, query_id=args.query_id,
                               errors=args.errors, payloads=args.payloads):
        print(json.dumps(event, ensure_ascii=True))


if __name__ == '__main__':
    main()
