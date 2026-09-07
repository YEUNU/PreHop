"""Adapter-owned extraction validation; never infer missing entities or relations."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

PROFILE = 'strict-extraction-v1'


class ExtractionFormatError(ValueError):
    pass


def extraction_response_format(field: str) -> dict:
    if field not in {'named_entities', 'triples'}:
        raise ValueError('Unknown extraction field')
    item = {'type': 'string'}
    if field == 'triples':
        item = {'type': 'array', 'items': item, 'minItems': 3, 'maxItems': 3}
    return {'type': 'json_schema', 'json_schema': {
        'name': f'{field}_adapter_v1', 'strict': True,
        'schema': {'type': 'object', 'properties': {field: {'type': 'array', 'items': item}},
                   'required': [field], 'additionalProperties': False},
    }}


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ExtractionFormatError('Duplicate extraction JSON key')
        value[key] = item
    return value


def _constant(value):
    raise ExtractionFormatError(f'Non-finite extraction JSON constant: {value}')


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ExtractionFormatError('Extraction values must be nonempty strings')
    return value


def normalize_entities(values):
    if not isinstance(values, list):
        raise ExtractionFormatError('Entity field must be a list')
    names = []
    for value in values:
        if isinstance(value, dict):
            keys = set(value) & {'entity', 'text'}
            if len(keys) != 1 or not set(value) <= {'entity', 'text', 'type', 'label'}:
                raise ExtractionFormatError('Ambiguous or unsupported entity object')
            for item in value.values():
                _name(item)
            value = value[next(iter(keys))]
        names.append(_name(value))
    return list(dict.fromkeys(names))


def normalize_triples(values):
    if not isinstance(values, list):
        raise ExtractionFormatError('Triple field must be a list')
    triples = []
    for value in values:
        if isinstance(value, dict):
            if set(value) == {'subject', 'predicate', 'object'}:
                value = [value[k] for k in ('subject', 'predicate', 'object')]
            elif set(value) == {'head', 'relation', 'tail'}:
                value = [value[k] for k in ('head', 'relation', 'tail')]
            else:
                raise ExtractionFormatError('Ambiguous or unsupported triple object')
        if not isinstance(value, list) or len(value) != 3:
            raise ExtractionFormatError('Triple must contain exactly three strings')
        triple = [_name(item) for item in value]
        if triple not in triples:
            triples.append(triple)
    return triples


def validate_extraction(raw: str, field: str, finish_reason: str) -> str:
    if field not in {'named_entities', 'triples'}:
        raise ValueError('Unknown extraction field')
    if finish_reason != 'stop':
        raise ExtractionFormatError(f'Incomplete extraction finish reason: {finish_reason}')
    if not isinstance(raw, str) or not raw.strip():
        raise ExtractionFormatError('Missing extraction response text')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ExtractionFormatError('Invalid extraction JSON') from exc
    if not isinstance(value, dict) or set(value) != {field}:
        raise ExtractionFormatError('Extraction top-level fields do not match contract')
    result = normalize_entities(value[field]) if field == 'named_entities' else normalize_triples(value[field])
    return json.dumps({field: result}, ensure_ascii=False)


class ExtractionAudit:
    def __init__(self, path: Path, *, profile=PROFILE):
        self.profile = profile
        self.path = path
        self.lock = threading.Lock()
        self.exhausted = 0

    def write(self, *, messages: Any, **record):
        prompt = json.dumps(messages, ensure_ascii=False, default=str)
        row = {'profile': self.profile, 'time': time.time(), 'prompt': messages,
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), **record}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            if record.get('status') == 'exhausted':
                self.exhausted += 1

    def assert_healthy(self):
        if self.exhausted:
            raise RuntimeError(f'Extraction exhausted for {self.exhausted} calls; inspect {self.path}')


def audit_evidence(audit: ExtractionAudit) -> dict:
    """Bind the completed index prefix while later query calls may append records."""
    from collections import Counter

    audit.assert_healthy()
    raw = audit.path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    counts = dict(Counter(row['status'] for row in rows))
    if not _valid_counts(counts, audit.profile):
        raise RuntimeError('Extraction audit has no complete successful evidence')
    return {'version': 1, 'path': str(audit.path.resolve()), 'prefix_bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(), 'counts': counts, 'profile': audit.profile}


def validate_audit_evidence(evidence: dict) -> None:
    from collections import Counter

    if not isinstance(evidence, dict) or evidence.get('version') != 1 or evidence.get('profile') not in {PROFILE, 'native-observation-v1'}:
        raise ValueError('Missing or incompatible extraction audit evidence')
    size = evidence.get('prefix_bytes')
    if type(size) is not int or size <= 0:
        raise ValueError('Invalid extraction audit byte boundary')
    with Path(evidence['path']).open('rb') as stream:
        raw = stream.read(size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != evidence.get('sha256'):
        raise ValueError('Extraction audit prefix is missing or changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    counts = dict(Counter(row['status'] for row in rows))
    if counts != evidence.get('counts') or not _valid_counts(counts, evidence['profile']):
        raise ValueError('Extraction audit does not attest successful completion')


def _valid_counts(counts, profile):
    if profile == 'native-observation-v1':
        # Attests observation integrity, not extraction quality. Native callers
        # must return normally before the indexer can publish completion evidence.
        return bool(counts) and set(counts) <= {'observed', 'native_exception'}
    return not counts.get('exhausted', 0) and bool(counts.get('accepted', 0))
