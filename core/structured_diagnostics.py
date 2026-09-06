"""Failure-only structured response metadata; never retain request/response text."""
from __future__ import annotations

import contextvars
import hashlib
import json
from contextlib import contextmanager
from typing import Any

_CALLER: contextvars.ContextVar[dict | None] = contextvars.ContextVar('structured_failure_caller', default=None)
_REQUEST_BUDGET: contextvars.ContextVar[dict | None] = contextvars.ContextVar('structured_request_budget', default=None)


@contextmanager
def structured_request_budget(attempts: int):
    """One shared wire-request ceiling across transport and format retries."""
    token = _REQUEST_BUDGET.set({'limit': attempts, 'used': 0})
    try:
        yield _REQUEST_BUDGET.get()
    finally:
        _REQUEST_BUDGET.reset(token)


def current_request_budget() -> dict | None:
    return _REQUEST_BUDGET.get()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8', errors='surrogatepass')).hexdigest()


def json_sha256(value: Any) -> str:
    return text_sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True))


@contextmanager
def indexing_failure_scope(source: str, content: str, title: str, chunk: str, page: int, chunk_index: int):
    """Identify the exact prepared native input without logging its contents."""
    token = _CALLER.set({'source_sha256': text_sha256(source), 'document_sha256': text_sha256(content),
                         'title_sha256': text_sha256(title), 'chunk_sha256': text_sha256(chunk),
                         'page': page if type(page) is int else None,
                         'page_chunk_index': chunk_index if type(chunk_index) is int else None})
    try:
        yield
    finally:
        _CALLER.reset(token)


def caller_metadata() -> dict:
    return dict(_CALLER.get() or {})


class DuplicateJSONPropertyError(ValueError):
    pass


class NonfiniteJSONConstantError(ValueError):
    pass


def parse_failure_metadata(error: Exception, raw: Any) -> dict:
    if isinstance(error, json.JSONDecodeError):
        known = {
            'Expecting value': 'expected_value',
            'Extra data': 'extra_data',
            'Expecting property name enclosed in double quotes': 'expected_quoted_property',
            "Expecting ':' delimiter": 'expected_colon',
            "Expecting ',' delimiter": 'expected_comma',
            'Unterminated string starting at': 'unterminated_string',
            'Invalid \\escape': 'invalid_escape',
            'Invalid \\uXXXX escape': 'invalid_unicode_escape',
            'Invalid control character at': 'invalid_control_character',
        }
        result = {'category': 'json_syntax', 'reason': known.get(error.msg, 'other_json_syntax'),
                  'offset': error.pos, 'line': error.lineno, 'column': error.colno}
    elif isinstance(error, DuplicateJSONPropertyError):
        result = {'category': 'duplicate_property'}
    elif isinstance(error, NonfiniteJSONConstantError):
        result = {'category': 'nonfinite_constant'}
    else:
        result = {'category': 'invalid_raw_type' if isinstance(error, TypeError) else 'other_parse_error'}
    if isinstance(raw, str):
        result['content'] = {'characters': len(raw), 'utf8_bytes': len(raw.encode('utf-8', errors='surrogatepass')),
            'sha256': text_sha256(raw), 'ascii': sum(ord(c) < 128 for c in raw),
            'non_ascii': sum(ord(c) >= 128 for c in raw), 'control': sum(ord(c) < 32 for c in raw),
            'whitespace': sum(c.isspace() for c in raw), 'surrogates': sum(0xD800 <= ord(c) <= 0xDFFF for c in raw),
            'bom': raw.count('\ufeff'), 'newlines': raw.count('\n')}
    return result
