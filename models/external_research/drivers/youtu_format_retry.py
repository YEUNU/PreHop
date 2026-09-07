"""Bounded unchanged-request retries at the controlled extraction boundary."""
from __future__ import annotations

import json
import threading
import time

from openai import APIConnectionError, APIStatusError

from core.native_structured_profile import (
    YoutuConstructionClient,
    validate_youtu_raw_response,
    youtu_response_format,
)
from models.external_research.extraction_contract import ExtractionAudit


class RetryingYoutuConstructionClient(YoutuConstructionClient):
    def __init__(self, client, attempts, audit_path=None):
        if type(attempts) is not int or attempts < 1:
            raise ValueError('Youtu extraction attempt budget must be positive')
        super().__init__(client.with_options(max_retries=0))
        self._audit = ExtractionAudit(audit_path) if audit_path is not None else None
        self._attempts = attempts
        self._stats_lock = threading.Lock()
        self._stats = {'attempts': 0, 'format_errors': 0, 'transport_errors': 0, 'exhausted': 0}

    def retry_stats(self):
        with self._stats_lock:
            return dict(self._stats)

    def _increment(self, key):
        with self._stats_lock:
            self._stats[key] += 1

    def _create(self, **kwargs):
        if any(key in kwargs for key in ('response_format', 'extra_body', 'structured_outputs')) or any(key.startswith('guided_') for key in kwargs):
            raise ValueError('Youtu construction structured format cannot be overridden')
        for attempt in range(self._attempts):
            self._increment('attempts')
            response = None
            try:
                response = self._client.chat.completions.create(**kwargs, response_format=youtu_response_format())
                validate_youtu_raw_response(response)
                from .youtu_graphrag import classify_native_extraction

                if classify_native_extraction(json.loads(response.choices[0].message.content)) != "success":
                    raise ValueError("Youtu extraction is empty or malformed")
                if self._audit is not None:
                    self._audit.write(messages=kwargs.get("messages"), attempt=attempt + 1,
                                      response=response.model_dump(), status="accepted")
                return response
            except (ValueError, APIConnectionError, APIStatusError) as exc:
                format_error = isinstance(exc, json.JSONDecodeError) or str(exc) in {
                    "Youtu structured extraction did not finish completely",
                    "Youtu structured extraction has no raw content",
                    "Youtu extraction is empty or malformed",
                    'duplicate extraction JSON property', 'non-finite extraction JSON constant',
                    'Youtu structured extraction top-level schema mismatch',
                    'Youtu structured extraction attributes must contain string arrays',
                    'Youtu structured extraction triples must contain exactly three strings',
                    'Youtu structured extraction entity_types must contain strings',
                    'Youtu structured extraction new_schema_types mismatch',
                }
                transport_error = isinstance(exc, APIConnectionError) or (
                    isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500))
                if self._audit is not None:
                    self._audit.write(messages=kwargs.get("messages"), attempt=attempt + 1,
                                      response=response.model_dump() if response is not None else None,
                                      status="retry" if (format_error or transport_error) and attempt + 1 < self._attempts else "exhausted",
                                      error=str(exc))
                if not format_error and not transport_error:
                    raise
                self._increment('format_errors' if format_error else 'transport_errors')
                if attempt + 1 == self._attempts:
                    self._increment('exhausted')
                    raise
                time.sleep(min(8, 2 ** attempt))
