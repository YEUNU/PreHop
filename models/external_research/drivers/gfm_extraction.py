"""Keep native GFM prompts while guarding text before native eval/parsing."""
from __future__ import annotations

from models.external_research.extraction_contract import (
    ExtractionFormatError,
    extraction_response_format,
    validate_extraction,
)


def guarded_chat_type(base):
    from pydantic import PrivateAttr

    class GuardedChat(base):
        _audit = PrivateAttr()
        _attempts = PrivateAttr()
        _ner_limits = PrivateAttr()
        _triple_limit = PrivateAttr()

        def configure_validation(self, audit, attempts, ner_limits, triple_limit):
            if attempts < 1 or triple_limit in ner_limits:
                raise ValueError('Ambiguous GFM extraction limits or invalid retry budget')
            self._audit, self._attempts = audit, attempts
            self._ner_limits, self._triple_limit = set(ner_limits), triple_limit
            return self

        def invoke(self, input, config=None, **kwargs):
            limit = kwargs.get('max_tokens')
            if limit in self._ner_limits:
                field = 'named_entities'
            elif limit == self._triple_limit:
                field = 'triples'
            else:
                raise RuntimeError('Unknown GFM extraction call contract')
            kwargs = {**kwargs, 'response_format': extraction_response_format(field)}
            prompt = [m.model_dump() if hasattr(m, 'model_dump') else m for m in input]
            for attempt in range(self._attempts):
                try:
                    result = super().invoke(input, config=config, **kwargs)
                except Exception as exc:
                    self._audit.write(messages=prompt, stage=field, attempt=attempt + 1,
                                      status='exhausted', error=f'{type(exc).__name__}: {exc}')
                    raise
                raw = result.content
                try:
                    canonical = validate_extraction(raw, field, result.response_metadata.get('finish_reason'))
                except ExtractionFormatError as exc:
                    self._audit.write(messages=prompt, stage=field, attempt=attempt + 1,
                                      response=raw, metadata=result.response_metadata,
                                      status='exhausted' if attempt + 1 == self._attempts else 'retry', error=str(exc))
                    if attempt + 1 == self._attempts:
                        raise
                    continue
                self._audit.write(messages=prompt, stage=field, attempt=attempt + 1,
                                  response=raw, metadata=result.response_metadata,
                                  normalized_response=canonical, status='accepted')
                # Only JSON strings/list structures reach native eval. No expression text.
                return result.model_copy(update={'content': canonical})
            raise AssertionError('Unreachable extraction retry state')

    return GuardedChat
