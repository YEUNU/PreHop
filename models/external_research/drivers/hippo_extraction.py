"""Validate both native OpenIE calls before permissive native parsing."""
from __future__ import annotations

from pathlib import Path

from models.external_research.extraction_contract import (
    ExtractionAudit,
    ExtractionFormatError,
    extraction_response_format,
    validate_extraction,
)


def install_extraction_adapter(openie, audit_path: Path, attempts: int):
    if attempts < 1:
        raise ValueError('Extraction attempt budget must be positive')
    native_llm = openie.llm_model
    audit = ExtractionAudit(audit_path)

    class ValidatedLLM:
        def __getattr__(self, name):
            return getattr(native_llm, name)

        def infer(self, messages, **kwargs):
            limit = kwargs.get('max_new_tokens')
            if openie.ner_max_tokens == openie.triple_max_tokens:
                raise RuntimeError('Cannot distinguish native extraction call contracts')
            if limit == openie.ner_max_tokens:
                field = 'named_entities'
            elif limit == openie.triple_max_tokens:
                field = 'triples'
            else:
                raise RuntimeError('Unknown native extraction call contract')
            kwargs = {**kwargs, 'response_format': extraction_response_format(field)}
            billable = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            for attempt in range(attempts):
                # A format retry must reach the provider, not repeat an invalid cache hit.
                try:
                    if attempt:
                        uncached = getattr(native_llm.infer, '__wrapped__', None)
                        if uncached is None:
                            raise RuntimeError('Native inference has no verified uncached retry boundary')
                        raw, metadata = uncached(native_llm, messages=messages, **kwargs)
                        cache_hit = False
                    else:
                        raw, metadata, cache_hit = native_llm.infer(messages=messages, **kwargs)
                except Exception as exc:
                    audit.write(messages=messages, stage=field, attempt=attempt + 1,
                                status="exhausted", error=f"{type(exc).__name__}: {exc}")
                    raise
                if not cache_hit:
                    for key in billable:
                        billable[key] += metadata.get(key, 0)
                try:
                    canonical = validate_extraction(raw, field, metadata.get('finish_reason'))
                except ExtractionFormatError as exc:
                    audit.write(messages=messages, stage=field, attempt=attempt + 1, response=raw,
                                metadata=metadata, cache_hit=cache_hit,
                                status='exhausted' if attempt + 1 == attempts else 'retry', error=str(exc))
                    if attempt + 1 == attempts:
                        raise
                    continue
                audit.write(messages=messages, stage=field, attempt=attempt + 1, response=raw,
                            normalized_response=canonical, metadata=metadata, cache_hit=cache_hit, status='accepted')
                result_metadata = dict(metadata, adapter_raw_response=raw, adapter_attempts=attempt + 1)
                if attempt:
                    result_metadata.update(billable)
                    cache_hit = False
                return canonical, result_metadata, cache_hit
            raise AssertionError('Unreachable extraction retry state')

    openie.llm_model = ValidatedLLM()
    # Preserve wire text in native output records, even when names were unwrapped.
    for method_name in ('ner', 'triple_extraction'):
        original = getattr(openie, method_name)

        def observed(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if 'adapter_raw_response' in result.metadata:
                result.response = result.metadata.pop('adapter_raw_response')
            return result

        setattr(openie, method_name, observed)
    return audit
