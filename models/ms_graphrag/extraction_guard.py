"""Public completion-provider wrapper for native MS GraphRAG output contracts."""
from __future__ import annotations

import math
import re

from models.external_research.extraction_contract import ExtractionFormatError

PROVIDER = 'prehop_validated_litellm'


class OutputTruncatedError(ExtractionFormatError):
    """The same output budget cannot complete this request reliably."""


def validate_records(raw):
    if not isinstance(raw, str) or not raw.strip().endswith('<|COMPLETE|>'):
        raise ExtractionFormatError('MS extraction has no completion delimiter')
    body = raw.strip()[:-len('<|COMPLETE|>')].strip()
    entities, edges = set(), []
    for record in body.split('##'):
        record = record.strip()
        if not record:
            continue
        if not record.startswith('(') or not record.endswith(')'):
            raise ExtractionFormatError('MS extraction record is not parenthesized')
        fields = record[1:-1].split('<|>')
        if fields[0] == '"entity"' and len(fields) == 4:
            if any(not value.strip().strip('"') for value in fields[1:]):
                raise ExtractionFormatError('MS extraction entity has empty fields')
            entities.add(fields[1].strip().strip('"').upper())
        elif fields[0] == '"relationship"' and len(fields) == 5:
            if any(not value.strip().strip('"') for value in fields[1:4]):
                raise ExtractionFormatError('MS relationship has empty fields')
            try:
                strength = float(fields[4])
            except ValueError as exc:
                raise ExtractionFormatError('MS relationship weight is not numeric') from exc
            if not math.isfinite(strength):
                raise ExtractionFormatError('MS relationship weight is non-finite')
            edges.append((fields[1].strip().strip('"').upper(), fields[2].strip().strip('"').upper()))
        else:
            raise ExtractionFormatError('MS extraction record type or field count mismatch')
    return entities, edges


def normalize_glean_preamble(raw):
    """Remove only unstructured leading commentary before a complete native tuple.

    Tuple text, delimiters, and descriptions remain verbatim. Ambiguous or
    malformed structured prefixes are rejected by the regular record validator.
    """
    match = re.search(r'(?m)^\s*(?=\("(?:entity|relationship)"<\|>)', raw)
    if match is None or match.start() == 0:
        return raw
    prefix = raw[:match.end()]
    if any(marker in prefix for marker in ('<|>', '<|COMPLETE|>', '##', '("entity"', '("relationship"')):
        return raw
    candidate = raw[match.end():]
    validate_records(candidate)
    return candidate


def validate_response(response, messages, response_format=None):
    if any(choice.finish_reason == 'length' for choice in response.choices):
        raise OutputTruncatedError('MS completion reached the registered output limit')
    if len(response.choices) != 1 or response.choices[0].finish_reason != 'stop':
        raise ExtractionFormatError('MS completion did not finish completely')
    message = response.choices[0].message
    if getattr(message, 'refusal', None) or getattr(message, 'tool_calls', None):
        raise ExtractionFormatError('MS completion returned refusal or tool calls')
    raw = response.content
    if not isinstance(raw, str) or not raw.strip():
        raise ExtractionFormatError('MS completion has no text')
    if response_format is not None:
        # Validate before native report parser can swallow JSON/schema exceptions.
        response_format.model_validate_json(raw)
        return
    from graphrag.prompts.index.extract_graph import CONTINUE_PROMPT, GRAPH_EXTRACTION_PROMPT, LOOP_PROMPT
    history = messages if isinstance(messages, list) else [{'role': 'user', 'content': messages}]
    if history[-1].get('content') == LOOP_PROMPT:
        if raw not in ('Y', 'N'):
            raise ExtractionFormatError('MS glean continuation must be Y or N')
        return
    # A completed prose-only native glean adds no facts.
    if (history[-1].get('content') == CONTINUE_PROMPT and '<|>' not in raw
            and raw.strip().endswith('<|COMPLETE|>')
            and '("entity"' not in raw and '("relationship"' not in raw):
        return {'graph_extraction': True, 'native_empty_glean': True}
    if history[-1].get('content') == CONTINUE_PROMPT:
        raw = normalize_glean_preamble(raw)
    prefix = GRAPH_EXTRACTION_PROMPT.lstrip().split('{entity_types}')[0]
    if any(str(item.get('content', '')).lstrip().startswith(prefix) for item in history):
        entities, edges = validate_records(raw)
        for item in history:
            if item.get('role') == 'assistant' and '<|>' in str(item.get('content', '')):
                previous, _ = validate_records(item['content'])
                entities |= previous
        return {'graph_extraction': True, 'normalized_extraction': raw, 'glean_preamble_removed': raw != response.content, 'native_unresolved_relationships': [
            [source, target] for source, target in edges if source not in entities or target not in entities
        ]}
    return {}


def register_guard(audit, attempts):
    from graphrag_llm.completion.completion_factory import register_completion
    from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion

    class GuardedCompletion(LiteLLMCompletion):
        def __init__(self, **kwargs):
            # Native cache middleware otherwise replays the same invalid completion on retry.
            super().__init__(**{**kwargs, 'cache': None})
            native_sync, native_async = self._completion, self._completion_async

            def inspect(response, args, attempt):
                try:
                    observation = validate_response(response, args['messages'], args.get('response_format'))
                except (ValueError, TypeError) as exc:
                    audit.write(messages=args['messages'], response=response.model_dump(), attempt=attempt + 1,
                                status='exhausted' if isinstance(exc, OutputTruncatedError) or attempt + 1 == attempts else 'retry', error=str(exc))
                    if isinstance(exc, OutputTruncatedError):
                        raise
                    raise ExtractionFormatError(str(exc)) from exc
                raw_response = response.model_dump()
                from graphrag.prompts.index.extract_graph import CONTINUE_PROMPT
                history = args['messages'] if isinstance(args['messages'], list) else []
                glean = bool(history and history[-1].get('content') == CONTINUE_PROMPT)
                # Give completion markers their own native record. Otherwise
                # concatenating a glean can hide its first tuple behind the marker.
                if (observation or {}).get('graph_extraction'):
                    body = observation.get('normalized_extraction', response.content).strip()[:-len('<|COMPLETE|>')].strip()
                    if glean and '<|>' not in body:
                        body = ''  # Native prose-only glean adds no facts.
                    canonical = (body.removesuffix('##').rstrip() + '##' if body else '') + '<|COMPLETE|>'
                    if glean:
                        canonical = '##' + canonical
                    response.choices[0].message.content = canonical
                audit.write(messages=args['messages'], response=raw_response, attempt=attempt + 1,
                            status='accepted', native_observation=observation or {},
                            normalized_response=response.content)
                return response

            def inspect_stream(chunks):
                content, finish = [], None
                for chunk in chunks:
                    for choice in chunk.get('choices', []):
                        if choice.get('index', 0) != 0:
                            raise ExtractionFormatError('MS stream contains multiple choices')
                        delta = choice.get('delta') or {}
                        if delta.get('refusal') or delta.get('tool_calls'):
                            raise ExtractionFormatError('MS stream contains refusal or tool calls')
                        if delta.get('content'):
                            content.append(delta['content'])
                        if choice.get('finish_reason') is not None:
                            finish = choice['finish_reason']
                if finish != 'stop' or not ''.join(content).strip():
                    raise ExtractionFormatError('MS streamed completion is incomplete or empty')
                return ''.join(content)

            def stream_sync(response, args):
                chunks = []
                try:
                    for chunk in response:
                        chunks.append(chunk.model_dump())
                        yield chunk
                    content = inspect_stream(chunks)
                except Exception as exc:
                    audit.write(messages=args['messages'], response=chunks, status='exhausted', error=str(exc))
                    raise
                audit.write(messages=args['messages'], response=chunks, status='accepted', normalized_response=content)

            async def stream_async(response, args):
                chunks = []
                try:
                    async for chunk in response:
                        chunks.append(chunk.model_dump())
                        yield chunk
                    content = inspect_stream(chunks)
                except Exception as exc:
                    audit.write(messages=args['messages'], response=chunks, status='exhausted', error=str(exc))
                    raise
                audit.write(messages=args['messages'], response=chunks, status='accepted', normalized_response=content)

            def sync(**args):
                for attempt in range(attempts):
                    try:
                        response = native_sync(**args)
                    except Exception as exc:
                        audit.write(messages=args["messages"], attempt=attempt + 1,
                                    status="exhausted", error=f"{type(exc).__name__}: {exc}")
                        raise
                    if args.get("stream"):
                        return stream_sync(response, args)
                    try:
                        return inspect(response, args, attempt)
                    except OutputTruncatedError:
                        raise
                    except ExtractionFormatError:
                        if attempt + 1 == attempts:
                            raise
                raise AssertionError('Unreachable retry state')

            async def asynchronous(**args):
                for attempt in range(attempts):
                    try:
                        response = await native_async(**args)
                    except Exception as exc:
                        audit.write(messages=args["messages"], attempt=attempt + 1,
                                    status="exhausted", error=f"{type(exc).__name__}: {exc}")
                        raise
                    if args.get("stream"):
                        return stream_async(response, args)
                    try:
                        return inspect(response, args, attempt)
                    except OutputTruncatedError:
                        raise
                    except ExtractionFormatError:
                        if attempt + 1 == attempts:
                            raise
                raise AssertionError('Unreachable retry state')

            self._completion, self._completion_async = sync, asynchronous

    register_completion(PROVIDER, GuardedCompletion)
