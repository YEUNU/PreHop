"""Record native calls without repairing output or overriding native fallback."""
from __future__ import annotations

from types import SimpleNamespace

PROFILE = 'native-observation-v1'


def record(audit, messages, response=None, error=None, **extra):
    payload = response.model_dump() if hasattr(response, 'model_dump') else response
    audit.write(messages=messages, response=payload,
                status='native_exception' if error is not None else 'observed',
                **({'error': f'{type(error).__name__}: {error}'} if error is not None else {}), **extra)


def observed_chat_type(base):
    from pydantic import PrivateAttr

    class ObservedChat(base):
        _audit = PrivateAttr()

        def configure_validation(self, audit, attempts, ner_limits, triple_limit):
            self._audit = audit
            return self

        def invoke(self, input, config=None, **kwargs):
            prompt = [m.model_dump() if hasattr(m, 'model_dump') else m for m in input]
            try:
                response = super().invoke(input, config=config, **kwargs)
            except Exception as exc:
                record(self._audit, prompt, error=exc, request_options=kwargs)
                raise
            record(self._audit, prompt, response, request_options=kwargs)
            return response

    return ObservedChat


def observe_hippo(openie, audit_path, attempts):
    from .extraction_contract import ExtractionAudit
    audit = ExtractionAudit(audit_path, profile=PROFILE)
    native = openie.llm_model

    class ObservedLLM:
        def __getattr__(self, name):
            return getattr(native, name)

        def infer(self, messages, **kwargs):
            try:
                result = native.infer(messages=messages, **kwargs)
            except Exception as exc:
                record(audit, messages, error=exc, request_options=kwargs)
                raise
            raw, metadata, cache_hit = result
            record(audit, messages, raw, metadata=metadata, cache_hit=cache_hit, request_options=kwargs)
            return result

    openie.llm_model = ObservedLLM()
    return audit


class ObservedYoutuClient:
    def __init__(self, client, attempts, audit_path=None):
        from .extraction_contract import ExtractionAudit
        self._client = client
        self._audit = ExtractionAudit(audit_path, profile=PROFILE) if audit_path is not None else None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self._client, name)

    def retry_stats(self):
        return {'attempts': self._calls, 'adapter_retries': 0, 'policy': PROFILE}

    def _create(self, **kwargs):
        self._calls += 1
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._audit is not None:
                record(self._audit, kwargs.get('messages'), error=exc)
            raise
        if self._audit is not None:
            record(self._audit, kwargs.get('messages'), response)
        return response


def register_ms_observer(audit, attempts):
    from graphrag_llm.completion.completion_factory import register_completion
    from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
    from models.ms_graphrag.extraction_guard import PROVIDER

    class ObservedCompletion(LiteLLMCompletion):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            native_sync, native_async = self._completion, self._completion_async

            def stream(response, args):
                try:
                    for chunk in response:
                        record(audit, args['messages'], chunk, stream=True)
                        yield chunk
                except Exception as exc:
                    record(audit, args['messages'], error=exc, stream=True)
                    raise

            async def astream(response, args):
                try:
                    async for chunk in response:
                        record(audit, args['messages'], chunk, stream=True)
                        yield chunk
                except Exception as exc:
                    record(audit, args['messages'], error=exc, stream=True)
                    raise

            def sync(**args):
                try:
                    response = native_sync(**args)
                except Exception as exc:
                    record(audit, args['messages'], error=exc)
                    raise
                if args.get('stream'):
                    return stream(response, args)
                record(audit, args['messages'], response)
                return response

            async def asynchronous(**args):
                try:
                    response = await native_async(**args)
                except Exception as exc:
                    record(audit, args['messages'], error=exc)
                    raise
                if args.get('stream'):
                    return astream(response, args)
                record(audit, args['messages'], response)
                return response

            self._completion, self._completion_async = sync, asynchronous

    register_completion(PROVIDER, ObservedCompletion)
