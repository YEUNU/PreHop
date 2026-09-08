"""Pinned single-pass GFM QA prompt and generation with controlled transport."""
from __future__ import annotations

import os
from types import SimpleNamespace

from models.external_research.extraction_contract import ExtractionAudit


def native_qa_client(transport, original_model, retries, audit_path):
    import tiktoken
    from gfmrag.llms.chatgpt import ChatGPT
    from openai import OpenAI

    audit = ExtractionAudit(audit_path, profile="native-observation-v1")
    sdk = OpenAI(base_url=transport.generation_base_url, api_key=transport.api_key,
                 max_retries=0, timeout=transport.timeout_seconds,
                 default_headers={'X-Prehop-Run-ID': os.environ.get('RAG_BENCHMARK_TIMESTAMP', '')})

    def create(**kwargs):
        # Keep the original method's temperature, message list and omitted cap.
        # Native QA supplies 60 seconds per call; the shared transport owns this
        # operational deadline, including time waiting for queue admission.
        kwargs['timeout'] = transport.timeout_seconds
        response = sdk.chat.completions.create(**kwargs, seed=transport.generation_seed)
        audit.write(messages=kwargs['messages'], response=response.model_dump(), status='observed')
        return response

    class TransportChatGPT(ChatGPT):
        def __init__(self):
            self.retry = retries
            self.model_name = transport.generation_model
            self.maximun_token = transport.generation_max_context_tokens
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        def token_len(self, text):
            # Preserve the pinned workflow's tokenizer for its diagnostic guard;
            # the registered replacement backbone owns the actual server context.
            return len(tiktoken.encoding_for_model(original_model).encode(text))

    return TransportChatGPT(), audit, sdk
