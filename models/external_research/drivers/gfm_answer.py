"""Pinned single-pass GFM QA prompt and generation with controlled transport."""
from __future__ import annotations

from types import SimpleNamespace

from models.external_research.extraction_contract import ExtractionAudit


def native_qa_client(transport, original_model, retries, audit_path):
    import tiktoken
    from gfmrag.llms.chatgpt import ChatGPT
    from openai import OpenAI

    audit = ExtractionAudit(audit_path)
    sdk = OpenAI(base_url=transport.generation_base_url, api_key=transport.api_key,
                 max_retries=0, timeout=transport.timeout_seconds)

    def create(**kwargs):
        # Keep the original method's temperature, message list and omitted cap.
        response = sdk.chat.completions.create(**kwargs, seed=transport.generation_seed)
        choice = response.choices[0]
        valid = (choice.finish_reason == 'stop' and isinstance(choice.message.content, str)
                 and bool(choice.message.content.strip()) and not getattr(choice.message, 'refusal', None))
        audit.write(messages=kwargs['messages'], response=response.model_dump(),
                    status='accepted' if valid else 'exhausted')
        if not valid:
            raise RuntimeError('GFM native answer is incomplete or empty')
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
