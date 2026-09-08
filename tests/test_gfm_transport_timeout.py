from types import ModuleType, SimpleNamespace

from models.external_research.drivers.gfm_answer import native_qa_client


def test_native_call_timeout_uses_shared_transport(tmp_path, monkeypatch):
    calls = []

    class NativeChat:
        def generate_sentence(self, messages):
            return self.client.chat.completions.create(
                messages=messages, timeout=60, temperature=0.0, model=self.model_name)

    module = ModuleType("gfmrag.llms.chatgpt")
    module.ChatGPT = NativeChat
    monkeypatch.setitem(__import__("sys").modules, "gfmrag.llms.chatgpt", module)

    class SDK:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda: {"choices": []})

    monkeypatch.setattr("openai.OpenAI", SDK)
    transport = SimpleNamespace(
        generation_base_url="http://localhost/v1", api_key="test",
        timeout_seconds=600, generation_seed=None, generation_model="test",
        generation_max_context_tokens=262144)
    qa, _, _ = native_qa_client(transport, "gpt-4o", 5, tmp_path / "audit.jsonl")
    messages = [{"role": "user", "content": "question"}]
    qa.generate_sentence(messages)
    assert calls == [{"messages": messages, "timeout": 600, "temperature": 0.0,
                      "model": "test", "seed": None}]
    assert qa.retry == 5
