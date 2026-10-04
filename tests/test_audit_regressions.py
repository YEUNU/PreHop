"""Executable boundary regressions; all mutable artifacts live in pytest temp dirs."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from core.admission import identity_sha256
from core.inference_transport import InferenceTransport
from core.strategy_registry import paper_environment_defaults

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_process_environment():
    before = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(before)


def _transport(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(('RAG_', 'VLLM_', 'OPENAI_', 'AZURE_', 'EMBEDDING_', 'MAX_EMBEDDING', 'NEO4J_VECTOR')):
            monkeypatch.delenv(key, raising=False)
    for key, value in paper_environment_defaults().items():
        monkeypatch.setenv(key, value)
    for key, value in {'RAG_INFERENCE_BASE_URL': 'http://litellm.test/v1', 'RAG_INFERENCE_API_KEY': 'synthetic',
                       'RAG_GENERATION_MODEL': 'gemma-4-31b-it', 'RAG_EMBEDDING_MODEL': 'qwen3-embedding-4b',
                       'RAG_PAPER_MODE': 'true', 'RAG_SKIP_PROJECT_ENV': 'true'}.items():
        monkeypatch.setenv(key, value)


def test_current_corpus_revalidates_manifest_without_rereading_source_bytes(tmp_path, monkeypatch):
    from core import admission
    corpus = tmp_path / 'data/multihoprag_corpus'
    corpus.mkdir(parents=True)
    source = corpus / 'a.txt'
    source.write_text('Title: A\n\nOriginal source')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = [{'source_id': 'a', 'filename': 'a.txt', 'content_sha256': digest}]
    manifest = {'schema_version': 2, 'paragraph_count': 1,
                'source_ids_sha256': hashlib.sha256(b'a').hexdigest(),
                'corpus_records_sha256': identity_sha256(records),
                'corpus_files_sha256': hashlib.sha256(f'a.txt\0{digest}'.encode()).hexdigest()}
    manifest['fingerprint'] = identity_sha256(manifest)
    path = corpus / 'corpus_manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(admission, 'ROOT', tmp_path)
    first = admission.current_corpus_identity('multihoprag')
    assert first['fingerprint'] == manifest['fingerprint']
    source.write_text('Changed source')
    assert admission.current_corpus_identity('multihoprag')['fingerprint'] == first['fingerprint']
    source.write_text('Title: A\n\nOriginal source')
    manifest['paragraph_count'] = 2
    path.write_text(json.dumps(manifest))


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['generate_response'])
async def test_per_request_model_override_never_reaches_client(monkeypatch, method):
    import types

    from core.vllm_client import VLLMClient
    _transport(monkeypatch)
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        raise AssertionError('unregistered request reached transport')
    fake = types.SimpleNamespace(base_url='http://litellm.test/v1',
                                 chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(VLLMClient, 'client', property(lambda _: fake))
    VLLMClient()
    assert calls == []


def test_target_empty_seed_and_query_instruction_survive_dotenv_reload(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    from core.paper_policy import configure_target_environment
    from core.strategy_registry import PAPER_TRANSPORT
    _transport(monkeypatch)
    fixture = tmp_path / 'synthetic.env'
    fixture.write_text('RAG_LLM_SEED=42\nEMBEDDING_QUERY_INSTRUCTION=shared_default\n')
    configure_target_environment('prehop', 'hotpotqa', 'run')
    load_dotenv(fixture, override=False)
    transport = InferenceTransport.resolve('prehop')
    assert transport.generation_seed is None
    assert transport.embedding_query_instruction == PAPER_TRANSPORT.query_instruction


@pytest.mark.asyncio
async def test_paper_request_uses_transport_seed_after_late_environment_resolution(monkeypatch):
    import types

    from core.vllm_client import VLLMClient
    _transport(monkeypatch)
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content='answer'))])
    fake = types.SimpleNamespace(base_url='http://litellm.test/v1',
                                 chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(VLLMClient, 'client', property(lambda _: fake))
    client = VLLMClient()
    await client.generate_response([{'role': 'user', 'content': 'question'}])
    assert 'seed' not in calls[0]
    await client.generate_response([{'role': 'user', 'content': 'question'}], seed=41)
    assert len(calls) == 2
    assert all('seed' not in call for call in calls)
