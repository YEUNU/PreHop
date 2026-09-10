"""Executable boundary regressions; all mutable artifacts live in pytest temp dirs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.admission import identity_sha256
from core.inference_transport import InferenceTransport
from core.strategy_registry import get_strategy, paper_environment_defaults
from models.official_baseline_runtime import _runtime_env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_process_environment():
    before = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(before)


def _transport(monkeypatch, strategy):
    for key in tuple(os.environ):
        if key.startswith(('RAG_', 'VLLM_', 'OPENAI_', 'AZURE_', 'EMBEDDING_', 'MAX_EMBEDDING', 'NEO4J_VECTOR')):
            monkeypatch.delenv(key, raising=False)
    for key, value in paper_environment_defaults().items():
        monkeypatch.setenv(key, value)
    for key, value in {'RAG_INFERENCE_BASE_URL': 'http://litellm.test/v1', 'RAG_INFERENCE_API_KEY': 'synthetic',
                       'RAG_GENERATION_MODEL': 'gemma-4-31b-it', 'RAG_EMBEDDING_MODEL': 'qwen3-embedding-4b',
                       'RAG_PAPER_MODE': 'true', 'RAG_SKIP_PROJECT_ENV': 'true'}.items():
        monkeypatch.setenv(key, value)
    seed = get_strategy(strategy).paper_generation_seed
    if seed is not None:
        monkeypatch.setenv('RAG_LLM_SEED', str(seed))


@pytest.mark.parametrize('strategy', ['lightrag', 'gfm_rag', 'linear_rag'])
def test_primary_child_can_resolve_actual_parent_environment(monkeypatch, strategy):
    _transport(monkeypatch, strategy)
    environment = _runtime_env(strategy)
    assert not any(value for key, value in environment.items() if key.startswith('VLLM_'))
    code = ('import hashlib; import core.inference_transport as t; '
            't._approved_gateway_identity=lambda: hashlib.sha256(b"http://litellm.test/v1").hexdigest(); '
            f'assert t.InferenceTransport.resolve({strategy!r}).strategy == {strategy!r}')
    result = subprocess.run([sys.executable, '-c', code], env=environment, cwd=ROOT, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def test_native_unseeded_adapter_initializes_without_generic_synthesis(monkeypatch):
    _transport(monkeypatch, 'lightrag')
    from models.external_research import adapter
    monkeypatch.setattr(adapter, 'get_llm_client', lambda _: pytest.fail('native adapter created generic client'))
    monkeypatch.setattr(adapter, 'OfficialQueryWorker', lambda *args: object())
    instance = adapter.ExternalResearchAdapter('lightrag')
    assert instance.llm is None


def test_paper_model_override_fails_before_request(monkeypatch):
    _transport(monkeypatch, 'prehop')


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
@pytest.mark.parametrize('method', ['generate_response', 'generate_eval_json'])
async def test_per_request_model_override_never_reaches_client(monkeypatch, method):
    import types

    from core.vllm_client import VLLMClient
    _transport(monkeypatch, 'prehop')
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        raise AssertionError('unregistered request reached transport')
    fake = types.SimpleNamespace(base_url='http://litellm.test/v1',
                                 chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(VLLMClient, 'client', property(lambda _: fake))
    monkeypatch.setattr(VLLMClient, 'judge_client', property(lambda _: fake))
    VLLMClient()
    assert calls == []


def test_target_empty_seed_and_native_instruction_survive_dotenv_reload(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    from core.paper_policy import configure_target_environment
    _transport(monkeypatch, 'lightrag')
    fixture = tmp_path / 'synthetic.env'
    fixture.write_text('RAG_LLM_SEED=42\nEMBEDDING_QUERY_INSTRUCTION=shared_default\n')
    configure_target_environment('lightrag', 'hotpotqa', 'run')
    load_dotenv(fixture, override=False)
    assert InferenceTransport.resolve('lightrag').generation_seed is None


@pytest.mark.asyncio
async def test_paper_request_uses_transport_seed_after_late_environment_resolution(monkeypatch):
    import types

    from core.vllm_client import VLLMClient
    _transport(monkeypatch, 'prehop')
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(usage=None)
    fake = types.SimpleNamespace(base_url='http://litellm.test/v1',
                                 chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    client = VLLMClient()
    await client._create_generation_request(fake, {'model': 'gemma-4-31b-it'})
    assert 'seed' not in calls[0]
    await client._create_generation_request(fake, {'model': 'gemma-4-31b-it', 'seed': 41})
    assert len(calls) == 2
    assert all('seed' not in call for call in calls)
