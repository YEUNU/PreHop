"""Executable boundary regressions; all mutable artifacts live in pytest temp dirs."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core.admission import identity_sha256
from core.inference_transport import InferenceTransport
from core.strategy_registry import get_strategy, paper_environment_defaults
from models.official_baseline_runtime import _runtime_env
from scripts import check_paper_runtime

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
    monkeypatch.setattr('core.inference_transport._approved_gateway_identity',
                        lambda: hashlib.sha256(b'http://litellm.test/v1').hexdigest())


def test_snapshot_preflight_reads_and_checks_real_manifest(tmp_path, monkeypatch):
    source = tmp_path / 'linear_rag/source'
    root = source.parent / 'artifacts/encoder'
    root.mkdir(parents=True)
    (root / 'config.json').write_text('{}')
    rows = [{'path': 'config.json', 'size': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}]
    manifest = {'schema_version': 1, 'repository_id': 'test/encoder', 'revision': 'revision',
                'file_count': 1, 'total_bytes': 2, 'tree_sha256': identity_sha256(rows)}
    (root / 'artifact_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(check_paper_runtime, 'official_root', lambda _: source)
    assert check_paper_runtime._check_local_snapshot('linear_rag', 'encoder', 'test/encoder', 'revision') == root
    (root / 'config.json').write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='content differs'):
        check_paper_runtime._check_local_snapshot('linear_rag', 'encoder', 'test/encoder', 'revision')
    (root / 'artifact_manifest.json').write_text('{invalid')
    with pytest.raises(RuntimeError, match='manifest is missing or invalid'):
        check_paper_runtime._check_local_snapshot('linear_rag', 'encoder', 'test/encoder', 'revision')


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


@pytest.mark.parametrize('endpoint', ['https://api.openai.com/v1', 'http://arbitrary.test/v1',
                                      'http://user:password@litellm.test/v1'])
def test_paper_gateway_rejects_vendor_mismatch_and_userinfo(monkeypatch, endpoint):
    _transport(monkeypatch, 'prehop')
    monkeypatch.setenv('RAG_INFERENCE_BASE_URL', endpoint)
    with pytest.raises(RuntimeError):
        InferenceTransport.resolve('prehop')


def test_native_unseeded_adapter_initializes_without_generic_synthesis(monkeypatch):
    _transport(monkeypatch, 'lightrag')
    from models.external_research import adapter
    monkeypatch.setattr(adapter, 'get_llm_client', lambda _: pytest.fail('native adapter created generic client'))
    monkeypatch.setattr(adapter, 'OfficialQueryWorker', lambda *args: object())
    instance = adapter.ExternalResearchAdapter('lightrag')
    assert instance.llm is None


def test_paper_model_override_fails_before_request(monkeypatch):
    _transport(monkeypatch, 'prehop')
    from core.vllm_client import VLLMClient
    with pytest.raises(RuntimeError, match='model override'):
        VLLMClient('unregistered')


@pytest.mark.parametrize('strategy', ['lightrag'])
@pytest.mark.parametrize('existing', ['result', 'index'])
def test_target_verifier_observes_run_environment_before_reuse(tmp_path, strategy, existing):
    for folder in ('scripts', 'core', 'bin'):
        (tmp_path / folder).mkdir()
    for relative in ('scripts/run_paper_target.sh', 'scripts/lib.sh', 'core/strategy_registry.py'):
        shutil.copy2(ROOT / relative, tmp_path / relative)
    shutil.copytree(ROOT / 'core', tmp_path / 'core', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path / '.env').write_text('# synthetic test fixture only\n')
    (tmp_path / 'data/musique_corpus').mkdir(parents=True)
    (tmp_path / 'data/musique_corpus/corpus_manifest.json').write_text('{}')
    (tmp_path / 'data/musique_queries.json').write_text('[]')
    run_id = 'arbitrary-safe-id'
    if existing == 'result':
        (tmp_path / f'data/results/{run_id}').mkdir(parents=True)
    else:
        (tmp_path / 'data/index_stats').mkdir()
        (tmp_path / f'data/index_stats/{strategy}_musique_{run_id}.json').write_text('{}')
    uv = tmp_path / 'bin/uv'
    uv.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                  'if sys.argv[1] == "-c": os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n'
                  'keys=["RAG_RUN_ID","RAG_LLM_SEED","RAG_INDEX_NAMESPACE","RAG_INDEX_STATS_PATH",'
                  '"RAG_LIGHTRAG_OUTPUT_ROOT","RAG_EMBEDDING_BATCH_SIZE"]\n'
                  'with open(os.environ["TRACE"],"a") as f: f.write(json.dumps({"argv":sys.argv[1:],'
                  '"env":{k:os.environ.get(k) for k in keys}})+"\\n")\n')
    uv.chmod(0o755)
    env = {'PATH': str(tmp_path / 'bin') + os.pathsep + os.defpath,
           'PYTHON_BIN': str(uv), 'TRACE': str(tmp_path / 'trace.jsonl'), 'RAG_SKIP_PROJECT_ENV': 'true',
           'RAG_INFERENCE_BASE_URL': 'http://litellm.test/v1', 'RAG_INFERENCE_API_KEY': 'synthetic',
           'RAG_GENERATION_MODEL': 'gemma-4-31b-it', 'RAG_EMBEDDING_MODEL': 'qwen3-embedding-4b'}
    result = subprocess.run(['bash', 'scripts/run_paper_target.sh', 'musique', strategy, run_id, '--check'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'trace.jsonl').read_text().splitlines()]
    assert calls[0]['argv'][0] == 'scripts/check_paper_runtime.py'
    call = calls[1]
    assert call['env']['RAG_RUN_ID'] == run_id
    assert call['env']['RAG_LLM_SEED'] == ''
    assert call['env']['RAG_INDEX_NAMESPACE'] == f'musique_{run_id}'
    assert call['env']['RAG_INDEX_STATS_PATH'] == f'data/index_stats/{strategy}_musique_{run_id}.json'
    assert call['env'][get_strategy(strategy).output_env] == f'{get_strategy(strategy).output_default}/runs/{run_id}'
    if existing == 'result':
        assert call['argv'][1:5] == [run_id, 'musique', strategy, '--exact-run-id']


def test_current_corpus_revalidates_manifest_and_actual_source_bytes(tmp_path, monkeypatch):
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
    with pytest.raises(ValueError, match='content digest'):
        admission.current_corpus_identity('multihoprag')
    source.write_text('Title: A\n\nOriginal source')
    manifest['paragraph_count'] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fingerprint'):
        admission.current_corpus_identity('multihoprag')


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
    client = VLLMClient()
    with pytest.raises(RuntimeError, match='request model'):
        await getattr(client, method)([{'role': 'user', 'content': 'question'}], model='unregistered')
    assert calls == []


def test_target_empty_seed_and_native_instruction_survive_dotenv_reload(tmp_path, monkeypatch):
    from dotenv import load_dotenv

    from core.paper_policy import configure_target_environment
    _transport(monkeypatch, 'lightrag')
    fixture = tmp_path / 'synthetic.env'
    fixture.write_text('RAG_LLM_SEED=42\nEMBEDDING_QUERY_INSTRUCTION=shared_default\n')
    configure_target_environment('lightrag', 'musique', 'run')
    load_dotenv(fixture, override=False)
    assert InferenceTransport.resolve('lightrag').generation_seed is None


def test_gate_rejects_bound_fabricated_canary_without_actual_policy(tmp_path, monkeypatch):
    from core.strategy_registry import PRIMARY_STRATEGIES
    from scripts import paper_gate_ledger as gate
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    def artifact(name, value):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(value))
        return {'path': path.name, 'sha256': gate.sha256_file(path)}
    targets = {}
    for strategy in PRIMARY_STRATEGIES:
        for dataset in gate.DATASETS:
            name = f'{dataset}-{strategy}'
            identity = {'dataset': dataset, 'strategy': strategy, 'run_id': 'fabricated'}
            index = artifact(name + '-index', {**identity, 'status': 'complete', 'fresh_index': True, 'source_count': 2})
            query = artifact(name + '-query', {**identity, 'answer': 'fabricated', 'documents': [{}], 'query_count': 1})
            admission = artifact(name + '-admission', {**identity, 'status': 'canary_passed', 'errors': [],
                                                       'index_sha256': index['sha256'], 'query_sha256': query['sha256']})
            invocation = artifact(name + '-invocation', {'target': f'{dataset}/{strategy}', 'argv': ['did-not-run'], 'exit_code': 0})
            targets[f'{dataset}/{strategy}'] = artifact(name, {**identity, 'stage': 'cold_canary_16',
                'status': 'canary_passed', 'exit_code': 0, 'invocation': invocation, 'index': index, 'query': query,
                'admission': admission})
    with pytest.raises((RuntimeError, TypeError), match='identity/status|policy'):
        gate._validate_evidence('cold_canary_16', tmp_path / 'evidence.json',
            {'status': 'canary_passed', 'stage': 'cold_canary_16', 'schema_version': 1, 'targets': targets})


def test_submission_verifier_resolves_and_restores_each_target(tmp_path, monkeypatch):
    from scripts import verify_submission_consistency as verifier
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    monkeypatch.setattr(verifier, 'STRATEGIES', ('linear_rag', 'lightrag'))
    for dataset in verifier.DATASETS:
        for strategy in verifier.STRATEGIES:
            path = tmp_path / verifier._artifact_path('campaign', dataset, strategy)
            path.parent.mkdir(parents=True)
            path.write_text('{}')
    observed = []
    def validate(path, payload, *, dataset, strategy, expected_count):
        observed.append((strategy, dataset, os.environ['RAG_RUN_ID'], os.environ.get('RAG_LLM_SEED'),
                         os.environ['EMBEDDING_QUERY_INSTRUCTION'], os.environ[get_strategy(strategy).output_env]))
        return []
    monkeypatch.setattr(verifier, '_validate_artifact', validate)
    monkeypatch.setattr(verifier, '_validate_admission', lambda *args: [])
    before = os.environ.copy()
    verifier.verify('campaign', check_documents=False, check_presentations=False)
    assert os.environ == before
    assert len(observed) == 4
    for strategy, dataset, run_id, seed, instruction, output in observed:
        assert run_id == f'campaign-{dataset}-{strategy}'
        assert output.endswith('/runs/' + run_id)
        assert seed == ''


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


@pytest.mark.parametrize('entrypoint', ['submission', 'gate'])
def test_verifier_cli_loads_canonical_dotenv_before_validation(tmp_path, monkeypatch, entrypoint):
    from scripts import paper_gate_ledger, verify_submission_consistency
    monkeypatch.setattr(check_paper_runtime, 'ROOT', tmp_path)
    monkeypatch.delenv('RAG_SKIP_PROJECT_ENV', raising=False)
    monkeypatch.delenv('RAG_INFERENCE_BASE_URL', raising=False)
    (tmp_path / '.env').write_text('RAG_INFERENCE_BASE_URL=http://synthetic-gateway.test/v1\n')
    seen = []
    def observe(*args, **kwargs):
        seen.append(os.environ.get('RAG_INFERENCE_BASE_URL'))
        return {'status': 'admitted'}
    if entrypoint == 'gate':
        module = paper_gate_ledger
        monkeypatch.setattr(module, 'initialize', observe)
        argv = ['gate', 'init', '--ledger', str(tmp_path / 'ledger.json'), '--campaign', 'synthetic']
    elif entrypoint == 'submission':
        module = verify_submission_consistency
        monkeypatch.setattr(module, 'verify', observe)
        argv = ['submission', '--matrix-prefix', 'synthetic']
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    assert seen == ['http://synthetic-gateway.test/v1']
