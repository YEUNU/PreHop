"""Optional real pinned-runtime regressions; no inference and no original artifact writes."""
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from core.strategy_registry import get_strategy, paper_environment_defaults
from models.official_baseline_runtime import _runtime_env
from scripts.build_official_package import stage_source

ROOT = Path(__file__).resolve().parents[1]


def _runtime(strategy):
    home = os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME')
    if not home:
        pytest.skip('set RAG_TEST_PINNED_RUNTIME_HOME for actual pinned-runtime import tests')
    root = Path(home) / strategy
    assert (root / 'source/.git').is_dir()
    assert (root / 'venv/bin/python').is_file()
    return root


def _canonical_child(monkeypatch, strategy):
    monkeypatch.setenv('RAG_OFFICIAL_BASELINE_HOME', os.environ['RAG_TEST_PINNED_RUNTIME_HOME'])
    for name, value in paper_environment_defaults().items():
        monkeypatch.setenv(name, value)
    for name in tuple(os.environ):
        if name.startswith(('VLLM_', 'OPENAI_', 'AZURE_OPENAI')):
            monkeypatch.setenv(name, '')
    for name, value in {'RAG_INFERENCE_BASE_URL': 'http://litellm.test/v1',
                       'RAG_INFERENCE_API_KEY': 'synthetic', 'RAG_GENERATION_MODEL': 'gemma-4-31b-it',
                       'RAG_EMBEDDING_MODEL': 'qwen3-embedding-4b', 'RAG_PAPER_MODE': 'true',
                       'RAG_LLM_SEED': '',
                       'EMBEDDING_QUERY_INSTRUCTION': '',
                       }.items():
        if name == 'EMBEDDING_QUERY_INSTRUCTION':
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr('core.inference_transport._approved_gateway_identity',
                        lambda: hashlib.sha256(b'http://litellm.test/v1').hexdigest())
    return _runtime_env(strategy)


@pytest.mark.parametrize(('strategy', 'native_import'), [('gfm_rag', 'from gfmrag import GFMRetriever\nfrom hydra import compose, initialize_config_module\nfrom hydra.utils import instantiate\nfrom langchain_openai import ChatOpenAI')])
def test_actual_pinned_imports_cannot_repopulate_synthetic_dotenv_aliases(tmp_path, monkeypatch, strategy, native_import):
    runtime = _runtime(strategy)
    staged = stage_source(runtime / 'source', get_strategy(strategy).revision, tmp_path / 'exports')
    # Native dotenv finds this synthetic ancestor of the exact upstream export.
    # The real project dotenv is never read by this isolated import test.
    (tmp_path / '.env').write_text('PREHOP_TEST_DOTENV_MARKER=loaded\nVLLM_URL=http://bypass.invalid/v1\nVLLM_API_KEY=poison\n'
                                  'OPENAI_PROVIDER=azure\nOPENAI_BASE_URL=http://bypass.invalid/v1\n'
                                  'RAG_INFERENCE_TIMEOUT=1\nEMBEDDING_QUERY_INSTRUCTION=poison\nRAG_LLM_SEED=41\n')
    environment = _canonical_child(monkeypatch, strategy)
    import_root = staged
    module_name = 'gfmrag'
    command = f'''
import hashlib,os,sys
from pathlib import Path
sys.path[:0] = [{str(import_root)!r}, {str(ROOT)!r}]
{native_import}
assert Path(sys.modules[{module_name!r}].__file__).is_relative_to(Path({str(staged)!r}))
# GFM's native dotenv aliases must remain shielded.
assert os.environ.get('PREHOP_TEST_DOTENV_MARKER')=={('loaded' if strategy == 'gfm_rag' else None)!r}
import core.inference_transport as transport
transport._approved_gateway_identity=lambda: hashlib.sha256(b"http://litellm.test/v1").hexdigest()
assert all(os.environ.get(name)=="" for name in transport._FORBIDDEN_AMBIENT_PROVIDER_KEYS)
effective=transport.InferenceTransport.resolve({strategy!r})
assert effective.timeout_seconds==600
assert effective.generation_seed==42
assert effective.embedding_query_instruction==transport.PAPER_TRANSPORT.query_instruction
print("native_import_and_transport_ok")
'''
    result = subprocess.run([str(runtime / 'venv/bin/python'), '-c', command], cwd=tmp_path,
                            env=environment, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'native_import_and_transport_ok' in result.stdout


def _inventory(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}
