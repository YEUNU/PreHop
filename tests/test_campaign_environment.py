"""The detached service retains canonical imports without ancestor dotenv reload."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_actual_graphrag_import_after_service_filter_never_loads_ancestor_dotenv(tmp_path):
    spec = importlib.util.find_spec('litellm')
    if spec is None or not spec.origin:
        pytest.skip('MS LiteLLM runtime unavailable')
    package = Path(spec.origin).parent
    shadow = tmp_path / 'litellm'
    shadow.mkdir()
    for source in package.iterdir():
        if source.name == '__pycache__':
            continue
        if source.name == '__init__.py':
            shutil.copy2(source, shadow / source.name)
        else:
            (shadow / source.name).symlink_to(source, target_is_directory=source.is_dir())
    (tmp_path / '.env').write_text('PREHOP_TEST_ANCESTOR=loaded\nVLLM_URL=poison\nVLLM_API_KEY=poison\nRAG_MS_CONCURRENT_REQUESTS=poison\n')
    script = f'''
import os,sys,socket,hashlib
sys.path[:0]=[{str(tmp_path)!r},{str(ROOT)!r}]
from core.inference_transport import _FORBIDDEN_AMBIENT_PROVIDER_KEYS
for key in _FORBIDDEN_AMBIENT_PROVIDER_KEYS: os.environ.pop(key,None)
os.environ.pop('LITELLM_MODE',None)
os.environ.pop('RAG_MS_CONCURRENT_REQUESTS',None)
os.environ.pop('PREHOP_TEST_ANCESTOR',None)
from scripts.paper_campaign import safe_environment
env=safe_environment()
os.environ.clear();os.environ.update(env)
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ.update(RAG_INFERENCE_BASE_URL='http://gateway.test/v1',RAG_INFERENCE_API_KEY='synthetic',RAG_GENERATION_MODEL='gemma-4-31b-it',RAG_EMBEDDING_MODEL='qwen3-embedding-4b')
from core.paper_policy import configure_target_environment
configure_target_environment('ms_graphrag','multihoprag','test-import')
def deny(*a,**k): raise AssertionError('network forbidden')
socket.socket.connect=deny
from graphrag.api.index import build_index
import litellm
assert litellm.__file__=={str(shadow/'__init__.py')!r}
assert os.environ['LITELLM_MODE']=='PRODUCTION'
assert 'PREHOP_TEST_ANCESTOR' not in os.environ
assert not os.environ.get('RAG_MS_CONCURRENT_REQUESTS')
assert all(os.environ.get(key)=='' for key in _FORBIDDEN_AMBIENT_PROVIDER_KEYS)
import core.inference_transport as transport
transport._approved_gateway_identity=lambda:hashlib.sha256(b'http://gateway.test/v1').hexdigest()
from models.ms_graphrag.official_indexer import _apply_shared_transport
_apply_shared_transport()
print('native_ms_import_environment_preserved')
'''
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true',
               RAG_EMBEDDING_BATCH_SIZE='16', NEO4J_VECTOR_DIMENSIONS='2560')
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr
    assert 'native_ms_import_environment_preserved' in result.stdout
