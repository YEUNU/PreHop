"""Native configuration/answer contracts and observational extraction validation."""
import ast
import copy
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.paper_policy import method_environment_defaults, preserve_method_environment
from models.external_research.drivers.linear_rag import LinearNativeInference, LinearRAGDriver
from models.external_research.drivers.youtu_graphrag import classify_native_extraction

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('attribute', [{}, {'nested': ['bad']}, [], 3, None, '', ' '])
def test_nested_native_attributes_are_rejected_without_repair(attribute):
    value = {'attributes': {'entity': [attribute]}, 'triples': []}
    before = copy.deepcopy(value)
    assert classify_native_extraction(value) == 'malformed'
    assert value == before


def test_valid_native_string_attributes_remain_unchanged():
    value = {'attributes': {'entity': ['scientist']}, 'triples': []}
    before = copy.deepcopy(value)
    assert classify_native_extraction(value) == 'success'
    assert value == before


def test_method_sentinels_are_generic_and_preserve_explicit_overrides(tmp_path):
    path = tmp_path / '.env'
    path.write_text('RAG_MS_CONCURRENT_REQUESTS=secret-not-read\nRAG_LINEAR_RAG_UNKNOWN=future\n')
    defaults = method_environment_defaults(path)
    assert defaults['RAG_MS_CONCURRENT_REQUESTS'] == ''
    assert defaults['RAG_LINEAR_RAG_UNKNOWN'] == ''
    env = {'RAG_MS_CONCURRENT_REQUESTS': 'explicit'}
    preserve_method_environment(env, path)
    assert env['RAG_MS_CONCURRENT_REQUESTS'] == 'explicit'
    assert env['RAG_LINEAR_RAG_UNKNOWN'] == ''
    from unittest.mock import patch

    from core.paper_policy import validate_paper_semantic_environment
    with patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError, match='unknown paper method'):
        validate_paper_semantic_environment('ms_graphrag', 'musique')


def _runtime(strategy):
    home = os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME')
    if not home:
        pytest.skip('actual pinned runtime not selected')
    return Path(home) / strategy


def test_exact_native_linear_qa_owns_prompt_parser_and_one_retrieval():
    runtime = _runtime('linear_rag')
    tree = ast.parse((runtime / 'source/src/LinearRAG.py').read_text())
    native = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LinearRAG')
    qa = next(node for node in native.body if isinstance(node, ast.FunctionDef) and node.name == 'qa')
    namespace = {'ThreadPoolExecutor': ThreadPoolExecutor, 'tqdm': lambda values, **kwargs: values}
    exec(compile(ast.Module(body=[qa], type_ignores=[]), '<exact-pinned-LinearRAG.qa>', 'exec'), namespace)  # noqa: S102 - execute exact approved native method in test isolation
    calls = []
    client = LinearNativeInference.__new__(LinearNativeInference)
    client.transport = SimpleNamespace(generation_model='gemma-4-31b-it', generation_seed=42)
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Thought: native\nAnswer: native answer'))])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    retrieval_calls = []
    def retrieve(questions):
        retrieval_calls.append(questions)
        return [{'question': questions[0]['question'], 'sorted_passage': ['1:B\nb', '0:A\na'], 'sorted_passage_scores': [.9, .8]}]
    engine = SimpleNamespace(retrieve=retrieve, llm_model=client, config=SimpleNamespace(max_workers=1),
                             graph=SimpleNamespace(vcount=lambda: 1))
    engine.qa = lambda questions: namespace['qa'](engine, questions)
    driver = LinearRAGDriver.__new__(LinearRAGDriver)
    driver.engine = engine
    driver.passages = ['0:A\na', '1:B\nb']
    driver.by_index = {0: {'source_id': 'a', 'title': 'A', 'text': 'a'}, 1: {'source_id': 'b', 'title': 'B', 'text': 'b'}}
    result = driver.query('question')
    assert len(retrieval_calls) == len(calls) == 1
    assert result['answer'] == 'native answer'
    assert [row['source_id'] for row in result['documents']] == ['b', 'a']
    assert calls[0]['max_tokens'] == 2000 and calls[0]['seed'] == 42
    assert calls[0]['messages'][1]['content'].startswith('1:B\nb\n0:A\na\n')


def test_exact_gfm_hydra_config_parent_resolves_native_defaults(tmp_path):
    runtime = _runtime('gfm_rag')
    script = f'''
import sys,socket
from pathlib import Path
sys.path[:0]=[{str(runtime / 'source')!r},{str(ROOT)!r}]
def deny(*args,**kwargs): raise AssertionError('network forbidden')
socket.socket.connect=deny
from hydra import compose,initialize_config_dir
with initialize_config_dir(config_dir={str(runtime / 'source/gfmrag/workflow/config/gfm_rag')!r},version_base=None):
 cfg=compose(config_name='qa_ircot_inference')
 assert cfg.graph_retriever.graph_constructor._target_.endswith('.KGConstructor')
 assert cfg.graph_retriever.ner_model._target_.endswith('.LLMNERModel')
 assert cfg.graph_retriever.el_model._target_.endswith('.ColbertELModel')
 from core.runtime_requirements import runtime_requirement
 from models.external_research.drivers.gfm_rag import configure_entity_linker
 from hydra.utils import instantiate
 model_path=Path({str(runtime / 'artifacts')!r})/runtime_requirement('gfm_rag')['entity_linker_snapshot_subdir']
 output=Path({str(tmp_path / 'new-run')!r})
 configure_entity_linker(cfg.graph_retriever,model_path,output)
 expected=output/'artifacts/gfm_entity_linker'
 assert cfg.graph_retriever.graph_constructor.el_model.root==str(expected)
 native=instantiate(cfg.graph_retriever.el_model)
 assert native.root==str(expected)
 assert native._index_root('test-fingerprint').is_relative_to(expected)
 assert native._metadata_path('test-fingerprint').is_relative_to(expected)
 assert not Path({str(tmp_path / 'tmp')!r}).exists()
 print('resolved')
'''
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', LITELLM_LOCAL_MODEL_COST_MAP='True')
    result = subprocess.run([str(runtime / 'venv/bin/python'), '-c', script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'resolved' in result.stdout


@pytest.mark.parametrize('shielded', [False, True])
def test_actual_ms_import_loads_synthetic_dotenv_and_preserves_method_sentinels(tmp_path, shielded):
    import importlib.util
    import shutil
    spec = importlib.util.find_spec('litellm')
    if spec is None or not spec.origin:
        pytest.skip('actual MS LiteLLM runtime is not installed')
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
    dotenv = tmp_path / '.env'
    dotenv.write_text('PREHOP_MS_DOTENV_MARKER=loaded\nRAG_MS_CONCURRENT_REQUESTS=poison\nRAG_LINEAR_RAG_FUTURE_ALIAS=poison\n')
    code = f'''
import os,sys,socket
sys.path[:0]=[{str(tmp_path)!r},{str(ROOT)!r}]
def deny(*args,**kwargs): raise AssertionError('network forbidden during import')
socket.socket.connect=deny
from core.paper_policy import preserve_method_environment
if {shielded!r}: preserve_method_environment(env_path=__import__('pathlib').Path({str(dotenv)!r}))
from graphrag.api.index import build_index
import litellm
assert litellm.__file__=={str(shadow / '__init__.py')!r}
assert os.environ['PREHOP_MS_DOTENV_MARKER']=='loaded'
assert os.environ['RAG_MS_CONCURRENT_REQUESTS']=={'' if shielded else 'poison'!r}
assert os.environ['RAG_LINEAR_RAG_FUTURE_ALIAS']=={'' if shielded else 'poison'!r}
print('native_ms_dotenv_observed')
'''
    environment = os.environ.copy()
    for name in ('RAG_MS_CONCURRENT_REQUESTS', 'RAG_LINEAR_RAG_FUTURE_ALIAS', 'PREHOP_MS_DOTENV_MARKER'):
        environment.pop(name, None)
    environment.update(PYTHONDONTWRITEBYTECODE='1', LITELLM_MODE='DEV', LITELLM_DEV_ENV_HOT_RELOAD='False',
                       LITELLM_LOCAL_MODEL_COST_MAP='True')
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'native_ms_dotenv_observed' in result.stdout


def test_actual_linear_sdk_uses_typed_transport_without_network(tmp_path):
    runtime = _runtime('linear_rag')
    script = f'''
import sys,os,json,hashlib
sys.path.insert(0,{str(ROOT)!r})
import httpx,openai
from core.strategy_registry import paper_environment_defaults
from core.inference_transport import _FORBIDDEN_AMBIENT_PROVIDER_KEYS
for name in _FORBIDDEN_AMBIENT_PROVIDER_KEYS: os.environ[name]=''
os.environ.update(paper_environment_defaults())
os.environ.update(RAG_PAPER_MODE='true',RAG_INFERENCE_BASE_URL='http://litellm.test/v1',RAG_INFERENCE_API_KEY='synthetic',RAG_GENERATION_MODEL='gemma-4-31b-it',RAG_EMBEDDING_MODEL='qwen3-embedding-4b',RAG_LLM_SEED='42')
import core.inference_transport as transport
transport._approved_gateway_identity=lambda:hashlib.sha256(b'http://litellm.test/v1').hexdigest()
requests=[]
def record(request):
 requests.append(request)
 return httpx.Response(200,json={{'id':'native','object':'chat.completion','created':0,'model':'gemma-4-31b-it','choices':[{{'index':0,'message':{{'role':'assistant','content':'Answer: native'}},'finish_reason':'stop'}}]}})
native_openai=openai.OpenAI
kwargs_seen=[]
def configured(**kwargs):
 kwargs_seen.append(kwargs)
 return native_openai(**kwargs,http_client=httpx.Client(transport=httpx.MockTransport(record)))
openai.OpenAI=configured
from models.external_research.drivers.linear_rag import LinearNativeInference
client=LinearNativeInference()
try:
 assert client.infer([{{'role':'user','content':'native prompt'}}])=='Answer: native'
 assert len(requests)==1 and str(requests[0].url)=='http://litellm.test/v1/chat/completions'
 payload=json.loads(requests[0].content)
 assert payload['model']=='gemma-4-31b-it' and payload['seed']==42
 assert payload['max_tokens']==2000 and payload['temperature']==0
 assert kwargs_seen[0]['timeout']==600 and kwargs_seen[0]['max_retries']==5
 print('actual_sdk_transport_verified')
finally: client.close()
'''
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true')
    result = subprocess.run([str(runtime / 'venv/bin/python'), '-c', script], cwd=tmp_path, env=env,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'actual_sdk_transport_verified' in result.stdout
