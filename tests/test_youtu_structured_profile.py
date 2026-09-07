"""Construction-only official SDK injection and versioned extraction identity."""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.native_structured_profile import YoutuConstructionClient, youtu_profile_sha256, youtu_response_format
from core.paper_policy import canonical_semantic_index_policy
from models.external_research.official_indexer import validate_native_generation_profile

ROOT = Path(__file__).resolve().parents[1]


def test_facade_returns_identical_sdk_response_without_shared_mutation():
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
        content='{"attributes":{},"triples":[],"entity_types":{},"new_schema_types":{"nodes":[],"relations":[],"attributes":[]}}'))])
    seen = []
    def create(**kwargs):
        seen.append(kwargs)
        kwargs['response_format']['json_schema']['schema']['properties'].clear()
        return response
    facade = YoutuConstructionClient(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        assert facade.chat.completions.create(messages=[], model='gemma-4-31b-it', temperature=.3) is response
    assert len(seen) == 2
    assert 'attributes' in youtu_response_format()['json_schema']['schema']['properties']
    for conflict in ('response_format', 'extra_body', 'guided_json', 'structured_outputs'):
        with pytest.raises(ValueError, match='cannot be overridden'):
            facade.chat.completions.create(**{conflict: {}})
    assert len(seen) == 2


def test_youtu_old_observation_cannot_admit_new_schema_profile():
    policy = canonical_semantic_index_policy('youtu_graphrag', 'musique')
    assert policy['semantic_config_id'] == 'youtu-native-agent-observation-v5'
    from core.native_structured_profile import native_youtu_profile_sha256
    assert policy['extraction_schema_sha256'] == native_youtu_profile_sha256()
    assert policy['schema_sha256'] != policy['extraction_schema_sha256']
    with pytest.raises(RuntimeError, match='[Ee]ffective extraction'):
        validate_native_generation_profile('youtu_graphrag', {}, policy)
    validate_native_generation_profile('youtu_graphrag', policy, policy)


def test_actual_native_llm_consumer_sdk_keeps_query_unformatted(tmp_path):
    home = os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME')
    if not home:
        pytest.skip('actual pinned Youtu runtime not selected')
    runtime = Path(home) / 'youtu_graphrag'
    script = f'''
import sys,json,os
sys.path[:0]=[{str(runtime/'source')!r},{str(ROOT)!r}]
os.environ.update(LLM_MODEL='gemma-4-31b-it',LLM_BASE_URL='http://gateway.test/v1',LLM_API_KEY='synthetic',OPENAI_PROVIDER='openai')
from utils.call_llm_api import LLMCompletionCall
from core.native_structured_profile import YoutuConstructionClient,youtu_response_format
import httpx
from openai import OpenAI
requests=[]
def respond(request):
 requests.append(json.loads(request.content))
 return httpx.Response(200,json={{'id':'test','object':'chat.completion','created':0,'model':'gemma-4-31b-it','choices':[{{'index':0,'message':{{'role':'assistant','content':'{{"attributes":{{}},"triples":[],"entity_types":{{}},"new_schema_types":{{"nodes":[],"relations":[],"attributes":[]}}}}'}},'finish_reason':'stop'}}]}})
builder=LLMCompletionCall(); query=LLMCompletionCall()
builder.client.close();query.client.close()
builder.client=OpenAI(base_url='http://gateway.test/v1',api_key='synthetic',http_client=httpx.Client(transport=httpx.MockTransport(respond))).with_options(timeout=37,max_retries=2)
query.client=OpenAI(base_url='http://gateway.test/v1',api_key='synthetic',http_client=httpx.Client(transport=httpx.MockTransport(respond))).with_options(timeout=37,max_retries=2)
builder.client=YoutuConstructionClient(builder.client)
try:
 assert json.loads(builder.call_api('native construction prompt'))['triples']==[]
 query.call_api('native retrieval prompt')
 assert len(requests)==2
 assert requests[0]['response_format']==youtu_response_format()
 assert 'response_format' not in requests[1]
 for payload in requests:
  assert payload['temperature']==.3 and payload['model']=='gemma-4-31b-it'
  assert 'seed' not in payload and 'max_tokens' not in payload and 'max_completion_tokens' not in payload
 assert builder.client.timeout==37 and builder.client.max_retries==2
 print('native_builder_only_schema_passed')
finally:builder.client.close();query.client.close()
'''
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true')
    result = subprocess.run([str(runtime/'venv/bin/python'), '-c', script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'native_builder_only_schema_passed' in result.stdout


@pytest.mark.parametrize('content', [
    '```json\n{}\n```',
    '{"attributes":{"Ada":[{"name":"London"}]},"triples":[],"entity_types":{},"new_schema_types":{"nodes":[],"relations":[],"attributes":[]}}',
    '{"attributes":{},"triples":[["Ada","London"]],"entity_types":{},"new_schema_types":{"nodes":[],"relations":[],"attributes":[]}}',
    '{"attributes":{},"triples":[],"entity_types":{},"new_schema_types":{"nodes":[],"relations":[],"attributes":[]},"extra":true}',
])
def test_native_cleaner_cannot_repair_invalid_raw_constrained_response(content):
    from core.native_structured_profile import validate_youtu_raw_response

    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content))])
    with pytest.raises(ValueError):
        validate_youtu_raw_response(response)
