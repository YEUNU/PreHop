"""Versioned native OpenIE configuration, requests and cache separation."""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.paper_policy import canonical_semantic_index_policy, validate_paper_semantic_environment
from core.semantic_config import semantic_config_sha256
from models.external_research.drivers.hipporag2 import configure_hippo_openie

ROOT = Path(__file__).resolve().parents[1]


def test_hippo_structured_profile_changes_semantic_identity():
    policy = canonical_semantic_index_policy('hipporag2', 'musique')
    assert policy['semantic_config_id'] == 'hipporag2-native-observation-v6'
    assert policy['openie_response_format'] == 'json_object'
    assert (policy['openie_ner_max_tokens'], policy['openie_triple_max_tokens']) == (512, 2048)
    old = {key: value for key, value in policy.items() if key not in {
        'semantic_config_id', 'openie_generation_profile', 'openie_response_format',
        'openie_ner_max_tokens', 'openie_triple_max_tokens'}}
    assert semantic_config_sha256(policy) != semantic_config_sha256(old)
    config = SimpleNamespace()
    configure_hippo_openie(config)
    assert config.response_format == {'type': 'json_object'}
    assert config.openie_ner_max_tokens == 512 and config.openie_triple_max_tokens == 2048


@pytest.mark.parametrize(('name', 'value'), [('RAG_HIPPORAG2_OPENIE_PROFILE', 'v1'),
                                           ('RAG_HIPPORAG2_OPENIE_RESPONSE_FORMAT', 'text'),
                                           ('RAG_HIPPORAG2_NER_MAX_TOKENS', '1024'),
                                           ('RAG_HIPPORAG2_TRIPLE_MAX_TOKENS', '4096')])
def test_hippo_profile_overrides_fail_before_native_execution(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match='paper semantic policy'):
        validate_paper_semantic_environment('hipporag2', 'musique')
    with pytest.raises(RuntimeError):
        configure_hippo_openie(SimpleNamespace())


@pytest.mark.parametrize('observed', [{}, {"openie_response_format": None, "openie_ner_max_tokens": 512, "openie_triple_max_tokens": 2048}])
def test_missing_or_legacy_native_observation_cannot_write_new_profile(observed):
    from models.external_research.official_indexer import validate_native_generation_profile

    policy = canonical_semantic_index_policy('hipporag2', 'musique')
    with pytest.raises(RuntimeError, match='[Ee]ffective.*profile'):
        validate_native_generation_profile('hipporag2', observed, policy)
    validate_native_generation_profile('hipporag2', {
        'openie_response_format': {'type': 'json_object'},
        'ner_normalization_profile': 'native-parser-v1',
        'extraction_validation_profile': 'native-observation-v1',
        'openie_ner_max_tokens': 512, 'openie_triple_max_tokens': 2048,
    }, policy)


def test_actual_hippo_openie_object_requests_preserve_parsers_and_separate_cache(tmp_path):
    home = os.environ.get('RAG_TEST_PINNED_RUNTIME_HOME')
    if not home:
        pytest.skip('actual pinned Hippo runtime not selected')
    runtime = Path(home) / 'hipporag2'
    code = f'''
import os,sys,json,types
sys.path[:0]=[{str(runtime / 'source/src')!r},{str(ROOT)!r}]
os.environ['OPENAI_API_KEY']='synthetic'
import httpx
from openai import OpenAI
from hipporag.utils.config_utils import BaseConfig
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.information_extraction.openie_openai import OpenIE
from hipporag import HippoRAG
from models.external_research.drivers.hipporag2 import configure_hippo_openie
config=BaseConfig(llm_name='gemma-4-31b-it',llm_base_url='http://litellm.test/v1',seed=42,temperature=0)
configure_hippo_openie(config)
llm=CacheOpenAI(cache_dir={str(tmp_path / 'cache')!r},global_config=config)
llm.openai_client.close()
requests=[]
def respond(request):
 payload=json.loads(request.content)
 requests.append(payload)
 content='{{"named_entities":["Ada","London"]}}' if payload.get('max_tokens')==512 else '{{"triples":[["Ada","visited","London"]]}}'
 return httpx.Response(200,json={{'id':'native','object':'chat.completion','created':0,'model':'gemma-4-31b-it','choices':[{{'index':0,'message':{{'role':'assistant','content':content}},'finish_reason':'stop'}}],'usage':{{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20}}}})
llm.openai_client=OpenAI(base_url=config.llm_base_url,api_key='synthetic',http_client=httpx.Client(transport=httpx.MockTransport(respond)))
openie=OpenIE(llm,max_workers=1,ner_max_tokens=config.openie_ner_max_tokens,triple_max_tokens=config.openie_triple_max_tokens)
try:
 ner=openie.ner('doc','Ada visited London.')
 assert ner.unique_entities==['Ada','London'] and not ner.metadata.get('error')
 triples=openie.triple_extraction('doc','Ada visited London.',ner.unique_entities)
 assert triples.triples==[['Ada','visited','London']] and not triples.metadata.get('error')
 assert len(requests)==2
 assert all(p['response_format']=={{'type':'json_object'}} and p['seed']==42 for p in requests)
 assert [p['max_tokens'] for p in requests]==[512,2048]
 assert openie.ner('doc','Ada visited London.').metadata['cache_hit'] is True
 assert len(requests)==2
 messages=requests[0]['messages']
 # Identical native messages/cap with no response_format must miss the v2 cache.
 _,_,cache_hit=llm.infer(messages=messages,max_new_tokens=512)
 assert cache_hit is False and len(requests)==3 and 'response_format' not in requests[-1]
 _,_,cache_hit=llm.infer(messages=[{{'role':'user','content':'native QA'}}])
 assert 'response_format' not in requests[-1]
 instance=HippoRAG.__new__(HippoRAG)
 instance.global_config=config
 instance.index_identity='hipporag2-paper-json-object-v2'
 identity=instance._openie_state_identity()
 assert identity['response_format']=={{'type':'json_object'}}
 assert (identity['ner_max_tokens'],identity['triple_max_tokens'])==(512,2048)
 print('native_openie_profile_cache_and_qa_boundary_passed')
finally: llm.openai_client.close()
'''
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true', LITELLM_LOCAL_MODEL_COST_MAP='True')
    result = subprocess.run([str(runtime / 'venv/bin/python'), '-c', code], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'native_openie_profile_cache_and_qa_boundary_passed' in result.stdout
