"""Infrastructure bindings for the unchanged, pinned upstream HopRAG runtime."""
from __future__ import annotations

import contextlib
import contextvars
import importlib
import io
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_HOME = ROOT / 'data/runtime_envs/hoprag-paper-20260908'
UPSTREAM = RUNTIME_HOME / 'source/third_party/HopRAG'
REVISION = 'a6e425b8f8a5d8131dd7805db40185ac76e09903'
_pos_lock = threading.Lock()
_pos_process = None
_setup_tag = None
_audit_lock = threading.Lock()
QUERY=contextvars.ContextVar('hoprag_query',default=None)
_usage={'generation_calls':0,'embedding_calls':0,'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,
        'reported_cost':0.0,'token_usage_complete':True,'cost_complete':True}

def usage_snapshot():
    with _audit_lock:return dict(_usage)

def observe_usage(kind,payload):
    from core.inference_telemetry import record
    usage=payload.get('usage') if isinstance(payload,dict) else None
    response_cost=payload.get('response_cost') if isinstance(payload,dict) else None
    record(kind,types.SimpleNamespace(usage=types.SimpleNamespace(**usage) if isinstance(usage,dict) else None,
                                     response_cost=response_cost))
    with _audit_lock:
        _usage[kind+'_calls']+=1
        if isinstance(usage,dict):
            for key in ['prompt_tokens','completion_tokens','total_tokens']:
                _usage[key]+=int(usage.get(key) or 0)
        else:_usage['token_usage_complete']=False
        if isinstance(response_cost,(int,float)):_usage['reported_cost']+=response_cost
        else:_usage['cost_complete']=False


def validate_runtime():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=UPSTREAM, text=True).strip()
    if revision != REVISION or dirty:
        raise RuntimeError('HopRAG upstream revision or source bytes changed')
    import hashlib
    for name,digest in json.loads((ROOT/'configs/hoprag_pos_model.json').read_text()).items():
        if hashlib.sha256((RUNTIME_HOME/'pos-model'/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('HopRAG POS model content drifted')
    for prefix in [Path(sys.prefix), RUNTIME_HOME / 'pos-env']:
        frozen = prefix.parent / (prefix.name + '.freeze.txt')
        actual = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(prefix / 'bin/python')], text=True)
        if sorted(actual.splitlines()) != sorted(frozen.read_text().splitlines()):
            raise RuntimeError('HopRAG pinned runtime dependencies drifted')
        subprocess.run(['uv','pip','check','--python',str(prefix/'bin/python')],check=True,capture_output=True)


def _tag(text):
    global _pos_process
    with _pos_lock:
        if _pos_process is None:
            log = Path(os.environ['RAG_HOP_OUTPUT_ROOT']) / 'pos.stderr.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                _pos_process = subprocess.Popen([str(RUNTIME_HOME/'pos-env/bin/python'), '-B', str(ROOT/'models/hoprag/pos_worker.py')],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stream, text=True, bufsize=1,
                    env={**os.environ, 'OMP_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':''})
        if _pos_process.poll() is not None:
            raise RuntimeError('Native HopRAG POS worker stopped')
        _pos_process.stdin.write(json.dumps(text)+'\n');_pos_process.stdin.flush()
        line = _pos_process.stdout.readline()
        if not line:
            raise RuntimeError('Native HopRAG POS worker returned no response')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value['result']


def _taskflow(name, *args, **kwargs):
    if name != 'pos_tagging' or args or kwargs:
        raise RuntimeError('Unexpected upstream Taskflow request')
    return _tag


def _unused(*args, **kwargs):
    raise RuntimeError('Unused local generation/embedding loader was invoked')


def _install_import_bridges():
    # Only the provider-specific local loaders are unavailable. POS calls use
    # the actual unchanged PaddleNLP implementation in its pinned CPU process.
    for name, attrs in [('paddlenlp', {'Taskflow':_taskflow}),
                        ('sentence_transformers', {'SentenceTransformer':_unused}),
                        ('modelscope', {'AutoModelForCausalLM':_unused,'AutoTokenizer':_unused,'AutoModelForSequenceClassification':_unused})]:
        module=types.ModuleType(name)
        for key,value in attrs.items():setattr(module,key,value)
        sys.modules[name]=module


def _observe(response):
    response.read()
    request=response.request
    try:payload=response.json()
    except ValueError:payload={}
    observe_usage('generation',payload)
    entry={'query':QUERY.get(),'request':json.loads(request.content), 'status_code':response.status_code,
           'response':response.text}
    path=Path(os.environ['RAG_HOP_OUTPUT_ROOT'])/'native_calls.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    with _audit_lock, path.open('a') as stream:
        stream.write(json.dumps(entry,ensure_ascii=False)+'\n')


def setup(corpus_tag):
    global _setup_tag
    if _setup_tag is not None:
        if _setup_tag != corpus_tag:raise RuntimeError('HopRAG runtime cannot mix corpus namespaces')
        return
    validate_runtime()
    from core.execution_profile import require_queue
    from core.index_namespace import index_namespace
    from core.inference_transport import InferenceTransport
    transport=InferenceTransport.resolve('hoprag');require_queue('hoprag')
    _install_import_bridges()
    sys.path.insert(0,str(UPSTREAM))
    with contextlib.redirect_stdout(io.StringIO()):config=importlib.import_module('config')
    safe=index_namespace(corpus_tag)
    config.local_model_name=transport.generation_model
    config.query_generator_model=transport.generation_model
    config.traversal_model=transport.generation_model
    config.default_gpt_model=transport.generation_model
    config.local_base=transport.generation_base_url
    config.local_key=transport.api_key
    config.deployment_sign={transport.generation_model:{'base':transport.generation_base_url,'key':transport.api_key}}
    config.embed_model=transport.embedding_model
    config.embed_dim=transport.embedding_dimensions
    config.llm_device='cpu'
    config.corpus_tag=corpus_tag
    config.node_name=f'HO_{safe}';config.edge_name=f'HO_{safe}_p2a'
    config.generator_label=f'HO_{safe}_'
    for name,suffix in [('node_dense_index_name','node_dense_idx'),('edge_dense_index_name','edge_dense_idx'),
                        ('node_sparse_index_name','node_sparse_idx'),('edge_sparse_index_name','edge_sparse_idx')]:
        setattr(config,name,f'HO_{safe}_{suffix}')
    config.neo4j_url=os.environ.get('NEO4J_URI','bolt://localhost:7687')
    config.neo4j_user=os.environ.get('NEO4J_USER','neo4j')
    config.neo4j_password=os.environ.get('NEO4J_PASSWORD','')
    config.neo4j_dbname=os.environ.get('NEO4J_DATABASE','neo4j')
    config.exception_log_path=str(Path(os.environ['RAG_HOP_OUTPUT_ROOT'])/'native_exceptions.log')
    Path(config.exception_log_path).parent.mkdir(parents=True,exist_ok=True)
    # Only replace the schema names baked into native Cypher templates.
    original=importlib.util.spec_from_file_location('_hoprag_original_config',UPSTREAM/'config.py')
    module=importlib.util.module_from_spec(original)
    with contextlib.redirect_stdout(io.StringIO()):original.loader.exec_module(module)
    replacements={module.edge_name:config.edge_name,module.node_name:config.node_name}
    for name,value in vars(config).copy().items():
        if isinstance(value,str) and ('MATCH ' in value or 'CREATE ' in value or 'CALL ' in value):
            for old,new in sorted(replacements.items(),key=lambda x:-len(x[0])):value=value.replace(old,new)
            setattr(config,name,value)
    tool=importlib.import_module('tool')
    import httpx
    from openai import OpenAI
    class ObservedOpenAI(OpenAI):
        def __init__(self,*args,**kwargs):
            kwargs['default_headers']={**kwargs.get('default_headers', {}), 'X-Prehop-Run-ID': os.environ.get('RAG_BENCHMARK_TIMESTAMP', '')}
            kwargs['http_client']=httpx.Client(timeout=transport.timeout_seconds,trust_env=False,event_hooks={'response':[_observe]})
            super().__init__(*args,**kwargs)
    tool.OpenAI=ObservedOpenAI
    from models.hoprag.response_recovery import install
    install(tool, Path(os.environ['RAG_HOP_OUTPUT_ROOT'])/'response_recovery.jsonl')
    from models.hoprag import official_indexer as indexer
    indexer._GEN_API_BASE=transport.generation_base_url
    indexer._EMBED_API_BASE=transport.embedding_base_url
    indexer._GEN_MODEL_NAME=transport.generation_model
    indexer._EMBED_MODEL_NAME=transport.embedding_model
    indexer._GEN_API_KEY=transport.api_key
    embed=indexer._VLLMEmbedClient(transport.embedding_base_url,transport.embedding_model,transport.embedding_dimensions)
    embed._sess.headers['X-Prehop-Run-ID']=os.environ.get('RAG_BENCHMARK_TIMESTAMP', '')
    def observe_embedding(response, *args, **kwargs):
        try:payload=response.json()
        except ValueError:payload={}
        observe_usage('embedding',payload)
        path=Path(os.environ['RAG_HOP_OUTPUT_ROOT'])/'native_embeddings.jsonl'
        with _audit_lock, path.open('a') as stream:
            stream.write(json.dumps({'query':QUERY.get(),'request':json.loads(response.request.body), 'status_code':response.status_code,
                                     'response':response.text})+'\n')
    embed._sess.hooks['response']=[observe_embedding]
    tool.load_embed_model=lambda name:embed
    tool.get_doc_embeds=lambda documents,model:model.encode(documents,normalize_embeddings=True).tolist()
    # Keyword filtering, chunking and edge construction remain native.
    # Response recovery is separately recorded by the adapter.
    _setup_tag=corpus_tag


def native_pipeline():
    module=importlib.import_module('HopGenerator')
    args=module.parser.parse_args([])
    from core.inference_transport import InferenceTransport
    transport=InferenceTransport.resolve('hoprag')
    args.model_name=args.traversal_model=transport.generation_model
    args.embedding_model=transport.embedding_model
    pipeline=module.RagPipeline(args)
    original=pipeline.retriever.emb_model
    class QueryEmbedding:
        def encode(self, documents, **kwargs):
            single=isinstance(documents,str)
            rows=[documents] if single else documents
            formatted=[transport.embedding_query_template.format(instruction=transport.embedding_query_instruction,query=text) for text in rows]
            return original.encode(formatted[0] if single else formatted,**kwargs)
    pipeline.retriever.emb_model=QueryEmbedding()
    return pipeline
