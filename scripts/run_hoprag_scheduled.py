"""Run a fresh HopRAG canary, full index and benchmark in the owned queue slot."""
from __future__ import annotations
import asyncio,json,logging,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CAMPAIGN=os.environ.get('HOPRAG_CAMPAIGN','hoprag-native-multihoprag-20260908')
BASE=ROOT/'data/results'/CAMPAIGN

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def phase(name):
    from core.paper_policy import configure_target_environment
    run_id=CAMPAIGN+('-canary' if name=='canary' else '')
    configure_target_environment('hoprag','multihoprag',run_id)
    os.environ.update(RAG_BENCHMARK_TIMESTAMP=run_id,RAG_CHUNK_CACHE='off',RAG_EMBEDDING_CACHE='off',
                      RAG_JUDGE_ENABLED='false',RAG_JUDGE_BATCH='false')
    from scripts.check_paper_runtime import check
    check('hoprag','multihoprag')
    from core.neo4j_service import Neo4jService
    async def execute():
        try:
            if name in {'canary','index'}:
                from scripts.paper_cold_canary import ensure_fresh_namespace
                from cli.index import run_indexing
                await ensure_fresh_namespace('hoprag','multihoprag')
                corpus=ROOT/'data/multihoprag_corpus'
                if name=='canary':
                    from scripts.index_pilot_fixture import stage_index_pilot
                    corpus=stage_index_pilot(BASE/'canary','multihoprag',count=16)
                await run_indexing(str(corpus),'hoprag','default','multihoprag')
                stats=Path(os.environ['RAG_INDEX_STATS_PATH'])
                result=json.loads(stats.read_text())
                if result['status']!='complete':raise RuntimeError('HopRAG index did not complete')
                if name=='canary':
                    from core.execution_profile import measurement_slot
                    from models.hoprag.hoprag_adapter import HopRAGAdapter
                    from cli.benchmark import _verify_active_index_snapshot
                    with measurement_slot('hoprag'):
                        engine=HopRAGAdapter(corpus_tag='multihoprag')
                        manifest={**json.loads((corpus/'corpus_manifest.json').read_text()),'path':str(corpus/'corpus_manifest.json')}
                        snapshot=await _verify_active_index_snapshot(engine,'hoprag','multihoprag',manifest,strict=True)
                        query=json.loads((ROOT/'data/multihoprag_queries.json').read_text())[0]
                        answer,sources,trace=await engine.run_workflow(query['query'])
                    save(BASE/'canary/evidence.json',{'status':'passed','answer':answer,'sources':sources,'trace':trace,'active_snapshot':snapshot})
                else:
                    from core.admission import sha256_file
                    from core.amortized_cost import indexing_cost,validate_cost
                    cost=indexing_cost(result);validate_cost(result.get('amortized_indexing_cost'),cost)
                    if not cost['continuous_run_eligible']:raise RuntimeError('HopRAG indexing cost is incomplete')
                    save(BASE/'completion.json',{'status':'index_complete','strategy':'hoprag','dataset':'multihoprag',
                        'run_id':run_id,'source_count':609,'stats_path':str(stats),'stats_sha256':sha256_file(stats),
                        'amortized_indexing_cost':cost,'benchmark_admitted':False})
            else:
                from cli.benchmark import run_benchmark
                await run_benchmark(str(ROOT/'data/multihoprag_queries.json'),'hoprag','default',
                    corpus_tag='multihoprag',output_dir=BASE,seed=42)
        finally:
            await Neo4jService.global_close()
    asyncio.run(execute())
    if name=='benchmark':
        subprocess.run([sys.executable,'-B','scripts/record_paper_completion.py',CAMPAIGN,'multihoprag','hoprag',
                        '--exact-run-id','--output',str(BASE/'admission.json')],check=True,cwd=ROOT)

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if len(sys.argv)>1:
        phase(sys.argv[1]);return
    os.environ.update(PYTHON_BIN=sys.executable,UV_PROJECT_ENVIRONMENT=sys.prefix,RAG_SKIP_PROJECT_ENV='true',PYTHONDONTWRITEBYTECODE='1')
    from models.hoprag.native_runtime import validate_runtime
    from scripts.index_matrix import source_digest
    validate_runtime()
    for name in ['canary','index','benchmark']:
        logging.info('Execution source: %s', source_digest())
        save(BASE/'status.json',{'state':'running','phase':name,'updated_at':time.time(),'pid':os.getpid()})
        with (BASE/(name+'.log')).open('x') as log:
            code=subprocess.call([sys.executable,'-B',__file__,name],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        if code:
            save(BASE/'status.json',{'state':'failed','phase':name,'exit_code':code,'updated_at':time.time()})
            raise SystemExit(code)
    save(BASE/'status.json',{'state':'admitted','updated_at':time.time()})

if __name__=='__main__':main()
