"""Resume the existing HopRAG node cache after bounded edge-scoring repair."""
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CAM='hoprag-fixed-multihoprag-20260909'
BASE=ROOT/'data/results'/CAM

def save(value):
    p=BASE/'status.json';tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def configure_recovery_environment():
    """Discard retired launch overrides and use the registered native policy."""
    from core.paper_policy import configure_target_environment
    from core.strategy_registry import get_strategy
    for name in ('RAG_HOP_MAX_THREADS', 'RAG_HOP_GATHER_WAVE',
                 'RAG_HOP_BUILD_CONCURRENCY', 'RAG_HOP_SEMANTIC_VARIANT',
                 'RAG_HOP_EDGE_FILTER'):
        os.environ.pop(name, None)
    for _field, name, value in get_strategy('hoprag').paper_index_environment:
        os.environ[name] = str(value).lower() if isinstance(value, bool) else str(value)
    configure_target_environment('hoprag','multihoprag',CAM)

async def index():
    configure_recovery_environment()
    from cli.index import run_indexing
    from core.neo4j_service import Neo4jService
    path=Path(os.environ['RAG_INDEX_STATS_PATH'])
    old=json.loads(path.read_text());previous=old.get('timing_seconds',{}).get('total_elapsed_seconds',0)
    backup=BASE/f'pre-edge-recovery-index-stats-{time.time_ns()}.json'
    backup.write_text(json.dumps(old,indent=2)+'\n')
    try:
        await run_indexing(str(ROOT/'data/multihoprag_corpus'),'hoprag','default','multihoprag')
        stats=json.loads(path.read_text())
        if stats.get('status')!='complete':raise RuntimeError('Resumed index incomplete')
        current=stats['timing_seconds']['total_elapsed_seconds']
        interrupted=json.loads((BASE/'interrupted-edge-attempts.json').read_text()) if (BASE/'interrupted-edge-attempts.json').exists() else []
        interrupted_seconds=sum(a['wall_seconds'] for a in interrupted)
        stats['resumed_index_cost']={'previous_attempt_wall_seconds':previous,
          'interrupted_edge_attempts':interrupted,'interrupted_edge_wall_seconds':interrupted_seconds,
          'recovery_wall_seconds':current,'continuous':False,
          'scope':'node construction plus failed edge attempt plus recovery'}
        stats['timing_seconds']['total_elapsed_seconds']=previous+interrupted_seconds+current
        stats['edge_execution']='bounded_exhaustive_cached_answers_v2'
        stats['edge_block_size']=int(os.environ.get('RAG_HOP_EDGE_BLOCK_SIZE','128'))
        path.write_text(json.dumps(stats,indent=2)+'\n')
    finally:await Neo4jService.global_close()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    os.environ['HOPRAG_CAMPAIGN']=CAM
    save({'state':'running','phase':'index-recovery','pid':os.getpid(),'updated_at':time.time()})
    try:
        asyncio.run(index())
        save({'state':'running','phase':'benchmark','pid':os.getpid(),'updated_at':time.time()})
        subprocess.run([sys.executable,'-B','scripts/run_hoprag_scheduled.py','benchmark'],check=True)
        save({'state':'admitted','updated_at':time.time()})
    except BaseException:
        save({'state':'failed','phase':'recovery','updated_at':time.time()})
        raise
