import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from models.hoprag.exact_edges import exact_edges

SOURCE=Path('data/runtime_envs/hoprag-paper-20260908/source/third_party/HopRAG/HopBuilder.py')
def dense(p,a):return (np.array(p.embedding.tolist())@np.array(a.embedding.tolist()).T).flatten().tolist()
def sparse(a,b):return len(a&b)/len(a|b)
class Session:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def run(self,*args):pass

def native():
    if not SOURCE.exists():pytest.skip('Pinned native source unavailable')
    if int(pd.__version__.split('.')[0]) >= 3:pytest.skip('Native parity requires the pinned HopRAG pandas 2 runtime')
    tree=ast.parse(SOURCE.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='QABuilder')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='create_edge')
    def sparse_df(df):
        sets=list(df.keywords);return {(str(a),str(b)):sparse(a,b) for a in sets for b in sets}
    ns={'pd': pd,'np': np,'pending_dot_answerable': dense,'sparse_similarities_df': sparse_df,
        'create_pending2answerable': '','create_abstract2answerable': ''}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)  # noqa: S102 - execute the pinned upstream function for parity testing
    return ns['create_edge']

@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('ties',[False,True])
def test_matches_native_exhaustive_selection(seed,ties):
    rng=np.random.default_rng(seed);data={};docs={}
    for i in range(9):
        doc=f'doc{i//3}';docs.setdefault(doc,[]).append(i)
        data[i,doc]={label:[(f'{label}{i if seed%2 else i%4}_{j}',{str(i%3),'common'},(np.ones(5) if ties else rng.normal(size=5))) for j in range(2)] for label in ['pending','answerable']}
    obj=SimpleNamespace(driver=SimpleNamespace(session=Session));native()(obj,data,docs)
    actual,abstract=exact_edges(data,docs,dense,sparse,chunk_size=2)
    cols=['node_id_x','node_id_y','question_y','keywords_both']
    assert actual[cols].to_dict('records')==obj.edges[cols].to_dict('records')
    np.testing.assert_allclose(actual.similarity,obj.edges.similarity,rtol=1e-12)
    assert abstract.question.tolist()==obj.abstract2chunk.question.tolist()
