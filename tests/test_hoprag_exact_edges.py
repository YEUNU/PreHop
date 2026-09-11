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

@pytest.mark.parametrize('chunk_size',[8,32,64,128])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('seed',range(4))
@pytest.mark.parametrize('ties',[False,True])
def test_matches_native_exhaustive_selection(seed,ties,chunk_size,dtype):
    rng=np.random.default_rng(seed);data={};docs={}
    for i in range(9):
        doc=f'doc{i//3}';docs.setdefault(doc,[]).append(i)
        data[i,doc]={label:[(f'{label}{i if seed%2 else i%4}_{j}',{str(i%3),'common'},(np.ones(5) if ties else rng.normal(size=5)).astype(dtype)) for j in range(2)] for label in ['pending','answerable']}
    obj=SimpleNamespace(driver=SimpleNamespace(session=Session));native()(obj,data,docs)
    actual,abstract=exact_edges(data,docs,dense,sparse,chunk_size=chunk_size,reuse_answer_vectors=True)
    cols=['node_id_x','node_id_y','question_y','keywords_both']
    assert actual[cols].to_dict('records')==obj.edges[cols].to_dict('records')
    np.testing.assert_array_equal(actual.similarity,obj.edges.similarity)
    assert abstract.question.tolist()==obj.abstract2chunk.question.tolist()


def test_answer_array_reused_without_dtype_conversion():
    from models.hoprag.exact_edges import PreparedDense
    frame=pd.DataFrame({'embedding':[np.array([1,2],dtype=np.float32),np.array([3,4],dtype=np.float32)]})
    cached=PreparedDense(frame);identity=id(cached.answerable)
    for _ in range(3):
        result=cached(frame)
        assert result.dtype==np.float32
        np.testing.assert_array_equal(result,[[5,11],[11,25]])
        assert id(cached.answerable)==identity
