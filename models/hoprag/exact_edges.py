"""Bounded exhaustive HopRAG pair scoring; preserve native selection semantics."""
import logging

import numpy as np
import pandas as pd

COLS = ['node_id_x','question_y','keywords_both','embedding_x','node_id_y','similarity']


class PreparedDense:
    """Reuse native-dtype answer vectors; allocate only one pending score block."""

    def __init__(self, answerable):
        import torch

        self.torch = torch
        self.answerable = np.array(answerable.embedding.tolist())
        self.dtype = self.answerable.dtype
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.answerable = torch.tensor(self.answerable).cuda()

    def __call__(self, pending, _answerable=None):
        pending_array = np.array(pending.embedding.tolist())
        if self.cuda:
            return self.torch.tensor(pending_array).cuda().mm(self.answerable.T).cpu().numpy()
        return pending_array.dot(self.answerable.T)


def exact_edges(node2questiondict, docid2nodes, dense, sparse, chunk_size=8, *, reuse_answer_vectors=False):
    rows=[]
    for (nid,did),groups in node2questiondict.items():
        for label,questions in groups.items():
            for qi,(question,keywords,embedding) in enumerate(questions):
                rows.append({'doc_id': did,'node_id': nid,'question_label': label,'question_id': qi,
                                 'embedding': embedding,'question': question,'keywords': keywords})
    df=pd.DataFrame(rows,columns=['doc_id','node_id','question_label','question_id','embedding','question','keywords'])
    p=df[df.question_label=='pending'].reset_index(drop=True)
    a=df[df.question_label=='answerable'].reset_index(drop=True)
    if p.empty or a.empty:
        return pd.DataFrame(columns=COLS), a
    if reuse_answer_vectors:
        dense = PreparedDense(a)
        logging.getLogger(__name__).info("HopRAG prepared answer vectors once: shape=%s dtype=%s block=%d", tuple(dense.answerable.shape), dense.dtype, chunk_size)
    pn=p.node_id.to_numpy();an=a.node_id.to_numpy();pdid=p.doc_id.to_numpy();adid=a.doc_id.to_numpy()
    pdocs=sorted(set(pdid));adocs=sorted(set(adid))
    pr={d:i for i,d in enumerate(pdocs)};ar={d:i for i,d in enumerate(adocs)}
    ai_groups={d:np.flatnonzero(adid==d) for d in adocs}
    weights={}
    # Native groupby(doc_id_x,doc_id_y).apply uses the first surviving pair's
    # keyword sets for the entire document pair, including its unusual behavior.
    for dp in pdocs:
        pis=np.flatnonzero(pdid==dp)
        for da,ais in ai_groups.items():
            first=None
            for pi in pis:
                valid=ais[an[ais]!=pn[pi]]
                if len(valid):first=(int(pi),int(valid[0]));break
            if first:
                pi,aj=first;weights[dp,da]=sparse(p.iloc[pi].keywords,a.iloc[aj].keywords)
    weight_matrix=np.array([[weights.get((dp,da),0.0) for da in adocs] for dp in pdocs])
    pcodes=np.array([pr[d] for d in pdid]);acodes=np.array([ar[d] for d in adid])
    best={};cross={}
    cross_used=set(a.loc[[any(dp!=da for dp in pdocs) for da in adid],'question'])
    def candidate(pi,aj,score):
        pp=p.iloc[pi];aa=a.iloc[aj]
        return {'node_id_x':pp.node_id,'node_id_y':aa.node_id,'question_x':pp.question,
            'question_y':aa.question,'embedding_x':pp.embedding,
            'keywords_both':pp.keywords.union(aa.keywords),'similarity':float(score),
            '_order':(pr[pp.doc_id],ar[aa.doc_id],pi,aj)}
    def key(row):return (-row['similarity'],row['_order'])
    for start in range(0,len(p),chunk_size):
        if start % 512 == 0:
            logging.getLogger(__name__).info('HopRAG exhaustive edge scoring: %d/%d pending, %d answerable',start,len(p),len(a))
        block=p.iloc[start:start+chunk_size]
        scores=np.asarray(dense(block,a)).reshape(len(block),len(a))
        # Adding native Python sparse floats promotes the native pandas sum.
        scores=scores.astype(np.float64)
        scores+=weight_matrix[pcodes[start:start+len(block)]][:,acodes]
        for offset in range(len(block)):
            pi=start+offset
            scores[offset,an==pn[pi]]=-np.inf
            vals=scores[offset];valid=np.flatnonzero(np.isfinite(vals))
            if not len(valid):continue
            peak=vals[valid].max();ties=valid[vals[valid]==peak]
            aj=min(ties,key=lambda j:(ar[adid[j]],int(j)))
            row=candidate(pi,int(aj),peak);q=row['question_x']
            if q not in best or key(row)<key(best[q]):best[q]=row
            valid=valid[adid[valid]!=pdid[pi]]
            if len(valid):
                threshold=np.partition(vals[valid],-min(2,len(valid)))[-min(2,len(valid))]
                pool=valid[vals[valid]>=threshold]
                chosen=sorted(pool,key=lambda j:(-vals[j],ar[adid[j]],int(j)))[:2]
                candidates=cross.get(q,[])+[candidate(pi,int(j),vals[j]) for j in chosen]
                cross[q]=sorted(candidates,key=key)[:2]
    one=pd.DataFrame([best[q] for q in sorted(best)])
    two=pd.DataFrame([r for q in sorted(cross) for r in cross[q]])
    if one.empty:return pd.DataFrame(columns=COLS),a
    abstract=a[~a.question.isin(set(one.question_y)|cross_used)]
    # Retain the native downstream sort/dedup and final concatenation order.
    one=one.sort_values('similarity',ascending=False).drop_duplicates(['node_id_x','node_id_y'])
    if not two.empty:
        two=two.sort_values('similarity',ascending=False).drop_duplicates(['node_id_x','node_id_y'])
        two=pd.concat([two.iloc[:1000000000],two.iloc[1000000000:]],ignore_index=True)
        two=two.sort_values('similarity',ascending=False).drop_duplicates(['node_id_x','node_id_y'])
        result=pd.concat([one.iloc[:250000000][COLS],two.iloc[:750000000][COLS]],ignore_index=True)
    else:result=one.iloc[:250000000][COLS]
    return result.drop_duplicates(['node_id_x','node_id_y']),abstract
