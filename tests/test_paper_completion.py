import json
import sys

import pytest

from scripts import record_paper_completion as recorder


@pytest.mark.parametrize('status,code', [('completed_unadmitted',0),('in_progress',1),('failed',1)])
def test_completion_does_not_apply_seed_policy(tmp_path, monkeypatch, status, code):
    monkeypatch.setattr(recorder,'ROOT',tmp_path)
    base=tmp_path/'data/results/example'
    result=base/'gfm_rag/multihoprag/seed_42/gfm_rag_multihoprag.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'status':status,'index_provenance':{'policy':{'generation_seed':42}}}))
    original=result.read_bytes()
    output=base/'admission.json'
    monkeypatch.setattr(sys,'argv',['record','example','multihoprag','gfm_rag','--exact-run-id','--output',str(output)])
    assert recorder.main()==code
    assert result.read_bytes()==original
    assert output.exists()==(code==0)
    if code==0:
        assert json.loads(output.read_text())['verification']=='disabled_by_user'
        recorder.main()
