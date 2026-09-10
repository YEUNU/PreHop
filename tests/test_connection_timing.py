import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from models.prehop.connection_timing import read_pairs, resolve_pairs
from scripts.analyze_ablation_links import usefulness
from scripts.prehop_connection_timing import replay


@pytest.mark.asyncio
async def test_online_reads_questions_not_historical_edges_and_counts_zero_starts():
    engine = SimpleNamespace(chunk_label="Chunk",q_plus_label="QPlus",q_minus_label="QMinus")
    engine.retry_query = AsyncMock(side_effect=[
        [{"id":"a","source":"doc","questions":[{"id":"q","query_embedding":[1,0]}]},
         {"id":"b","source":"doc","questions":[]}],
        [{"source":"doc","count":4}]])
    engine._process_hop_wave = AsyncMock(return_value=[{"src_id":"a","tgt_id":"z","source_question_ids":["q"]}])
    pairs, count = await resolve_pairs(engine,["b","a","a"])
    assert pairs == [{"source_id":"a","id":"z","activated_question_ids":["q"]}]
    assert count == 1
    assert engine._process_hop_wave.await_args.args[0][0]["ann_pools"] == {"q_minus":5}
    assert all("HOP_ANSWER" not in call.args[0] for call in engine.retry_query.await_args_list)


@pytest.mark.asyncio
async def test_matched_hydration_and_alternating_replay(tmp_path,monkeypatch):
    from models.prehop import connection_timing as timing
    path = tmp_path / "links.json"
    path.write_text(json.dumps({"experiment":"test","question_counts":{"a":1,"b":0}}))
    pairs=[{"source_id":"a","id":"z","activated_question_ids":["q"]}]
    monkeypatch.setattr(timing,"resolve_pairs",AsyncMock(return_value=(pairs,1)))
    monkeypatch.setattr(timing,"read_pairs",AsyncMock(return_value=pairs))
    hydrate = AsyncMock(side_effect=lambda _e,pairs,_excluded:pairs)
    monkeypatch.setattr(timing,"hydrate",hydrate)
    result = await replay(SimpleNamespace(),{"query":["a","b"]},path,"test",repetitions=2,warmups=0)
    assert result["identical_start_fraction"] == 1.0
    assert result["details"][0]["order"] == ("precomputed","online")
    assert result["details"][1]["order"] == ("online","precomputed")
    assert hydrate.await_count == 4


@pytest.mark.asyncio
async def test_stored_arm_reads_experiment_scoped_neo4j_edges():
    engine=SimpleNamespace(chunk_label='Chunk',retry_query=AsyncMock(return_value=[{'source_id':'a','pairs':[{'source_id':'a','id':'b'}]}]))
    pairs=await read_pairs(engine,['a'],'experiment1')
    assert pairs==[{'source_id':'a','id':'b'}]
    query,parameters=engine.retry_query.await_args.args
    assert 'HOP_TIMING' in query and parameters['experiment']=='experiment1'
    assert 'HOP_ANSWER' not in query


def test_link_utility_separates_direct_and_next_and_zero_gain():
    payload = {"direct":[{"id":"d","text":"fact one"}],
               "expanded":[{"id":"d","text":"fact one","path_type":"hop"},
                           {"id":"h","text":"fact two","path_type":"hop"},
                           {"id":"h","text":"fact two","path_type":"next"},
                           {"id":"n","text":"fact three","path_type":"next"}],
               "selected":[{"id":"d"},{"id":"h"}]}
    result = usefulness(payload,["fact one","fact two","fact three"],"multihoprag")
    assert result["hop_destinations"] == 2
    assert result["hop_destination_relevance"] == 1
    assert result["added_gold_coverage"] == pytest.approx(1/3)
    assert result["retained_added_coverage"] == pytest.approx(1/3)
    assert result["retained_next_overlap_coverage"] == pytest.approx(1/3)
    empty = usefulness({"direct":[],"expanded":[],"selected":[]},["fact"],"multihoprag")
    assert empty["hop_destination_relevance"] is None
    assert empty["hop_destinations"] == 0 and empty["added_gold_coverage"] == 0
