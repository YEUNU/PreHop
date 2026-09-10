import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from models.prehop.connection_timing import CONTRACT, read_pairs, resolve_pairs
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


def make_store(path):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata (key TEXT,value TEXT)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", [("contract",CONTRACT),("namespace","test"),("status","complete")])
        db.execute("CREATE TABLE starts (id TEXT,pairs TEXT,matches INTEGER)")
        db.executemany("INSERT INTO starts VALUES (?,?,?)",[("a",json.dumps([{"source_id":"a","id":"z","activated_question_ids":["q"]}]),1),("b","[]",0)])


@pytest.mark.asyncio
async def test_matched_hydration_and_alternating_replay(tmp_path,monkeypatch):
    from models.prehop import connection_timing as timing
    path = tmp_path / "links.sqlite3"
    make_store(path)
    engine = SimpleNamespace()
    async def resolve(_engine,starts):
        return read_pairs(path,starts,"test")
    monkeypatch.setattr(timing,"resolve_pairs",resolve)
    hydrate = AsyncMock(side_effect=lambda _e,pairs,_excluded:pairs)
    monkeypatch.setattr(timing,"hydrate",hydrate)
    result = await replay(engine,{"query":["a","b"]},path,"test",repetitions=2,warmups=0)
    assert result["identical_start_fraction"] == 1.0
    assert result["details"][0]["order"] == ("precomputed","online")
    assert result["details"][1]["order"] == ("online","precomputed")
    assert hydrate.await_count == 4
    with pytest.raises(ValueError,match="lacks"):
        read_pairs(path,["unknown"],"test")
    with pytest.raises(ValueError,match="namespace"):
        read_pairs(path,["a"],"other")


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


def test_activation_export_keeps_zero_starts_and_records_failures():
    from scripts.export_ablation_activations import export
    result = {"status":"completed_unadmitted", "evaluation_scope":"full_benchmark",
              "details":[{"query_id":"zero"},{"query_id":"failed","error":"timeout"}]}
    activations, failures = export(result, {"zero":{"starts":[]}})
    assert activations == {"zero":[]} and failures == ["failed"]
    with pytest.raises(KeyError):
        export(result, {})


def test_annotation_agreement_includes_unclear_and_does_not_invent_adjudication():
    from scripts.analyze_ablation_annotations import compare
    first = {"a":{"supplies_source_question":"yes","meaningful_transition":"yes"},
             "b":{"supplies_source_question":"unclear","meaningful_transition":"no"}}
    second = {"a":first["a"],"b":{"supplies_source_question":"no","meaningful_transition":"no"}}
    report = compare(first,second,first)
    assert report["fields"]["supplies_source_question"]["agreement"] == .5
    assert report["fields"]["meaningful_transition"]["agreement"] == 1
    assert report["fields"]["supplies_source_question"]["adjudicated"]["unclear"] == 1


@pytest.mark.asyncio
async def test_edge_sample_is_unique_seeded_and_keeps_every_question_pair():
    from scripts.export_ablation_edge_sample import sample_edges
    population = [{"src":f"s{i:04}","dst":"t"} for i in range(130)]
    pairs = [{"question":"first", "answerable_question":"answer one"},
             {"question":"second", "answerable_question":"answer two"}]
    def service():
        return SimpleNamespace(execute_query=AsyncMock(side_effect=[population, []] +
            [[{"source_text":"source", "destination_text":"target", "matched_question_pairs":pairs}]
             for _ in range(100)]))
    a = await sample_edges(service(), "test")
    b = await sample_edges(service(), "test")
    assert a == b and a["population_edges"] == 130 and a["sample_size"] == 100
    assert len({r["sample_id"] for r in a["items"]}) == 100
    assert all(r["matched_question_pairs"] == pairs for r in a["items"])
