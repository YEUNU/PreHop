"""Compute query-conditioned HOP utility from explicit ablation trace events."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def usefulness(payload, gold, dataset):
    from utils.metrics import _official_multihoprag_fact_match
    direct = {str(n["id"]): n for n in payload["direct"]}
    hop = {str(n["id"]): n for n in payload["expanded"] if n.get("path_type") == "hop"}
    nxt = {str(n["id"]) for n in payload["expanded"] if n.get("path_type") == "next"}
    selected = {str(n["id"]) for n in payload["selected"]}
    def matched(nodes):
        if dataset == "multihoprag":
            return {i for i, fact in enumerate(gold) if any(_official_multihoprag_fact_match(fact, str(n.get("text") or "")) for n in nodes)}
        if dataset == "hotpotqa":
            from utils.hotpotqa import project_sentences
            return set(map(tuple,gold)) & set(map(tuple,project_sentences(list(nodes), "data/hotpotqa_corpus/sentences.sqlite3")))
        raise ValueError("Use the dataset's declared evidence matcher; unknown dataset")
    d = matched(direct.values())
    added = {k:n for k,n in hop.items() if k not in direct}
    retained = {k:n for k,n in added.items() if k in selected}
    denom = len(set(map(tuple,gold))) if dataset == "hotpotqa" else len(gold)
    return {"hop_destination_relevance": sum(bool(matched([n])) for n in hop.values())/len(hop) if hop and denom else None,
            "added_gold_coverage": len(matched(added.values())-d)/denom if denom else None,
            "retained_added_coverage": len(matched(retained.values())-d)/denom if denom else None,
            "retained_next_overlap_coverage": len(matched([n for k,n in retained.items() if k in nxt])-d)/denom if denom else None,
            "hop_destinations": len(hop), "hop_next_overlap": len(set(hop)&nxt),
            "evidence_bearing": bool(denom)}


def read_events(paths, event):
    values = {}
    for path in paths:
        path = Path(path)
        for line in path.open():
            row = json.loads(line)
            if row["event"] != event:
                continue
            qid = row.get("identity", {}).get("query_id")
            if not qid or qid in values:
                raise ValueError("Expected exactly one identified ablation traversal per query")
            blob = path.parent / row["payload"]
            raw = gzip.decompress(blob.read_bytes())
            if hashlib.sha256(raw).hexdigest() != row["payload_sha256"]:
                raise ValueError("Trace payload hash mismatch")
            values[qid] = json.loads(raw)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if result.get("status") != "completed_unadmitted" or result.get("evaluation_scope") != "full_benchmark":
        raise ValueError("Use a completed full-query ablation result")
    events = read_events(args.events, "ablation_link_usefulness")
    rows = []
    for row in result["details"]:
        if row.get("error"):
            events.pop(row["query_id"], None)
            rows.append({"query_id":row["query_id"],"failed":True})
            continue
        payload = events.pop(row["query_id"])
        dataset = str(result["dataset"]).lower().replace("multihop-rag", "multihoprag")
        expected = row["expected_sources"]
        if dataset not in {"multihoprag", "hotpotqa"}:
            raise ValueError("Unsupported dataset")
        gold = expected["facts"] if dataset == "multihoprag" else expected["supporting_facts"]
        rows.append({"query_id":row["query_id"],**usefulness(payload,gold,dataset)})
    if events:
        raise ValueError("Trace includes queries outside the result")
    keys = ["hop_destination_relevance","added_gold_coverage","retained_added_coverage","retained_next_overlap_coverage","hop_destinations","hop_next_overlap"]
    aggregates = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        aggregates[key] = {"mean":sum(values)/len(values) if values else None,"eligible_queries":len(values)}
    with args.output.open("x") as f:
        json.dump({"source_result":str(args.result.resolve()),"source_sha256":hashlib.sha256(args.result.read_bytes()).hexdigest(),
                   "queries":len(rows),"failed_queries":sum(bool(r.get("failed")) for r in rows),
                   "denominator_policy":"Utility is conditional on successful queries; failures remain counted separately. Missing events for successful queries are errors.",
                   "aggregates":aggregates,"details":rows},f,indent=2)


if __name__ == "__main__":
    main()
