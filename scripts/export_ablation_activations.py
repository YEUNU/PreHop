"""Export recorded starting passages for paired connection-timing replay."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_ablation_links import read_events


def export(result, events):
    if result.get("status") != "completed_unadmitted" or result.get("evaluation_scope") != "full_benchmark":
        raise ValueError("Use a completed full-query reference ablation")
    activations, failures = {}, []
    for row in result["details"]:
        qid = row["query_id"]
        if qid in activations or qid in failures:
            raise ValueError("Duplicate result query ID")
        if row.get("error"):
            failures.append(qid)
            events.pop(qid, None)
            continue
        payload = events.pop(qid)
        starts = payload["starts"]
        if not isinstance(starts, list) or any(not isinstance(s, str) or not s for s in starts):
            raise ValueError("Missing or invalid recorded starting passages")
        activations[qid] = sorted(set(starts))
    if events:
        raise ValueError("Trace contains queries outside reference result")
    return activations, failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--events", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    activations, failures = export(json.loads(args.result.read_text()), read_events(args.events, "ablation_link_usefulness"))
    with args.output.open("x") as f:
        json.dump(activations, f, sort_keys=True)
    print(json.dumps({"successful_queries": len(activations), "failed_query_ids": failures,
                      "reference_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
                      "activations_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
