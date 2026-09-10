"""Export complete HotpotQA outputs in the official answer/sp JSON format."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hotpotqa import METRICS, score
from utils.metrics import extract_final_answer


def export(result, gold):
    if result.get("status") != "completed_unadmitted" or result.get("evaluation_scope") != "full_benchmark":
        raise ValueError("Export requires a completed full benchmark")
    if str(result.get("dataset", "")).lower() != "hotpotqa":
        raise ValueError("Not a HotpotQA result")
    rows = result["details"]
    by_id = {row["query_id"]:row for row in rows}
    gold_by_id = {row["_id"]:row for row in gold}
    if len(by_id) != len(rows) or set(by_id) != set(gold_by_id):
        raise ValueError("Prediction and gold query IDs differ")
    output = {"answer":{},"sp":{}}
    totals = dict.fromkeys(METRICS,0.0)
    for qid, row in by_id.items():
        answer = "" if row.get("error") else extract_final_answer(row["answer"])
        facts = [] if row.get("error") else row["predicted_supporting_facts"]
        output["answer"][qid],output["sp"][qid] = answer,facts
        metrics = score(answer,facts,gold_by_id[qid]["answer"],gold_by_id[qid]["supporting_facts"])
        for key,value in metrics.items():
            totals[key] += value
    return output,{key:value/len(rows) for key,value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result",type=Path)
    parser.add_argument("--gold",type=Path,default=Path("data/hotpotqa_raw/hotpot_dev_fullwiki_v1.json"))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    predictions,metrics = export(json.loads(args.result.read_text()),json.loads(args.gold.read_text()))
    with args.output.open("x") as f:
        json.dump(predictions,f,ensure_ascii=False)
    print(json.dumps({"official_rule_metrics":metrics,"result_sha256":hashlib.sha256(args.result.read_bytes()).hexdigest(),
                      "gold_sha256":hashlib.sha256(args.gold.read_bytes()).hexdigest()},indent=2))


if __name__ == "__main__":
    main()
