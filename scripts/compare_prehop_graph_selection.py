"""Compare recorded selection prompts with stored-link evidence on frozen candidates.

Run with ``python -m scripts.compare_prehop_graph_selection --reference RESULT``.
The reference and its traces come from your own Prehop
run. The command uses the existing inference client and does not query a graph,
generate answers or change defaults.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import statistics
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_inputs(reference):
    """Recover ordered pools and actual prompts, without evaluation annotations."""
    from models.prehop.graphrag import GraphRAG

    rows = {r["query_id"]: r for r in reference["details"]}
    traces = {Path((r.get("prehop_trace") or r["historical_pipeline_trace"])["events_path"]) for r in rows.values()}
    starts, requests = {}, {}
    fields = {"id", "title", "doc", "source", "page", "sent_id", "publisher",
              "published_at", "author", "category", "text", "retrieval_paths"}
    for trace in sorted(traces):
        with trace.open() as stream:
            for event in map(json.loads, stream):
                qid = event.get("identity", {}).get("query_id")
                kind = event.get("event")
                if qid not in rows or kind not in {"SimilarityScoringMixin._role_body_list_ranking.start", "http.request"}:
                    continue
                raw = gzip.decompress((trace.parent / event["payload"]).read_bytes())
                data = json.loads(raw)
                if kind == "http.request":
                    request = json.loads(data["body"])
                    if (request.get("response_format") or {}).get("json_schema", {}).get("name") == "prehop_ranking_v1":
                        requests.setdefault(qid, request)
                else:
                    # Selection needs passage text and stored paths, not cached vectors.
                    starts[qid] = {"query_text": data["query_text"], "top_k": data["top_k"],
                                   "ordered": [{k: v for k, v in node.items() if k in fields}
                                               for node in data["ordered"]]}
    policy = reference.get("ablation") or {}
    order = policy.get("candidate_order_input_order", "search")
    seed = policy.get("candidate_order_shuffle_seed", 0)
    inputs = []
    for qid, row in rows.items():
        start = starts[qid]
        canonical, pool = GraphRAG._prepare_ranking_candidates(start["query_text"], start["ordered"], order, seed)
        inputs.append({"query_id": qid, "original_query_id": row.get("original_query_id") or qid,
                       "nodes": pool, "canonical_ids": [GraphRAG._node_identity(n) for n in canonical],
                       "query": start["query_text"], "top_k": start["top_k"],
                       "request": requests[qid]})
    return inputs


def apply_ranking(payload, candidates, canonical_ids, top_k):
    """Use the runtime's direct ID lookup and short-ranking fill semantics."""
    from models.prehop.graphrag import GraphRAG

    selected = [candidates[candidate_id] for candidate_id in payload["ranking"]]
    by_node_id = {GraphRAG._node_identity(n): n for n in candidates.values()}
    GraphRAG._complete_ranking(selected, [by_node_id[node_id] for node_id in canonical_ids], top_k)
    return GraphRAG._build_unique_sources(selected)


async def generate(inputs, output):
    from core import inference_telemetry
    from core.execution_profile import resolved_execution_environment
    from core.generation_profiles import request_settings
    from core.structured_outputs import ranking_contract
    from core.vllm_client import VLLMClient
    from utils.prompts.graph_evidence import add_graph_evidence

    output.mkdir(parents=True, exist_ok=True)
    target = output / "selections.jsonl"
    done = {}
    client = VLLMClient()
    semaphore = asyncio.Semaphore(int(resolved_execution_environment()["RAG_BENCHMARK_CONCURRENCY"]))
    jobs = [(r, condition) for r in inputs for condition in ("current", "graph_aware")]
    random.Random(42).shuffle(jobs)
    count = 0
    with target.open("a", buffering=1) as stream:
        async def one(row, condition):
            nonlocal count
            candidates = {f"C{i:03d}": node for i, node in enumerate(row["nodes"])}
            prompt = row["request"]["messages"][0]["content"]
            if condition == "graph_aware":
                prompt = add_graph_evidence(prompt, candidates)
            record = {k: row[k] for k in ("query_id", "original_query_id")}
            record.update(condition=condition, model=client.model_name,
                          messages=[{"role": "user", "content": prompt}],
                          settings=request_settings("ranking"))
            async with semaphore:
                token = inference_telemetry.begin()
                started = time.perf_counter()
                try:
                    payload = await client.generate_json(record["messages"],
                        structured_contract=ranking_contract(list(candidates), row["top_k"]),
                        **record["settings"])
                    record["response"] = payload
                    record.update(status="completed",
                                  retrieved_sources=apply_ranking(payload, candidates, row["canonical_ids"], row["top_k"]))
                except Exception as exc:  # noqa: BLE001 -- Keep failed queries in the comparison population.
                    record.update(status="failed", error_type=type(exc).__name__, retrieved_sources=[])
                finally:
                    record["selection_seconds"] = time.perf_counter() - started
                    record["usage"] = inference_telemetry.finish(token)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[row["query_id"], condition] = record
                count += 1
                if count % 10 == 0 or count == len(jobs):
                    print(f"Selection calls completed: {count}/{len(jobs)}", flush=True)
        try:
            await asyncio.gather(*(one(row, condition) for row, condition in jobs))
        finally:
            await client.global_close()
    return done


def evaluate(reference, calls, output):
    from scripts.ablation_statistics import cluster_interval
    from scripts.evaluate_saved_retrieval import METRICS, hotpot_metrics, multihop_metrics
    from utils.hotpotqa import project_sentences, score
    from utils.official_results import UPSTREAM, dataset_key

    dataset = dataset_key(reference["dataset"])
    store = ROOT / "data/hotpotqa_corpus/sentences.sqlite3"

    @lru_cache(maxsize=100000)
    def project(serialized):
        return project_sentences([json.loads(serialized)], store)

    conditions = {}
    for condition in ("current", "graph_aware"):
        details = []
        for source in reference["details"]:
            rec = calls[source["query_id"], condition]
            passages = rec["retrieved_sources"]
            if dataset == "multihoprag":
                metrics = multihop_metrics(passages, source["expected_sources"]["facts"])
                support = {}
            else:
                gold = source["expected_sources"]["supporting_facts"]
                metrics = hotpot_metrics(passages, gold, lambda s: project(json.dumps(s, sort_keys=True)))
                predicted = project_sentences(passages, store)
                support = {k: v for k, v in score("", predicted, "", gold).items() if k.startswith("sp_")}
            details.append({"query_id": source["query_id"], "original_query_id": rec["original_query_id"],
                            "status": rec["status"], "retrieved_sources": passages,
                            "ranking": metrics, "support": support,
                            "selection_seconds": rec["selection_seconds"], "usage": rec["usage"]})
        conditions[condition] = details
        eligible = [r for r in details if r["ranking"] is not None]
        official_keys = METRICS[:4] if dataset == "multihoprag" else ("sp_em", "sp_prec", "sp_recall", "sp_f1")
        field = "ranking" if dataset == "multihoprag" else "support"
        official = {"dataset": dataset, "condition": condition, "implementation": UPSTREAM[dataset],
                    "scope": "Selection only; answer and joint outcomes unmeasured", "answer": None,
                    "rows": len(details), "metric_rows": len(eligible),
                    "failed_rows": sum(r["status"] != "completed" for r in details),
                    "metrics": {k: statistics.mean(r[field][k] for r in eligible) if eligible else None
                                for k in official_keys},
                    "details": [{"query_id": r["query_id"], "metrics": {k: r[field][k] for k in official_keys}} for r in eligible]}
        dump(output / f"{condition}.json", {"dataset": dataset, "details": details})
        dump(output / f"{condition}.official.json", official)
        dump(output / f"{condition}.diagnostics.json", {
            "ranking_definition": "MultiHop-RAG ranking; adapted sentence ranking for HotpotQA",
            "metrics": {k: statistics.mean(r["ranking"][k] for r in eligible) if eligible else None
                        for k in METRICS}})
    comparison = {"dataset": dataset, "scope": "Fixed candidates; graph_aware minus current selection",
                  "latency_scope": "Selection calls only; no full-query or answer timing", "ranking": {}, "support": {}}
    for field, metrics in (("ranking", METRICS), ("support", ("sp_em", "sp_prec", "sp_recall", "sp_f1"))):
        for metric in metrics:
            pairs = [(a, b) for a, b in zip(conditions["graph_aware"], conditions["current"], strict=True)
                     if a[field] is not None and metric in a[field]]
            if pairs:
                comparison[field][metric] = cluster_interval([a[field][metric] - b[field][metric] for a, b in pairs],
                                                             [a["original_query_id"] for a, b in pairs])
    dump(output / "comparison.json", comparison)
    print(json.dumps({"dataset": dataset, "comparison": str(output / "comparison.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Your completed Prehop result, with its recorded trace files")
    args = parser.parse_args()
    from scripts.runner_environment import _load_runner_environment
    _load_runner_environment()
    os.environ["RAG_PAPER_MODE"] = "true"
    reference = json.loads(args.reference.read_text())
    inputs = read_inputs(reference)
    output = ROOT / "data/results" / f"graph-selection-{time.time_ns()}"
    dump(output / "reference.json", {"path": str(args.reference.resolve()), "sha256": sha(args.reference),
                                         "rows": len(inputs), "dataset": reference["dataset"]})
    calls = asyncio.run(generate(inputs, output))
    evaluate(reference, calls, output)


if __name__ == "__main__":
    main()
