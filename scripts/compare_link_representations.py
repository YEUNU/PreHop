"""Prepare and replay a count-matched link-representation pilot without reindexing.

Preparation reads an explicit archived snapshot and the original Neo4j namespace.
Only Q+ -> body ANN lookups are new. No database writes or model requests occur
during preparation. Selection and evaluation are separate, resumable commands.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from scripts.analyze_link_supply import additions, coupled_sample, digest, rewire_edges, source
from scripts.campaign_runtime import atomic_json, close_owned, drain, lock

ARMS = ("question", "body", "qplus_body", "shuffled")
BASE_ARM = "base_only"
NODE_FIELDS = ("id", "source", "title", "text", "page", "sent_id", "publisher", "published_at", "author", "category")


def read(path):
    return json.loads(Path(path).read_text())


def comparison_arms(protocol):
    arms = tuple(protocol.get("arms", ARMS))
    if arms not in (ARMS, (*ARMS, BASE_ARM)):
        raise ValueError("Expected the four expansion arms, optionally followed by the common-base control")
    return arms


def adjacency(edges, *, undirected=False):
    result = defaultdict(set)
    for a, b in edges:
        result[a].add(b)
        if undirected:
            result[b].add(a)
    return result


def match_degrees(graphs, nodes, seed):
    """Match unique outgoing degree before seeing any query or gold labels."""
    arms = ARMS[:-1]
    adj = {arm: adjacency(graphs[arm]) for arm in arms}
    matched = {arm: [] for arm in arms}
    sources = {n["id"]: n["source"] for n in nodes}
    for node_id in sorted(sources):
        pools = [adj[arm][node_id] for arm in arms]
        for pool in pools:
            if any(t not in sources or sources[t] == sources[node_id] for t in pool):
                raise ValueError("Links must have known, different-source endpoints")
        selected = coupled_sample(pools, min(map(len, pools)), f"degree:{seed}:{node_id}")
        for arm, targets in zip(arms, selected, strict=True):
            matched[arm].extend((node_id, t) for t in targets)
    matched["shuffled"], audit = rewire_edges(matched["question"], sources, seed)
    return matched, audit


def candidates_for_query(direct, nodes, next_edges, hop_edges, max_added, seed):
    """Expand only saved direct starts; match new unique passages across arms."""
    starts = {n["id"]: n for n in direct}
    if len(starts) != len(direct):
        raise ValueError("Duplicate direct passage")
    base = set(starts) | set().union(*(next_edges.get(s, set()) for s in starts))
    pools = [additions(starts, base, hop_edges[arm]) for arm in ARMS]
    budget = min(max_added, *(len(p) for p in pools))
    selected = coupled_sample(pools, budget, seed)
    outputs = {}
    for arm, extra in zip(ARMS, selected, strict=True):
        kept = base | set(extra)
        scored = {n: {**nodes[n], "representation_score": 0.0} for n in kept}
        for n, direct_node in starts.items():
            scored[n]["representation_score"] = float(direct_node["representation_score"])
        for start, direct_node in starts.items():
            targets = next_edges.get(start, set()) | hop_edges[arm].get(start, set())
            for target in (targets & kept) - set(starts):
                scored[target]["representation_score"] = max(
                    scored[target]["representation_score"], .5 * float(direct_node["representation_score"]),
                )
        outputs[arm] = [scored[n] for n in sorted(scored)]
    return outputs, {"starts": sorted(starts), "base": sorted(base), "added_budget": budget,
                     "available_additions": dict(zip(ARMS, map(len, pools), strict=True))}


async def read_index(snapshot):
    """Use the production ANN lookup, but never merge its proposed edges."""
    from core.neo4j_service import Neo4jService
    from models.prehop.indexing.hop_edges import HopEdgeMixin

    namespace = snapshot["namespaces"]["question"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", namespace):
        raise ValueError("Invalid archived namespace")
    label = f"PR_{namespace}_Chunk"
    db = Neo4jService()

    class ReadOnlyLinks(HopEdgeMixin):
        body_vector_index = f"prehop_{namespace}_vector_idx"

        async def retry_query(self, query, params):
            return await db.execute_query(query, params)

    try:
        nodes = {}
        ids = sorted(n["id"] for n in snapshot["nodes"])
        for offset in range(0, len(ids), 128):
            rows = await db.execute_query(
                f"MATCH (n:{label}) WHERE n.id IN $ids RETURN properties(n) AS node",
                {"ids": ids[offset:offset + 128]},
            )
            for row in rows:
                n = row["node"]
                nodes[n["id"]] = {k: n[k] for k in (*NODE_FIELDS, "embedding") if k in n}
        if set(nodes) != set(ids):
            raise ValueError("Original index no longer covers the archived corpus")
        for old in snapshot["nodes"]:
            if any(nodes[old["id"]].get(k) != old[k] for k in ("source", "title", "text")):
                raise ValueError("Original indexed passage changed")
        counts = Counter(n["source"] for n in nodes.values())
        proposals = []
        engine = ReadOnlyLinks()
        for offset in range(0, len(ids), 64):
            wave = await db.execute_query(
                f"MATCH (n:{label}) WHERE n.id IN $ids "
                "OPTIONAL MATCH (n)-[:HAS_Q_PLUS]->(q) "
                "RETURN n.id AS id, n.source AS source, "
                "collect(q { .id, .text, .query_embedding }) AS questions ORDER BY id",
                {"ids": ids[offset:offset + 64]},
            )
            for item in wave:
                item["ann_pools"] = {"body": counts[item["source"]] + 1}
                item["questions"].sort(key=lambda q: q["id"])
                if any(not q.get("query_embedding") for q in item["questions"]):
                    raise ValueError("Stored Q+ is missing its query embedding")
            proposals.extend(await engine._find_hop_candidates_batch(wave, "body"))
            if offset % 640 == 0:
                print(f"Read-only Q+ -> body matching: {min(offset + 64, len(ids))}/{len(ids)}", flush=True)
        return nodes, proposals
    finally:
        await close_owned(db.close, primary_error=sys.exc_info()[1])


def verify_archive(archive):
    protocol, prepared = read(archive / "protocol.json"), read(archive / "prepared.json")
    if digest(archive / "protocol.json") != prepared["protocol_sha256"]:
        raise ValueError("Archived protocol changed")
    for name, expected in prepared["files"].items():
        if digest(archive / name) != expected:
            raise ValueError(f"Archived input changed: {name}")
    for key in protocol["sources"]:
        source(protocol, key)
    return protocol


async def prepare(args):
    from core.config import RAGConfig
    from core.generation_profiles import request_settings
    from core.inference_transport import InferenceTransport
    from core.structured_outputs import ranking_contract
    from models.prehop.ablation_inputs import input_path
    from models.prehop.graphrag import GraphRAG
    from scripts.ablation_statistics import compare
    from utils.provenance import code_provenance

    if RAGConfig.FINAL_RANK_VARIANT != "fused":
        raise ValueError("Pilot requires the shared fused ranking configuration")
    archived = verify_archive(args.archive)
    snapshot = read(args.archive / "snapshot.json")
    queries_path = source(archived, "queries")
    queries = read(queries_path)
    by_id = {q["_id"]: q for q in queries}
    if len(by_id) != len(queries) or set(by_id) != {r["query_id"] for r in snapshot["rows"]}:
        raise ValueError("Query population mismatch")
    # Freeze the sample before constructing new links or inspecting query-level scores.
    ids = sorted(by_id)
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.limit]
    if not ids:
        raise ValueError("Empty pilot")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "experiment": "link-representation-pilot-v1", "created_at": time.time(), "seed": args.seed,
        "status": "exploratory; historical question/body results already observed",
        "arms": ARMS, "query_ids": ids, "queries": str(queries_path), "queries_sha256": digest(queries_path),
        "archive": str(args.archive.resolve()), "archive_protocol_sha256": digest(args.archive / "protocol.json"),
        "source_artifacts": archived["sources"], "code_provenance": code_provenance(),
        "hypothesis": "Does Q+ -> Q- improve selected evidence over passage ANN and Q+ -> body?",
        "primary_metric": "MAP@10", "primary_contrast": "question minus body, LLM selection",
        "secondary": ["question minus qplus_body", "question minus shuffled", "Recall@10", "complete evidence@10",
                      "pre-selector fact coverage", "graph by selector interaction", "tokens", "terminal failures"],
        "fixed": {"starts": "saved multichannel direct inputs", "base": "direct + bidirectional NEXT",
                  "hop_depth": 1, "path_decay": .5, "semantic_scoring": "body only, no bridge embeddings",
                  "candidate_order": "shared production fused scorer", "final_top_k": 12,
                  "max_added_passages": 12, "reader_calls": 0},
        "degree_control": "Per-source minimum unique outdegree across three semantic graphs; shared random priorities. "
                          "Shuffle the resulting question graph preserving its in/out degrees and source exclusion.",
        "candidate_control": "Per-query minimum additional unique passages across all four arms, capped at 12; "
                             "shared random priorities; retain zero-budget queries. Equal counts do not match tokens.",
        "uncertainty": "Paired original-question cluster bootstrap, 10000 resamples, seed 42; "
                       "conditional on one graph shuffle, sample, index and selector realization; not indexing variation.",
        "interpretation": "Count-matched controlled variants, not the unmodified main method. Pilot is for diagnosis; "
                          "retain all arms and null/failure IDs. A positive pilot is not confirmatory evidence.",
        "generation_model": InferenceTransport.resolve("core").generation_model,
        "ranking_settings": request_settings("ranking"), "ranking_schema": ranking_contract(["C000"], 12).provenance(),
        "selector_prompt_sha256": digest(Path("utils/prompts/evidence_ranking.py")),
        "script_sha256": digest(Path(__file__)), "planned_selector_requests": len(ids) * len(ARMS),
    }
    atomic_json(args.output / "protocol.json", protocol)
    # Preserve the inconvenient historical evidence alongside the follow-up's design.
    historical = compare(read(source(archived, "question_result")), read(source(archived, "body_result")),
                         ["official_map@10"], queries)
    atomic_json(args.output / "historical-audit.json", {
        "candidate_supply": read(args.archive / "comparison.json")["conditions"],
        "selected_evidence": historical,
        "caveat": "Historical selected results have unequal pool sizes and effective bridge scoring; "
                  "they do not isolate link representation. New pilot controls both.",
        "archived_comparison_sha256": digest(args.archive / "comparison.json"),
    })
    nodes, proposals = await read_index(snapshot)
    original = {"question": snapshot["question_edges"]["HOP_ANSWER"],
                "body": snapshot["body_edges"]["HOP_ANSWER"],
                "qplus_body": sorted({(p["source_chunk_id"], p["target_id"]) for p in proposals})}
    edges, shuffle_audit = match_degrees(original, snapshot["nodes"], args.seed)
    atomic_json(args.output / "graphs.json", {"original": original, "matched": edges,
                                              "qplus_body_proposals": proposals, "shuffle_audit": shuffle_audit})
    next_edges = adjacency(snapshot["question_edges"]["NEXT"], undirected=True)
    hop_edges = {arm: adjacency(edges[arm]) for arm in ARMS}
    reference_rows = {r["query_id"]: r for r in snapshot["rows"]}
    direct_dir = source(archived, "direct_manifest").parent
    engine = object.__new__(GraphRAG)  # Pure shared scorer; no client/index initialization.
    files, audits = {}, []
    for count, qid in enumerate(ids, 1):
        query = GraphRAG._normalize_entity_term(by_id[qid]["query"]) or by_id[qid]["query"].strip()
        input_file = input_path(direct_dir, query)
        from models.prehop.ablation_inputs import read_input
        frozen = read_input(direct_dir, query)
        if frozen["query"] != query:
            raise ValueError("Frozen query mismatch")
        for n in frozen["base_candidates"]:
            if n["embedding"] != nodes[n["id"]]["embedding"]:
                raise ValueError("Original direct and live body embeddings differ")
        pools, audit = candidates_for_query(frozen["base_candidates"], nodes, next_edges, hop_edges, 12,
                                           f"candidates:{args.seed}:{qid}")
        if (audit["starts"] != sorted(reference_rows[qid]["starts"])
                or audit["base"] != sorted(reference_rows[qid]["base"])):
            raise ValueError("Archived starts or common NEXT base differ")
        conditions = {}
        for arm, pool in pools.items():
            selected, ordered = await engine._score_and_select(frozen["query_embedding"], pool, 12, query,
                                                               selection_variant="global")
            conditions[arm] = {"ordered": [{k: n[k] for k in NODE_FIELDS if k in n} for n in ordered],
                               "fused_sources": GraphRAG._build_unique_sources(selected)}
        name = f"inputs/{qid}.json"
        atomic_json(args.output / name, {"query_id": qid, "query": query, "conditions": conditions})
        files[name] = digest(args.output / name)
        audits.append({"query_id": qid, "input_path": str(input_file), "input_sha256": digest(input_file), **audit})
        if count % 32 == 0:
            print(f"Frozen selector inputs: {count}/{len(ids)}", flush=True)
    atomic_json(args.output / "input-audit.json", {"details": audits})
    for name in ("protocol.json", "graphs.json", "input-audit.json", "historical-audit.json"):
        files[name] = digest(args.output / name)
    atomic_json(args.output / "prepared.json", {"files": files, "selector_requests": len(ids) * len(ARMS),
                                               "model_requests_made": 0, "database_writes": 0})


def verify_prepared(output):
    for name, expected in read(output / "prepared.json")["files"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Frozen pilot input changed: {name}")
    protocol = read(output / "protocol.json")
    if digest(Path(protocol["queries"])) != protocol["queries_sha256"]:
        raise ValueError("Query annotations changed")
    return protocol


async def add_base(prepared, output):
    """Copy frozen expansion arms and rescore D+NEXT from saved trace embeddings."""
    from core.config import RAGConfig
    from models.prehop.ablation_inputs import read_input
    from models.prehop.graphrag import GraphRAG
    from scripts.compare_prehop_selectors import payload
    from utils.provenance import code_provenance

    old = verify_prepared(prepared)
    if comparison_arms(old) != ARMS or RAGConfig.FINAL_RANK_VARIANT != "fused":
        raise ValueError("Extend a four-arm pilot with the shared fused scorer")
    archived = verify_archive(Path(old["archive"]))
    snapshot = read(Path(old["archive"]) / "snapshot.json")
    next_edges = adjacency(snapshot["question_edges"]["NEXT"], undirected=True)
    audit = read(prepared / "input-audit.json")
    audits = {row["query_id"]: row for row in audit["details"]}
    reference = read(source(archived, "question_result"))
    ids = set(old["query_ids"])
    event_paths = sorted({r["prehop_trace"]["events_path"] for r in reference["details"] if r["query_id"] in ids})
    events = {}
    for path in map(Path, event_paths):
        for line in path.open():
            event = json.loads(line)
            qid = event.get("identity", {}).get("query_id")
            if qid in ids and event["event"] == "SimilarityScoringMixin._score_and_select.start":
                if qid in events:
                    raise ValueError("Duplicate archived scoring input")
                events[qid] = (path.parent, event)
    if set(events) != ids:
        raise ValueError("Missing archived scoring input")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        **old, "experiment": "link-representation-pilot-v2", "created_at": time.time(),
        "arms": [*ARMS, BASE_ARM], "planned_selector_requests": len(ids) * (len(ARMS) + 1),
        "parent_prepared": str(prepared.resolve()), "parent_manifest_sha256": digest(prepared / "prepared.json"),
        "parent_primary_contrast": old["primary_contrast"],
        "parent_selector_commits": len(list((prepared / "calls").glob("*/*.json"))),
        "status": "Exploratory extension after observing candidate coverage and fused scores; preserve all parent arms.",
        "hypothesis": "Does structured HOP expansion improve selected evidence over shuffled additions and D+NEXT?",
        "primary_contrast": "question minus shuffled, LLM selection",
        "secondary": ["each structured arm minus base_only", "shuffled minus base_only", "all structured pairs",
                      "Recall@10", "complete evidence@10", "evidence supply/retention/completeness funnel",
                      "graph by selector interaction", "tokens", "terminal failures"],
        "population": "All sampled evidence-bearing queries for paired ranking comparisons; null IDs retained. "
                      "Supply-conditioned subsets are descriptive and differ by arm; never substitute them for the population.",
        "base_control": "Exact D+NEXT pool, rescored without HOP contributions using saved body/query embeddings. "
                        "Fewer candidates than expansion arms; this contrast does not match input counts or tokens.",
        "funnel": "Distinct facts absent from the full D+NEXT pool, supplied by added candidates, retained in top 10, "
                  "and accompanied by complete selected evidence. Also report all completeness gains/losses versus "
                  "the base-only selector. Zero supplied facts gives undefined conditional retention, not zero success.",
        "interpretation_rules": [
            "Positive contrasts support only the evaluated configurations; pointwise intervals are exploratory.",
            "An interval including zero is inconclusive, not evidence of equivalence or useless graphs.",
            "A negative interval supports harm for that contrast; retain and report it without changing the subset.",
            "LLM versus fused interaction does not identify joint comparison as the causal mechanism.",
            "Pilot outcomes do not automatically trigger seed/prompt/subset searches or selective full reruns.",
        ],
        "code_provenance": code_provenance(), "script_sha256": digest(Path(__file__)),
    }
    atomic_json(output / "protocol.json", protocol)
    for name in ("graphs.json", "historical-audit.json"):
        shutil.copy2(prepared / name, output / name)
    engine = object.__new__(GraphRAG)
    files = {}
    for number, qid in enumerate(old["query_ids"], 1):
        data = read(prepared / "inputs" / f"{qid}.json")
        row = audits[qid]
        direct_path = Path(row["input_path"])
        if digest(direct_path) != row["input_sha256"]:
            raise ValueError("Saved direct input changed")
        frozen = read_input(direct_path.parent, data["query"])
        directory, event = events[qid]
        scored = payload(directory, event)
        if scored["query_embedding"] != frozen["query_embedding"] or scored["query_text"] != data["query"]:
            raise ValueError("Archived query or embedding mismatch")
        recorded_nodes = {n["id"]: n for n in scored["candidates"]}
        visible_nodes = {n["id"]: n for n in data["conditions"]["question"]["ordered"]}
        nodes = {}
        for node_id in row["base"]:
            n = recorded_nodes[node_id]
            # Neo4j omits absent properties; archived query projections may
            # materialize those same optional properties as null. Preserve
            # the parent's exact rendering after checking semantic equality.
            visible = visible_nodes[node_id]
            if any(n.get(k) != visible.get(k) for k in NODE_FIELDS):
                raise ValueError(f"Archived baseline passage differs from frozen pilot: {qid}/{node_id}")
            nodes[node_id] = {**visible, "embedding": n["embedding"]}
        for n in frozen["base_candidates"]:
            if n["embedding"] != nodes[n["id"]]["embedding"]:
                raise ValueError("Archived direct embedding mismatch")
        pools, base_audit = candidates_for_query(frozen["base_candidates"], nodes, next_edges,
                                                {arm: {} for arm in ARMS}, 0, f"base:{qid}")
        if any(base_audit[k] != row[k] for k in ("starts", "base")):
            raise ValueError("Common baseline identities changed")
        selected, ordered = await engine._score_and_select(frozen["query_embedding"], pools["question"], 12,
                                                           data["query"], selection_variant="global")
        data["conditions"][BASE_ARM] = {"ordered": [{k: n[k] for k in NODE_FIELDS if k in n} for n in ordered],
                                       "fused_sources": GraphRAG._build_unique_sources(selected)}
        name = f"inputs/{qid}.json"
        atomic_json(output / name, data)
        files[name] = digest(output / name)
        row["parent_input_sha256"] = digest(prepared / name)
        row["base_trace_payload_sha256"] = event["payload_sha256"]
        if number % 32 == 0:
            print(f"Added frozen D+NEXT control: {number}/{len(ids)}", flush=True)
    atomic_json(output / "input-audit.json", audit)
    for name in ("protocol.json", "graphs.json", "input-audit.json", "historical-audit.json"):
        files[name] = digest(output / name)
    atomic_json(output / "prepared.json", {"files": files, "selector_requests": protocol["planned_selector_requests"],
                                          "model_requests_made": 0, "database_writes": 0,
                                          "preserved_parent_arms": ARMS})


async def select(output, workers, limit):
    from core import inference_telemetry
    from core.generation_profiles import request_settings
    from core.inference_transport import InferenceTransport
    from core.vllm_client import VLLMClient
    from models.prehop.graphrag import GraphRAG
    from models.prehop.tracing import TraceRecorder, attach_client, trace_identity

    protocol = verify_prepared(output)
    if (digest(Path("utils/prompts/evidence_ranking.py")) != protocol["selector_prompt_sha256"]
            or request_settings("ranking") != protocol["ranking_settings"]):
        raise ValueError("Pilot selector code or settings changed; prepare a new run")
    # Preparation and selection have separate code provenance. A bug fix before
    # the first model call need not rebuild frozen ANN proposals. Once selection
    # starts, never mix selector implementations across resume segments.
    from utils.provenance import code_provenance
    runtime = {"script_sha256": digest(Path(__file__)), "code_provenance": code_provenance(),
               "generation_model": protocol["generation_model"],
               "generation_model_revision": None, "python": sys.version, "workers": workers,
               "transport": InferenceTransport.resolve("core").policy_dict()}
    runtime_path = output / "selection-runtime.json"
    if runtime_path.exists():
        if read(runtime_path) != runtime:
            raise ValueError("Selection runtime changed across resume segments")
    else:
        atomic_json(runtime_path, runtime)
    client = VLLMClient()
    semaphore = asyncio.Semaphore(workers)
    jobs = [(qid, arm) for qid in protocol["query_ids"][:limit] for arm in comparison_arms(protocol)]
    random.Random(protocol["seed"]).shuffle(jobs)

    async def one(qid, arm):
        async with semaphore:
            target = output / "calls" / arm / f"{qid}.json"
            path = output / "inputs" / f"{qid}.json"
            if target.exists():
                record = read(target)
                if record["input_sha256"] != digest(path) or digest(Path(record["events_path"])) != record["events_sha256"]:
                    raise ValueError("Committed selection or trace changed")
                return record["status"] != "completed"
            data = read(path)
            recorder = TraceRecorder(output / "traces" / arm / qid / uuid.uuid4().hex,
                                     metadata={"query_id": qid, "arm": arm},
                                     secrets=(os.environ.get("RAG_INFERENCE_API_KEY", ""),
                                              os.environ.get("NEO4J_PASSWORD", "")))
            engine = object.__new__(GraphRAG)
            # Per-call wrapper, shared underlying SDK pools; do not repeatedly
            # wrap one mutable client across concurrent trace recorders.
            engine.llm = attach_client(VLLMClient(), recorder)
            engine.trace_recorder = recorder
            record = {"query_id": qid, "arm": arm, "input_sha256": digest(path), "model": client.model_name,
                      "transport": InferenceTransport.resolve("core").policy_dict()}
            started = time.perf_counter()
            token = inference_telemetry.begin()
            try:
                _, prompt = engine._ranking_prompt(data["query"], data["conditions"][arm]["ordered"], 12)
                estimated_tokens = engine.llm._count_tokens([{"role": "user", "content": prompt}])
                record["estimated_prompt_tokens"] = estimated_tokens
                if estimated_tokens > engine.llm._generation_max_context_tokens - 1024:
                    raise ValueError("Selector context exceeds the unchanged-message limit")
                with trace_identity(query_id=qid, arm=arm):
                    selected = await engine._role_body_list_ranking(data["query"], data["conditions"][arm]["ordered"], 12)
                record.update(status="completed", retrieved_sources=GraphRAG._build_unique_sources(selected))
            except OSError:
                raise  # Mandatory trace/file failures must not become a committed model failure.
            except Exception as exc:  # noqa: BLE001 -- terminal failures retained, never regenerated for quality
                record.update(status="failed", error=type(exc).__name__, retrieved_sources=[])
            finally:
                record["usage"] = inference_telemetry.finish(token)
                record["selector_seconds"] = time.perf_counter() - started
            events = recorder.directory / "events.jsonl"
            record.update(events_path=str(events.resolve()), events_sha256=digest(events))
            atomic_json(target, record)  # Trace is durable before the result commit.
            print(f"Committed {arm}/{qid}: {record['status']}", flush=True)
            return record["status"] != "completed"

    try:
        if client.model_name != protocol["generation_model"]:
            raise ValueError("Prepared generation model differs")
        return sum(await drain([asyncio.create_task(one(*job)) for job in jobs]))
    finally:
        await close_owned(client.global_close, primary_error=sys.exc_info()[1])


def evidence_funnel(facts, base, pool, selected, base_selected=None):
    """Descriptive gold evaluation only; identities prevent double-counting facts."""
    from utils.metrics import _official_multihoprag_fact_match

    facts = list(dict.fromkeys(facts))

    def covered(sources):
        return {i for i, fact in enumerate(facts)
                if any(_official_multihoprag_fact_match(fact, n["text"]) for n in sources)}

    base_facts, supplied = covered(base), covered(pool)
    new = supplied - base_facts
    survived = new & covered(selected[:10])
    complete = bool(facts) and len(covered(selected[:10])) == len(facts)
    baseline_complete = (bool(facts) and len(covered(base_selected[:10])) == len(facts)
                         if base_selected is not None else None)
    return {"distinct_facts": len(facts), "base_fact_count": len(base_facts),
            "supplied_fact_indices": sorted(new), "survived_fact_indices": sorted(survived),
            "supplied_new_fact": bool(new), "retained_new_fact": bool(survived),
            "complete_after_retention": bool(survived) and complete, "selected_complete": complete,
            "complete_gain_over_base": (complete and not baseline_complete) if baseline_complete is not None else None,
            "complete_loss_to_base": (baseline_complete and not complete) if baseline_complete is not None else None}


def funnel_summary(values):
    supplied = sum(v["supplied_new_fact"] for v in values)
    retained = sum(v["retained_new_fact"] for v in values)
    complete = sum(v["complete_after_retention"] for v in values)
    has_base = bool(values) and all(v["complete_gain_over_base"] is not None for v in values)
    return {"eligible_queries": len(values), "supply_queries": supplied, "retention_queries": retained,
            "complete_after_retention_queries": complete,
            "retention_given_supply": retained / supplied if supplied else None,
            "completion_given_retention": complete / retained if retained else None,
            "all_complete_gains_over_base": sum(v["complete_gain_over_base"] for v in values) if has_base else None,
            "all_complete_losses_to_base": sum(v["complete_loss_to_base"] for v in values) if has_base else None,
            "scope": "Descriptive arm-specific supply subsets, not a paired subgroup performance comparison."}


def evaluate(output, *, fused_only=False):
    from scripts.ablation_statistics import cluster_interval
    from scripts.evaluate_saved_retrieval import multihop_metrics
    from utils.metrics import _official_multihoprag_fact_match

    protocol = verify_prepared(output)
    arms = comparison_arms(protocol)
    queries = {q["_id"]: q for q in read(protocol["queries"])}
    audits = {r["query_id"]: r for r in read(output / "input-audit.json")["details"]}
    selectors = ("fused",) if fused_only else ("fused", "llm")
    details, null_ids = [], []
    for qid in protocol["query_ids"]:
        q = queries[qid]
        data = read(output / "inputs" / f"{qid}.json")
        facts = q["evidence_facts"]
        scores = {}
        selections = {}

        for arm in arms:
            for selector in selectors:
                key = (arm, selector)
                if selector == "fused":
                    selections[key] = (data["conditions"][arm]["fused_sources"], None)
                else:
                    call = read(output / "calls" / arm / f"{qid}.json")
                    if (call["input_sha256"] != digest(output / "inputs" / f"{qid}.json")
                            or digest(Path(call["events_path"])) != call["events_sha256"]):
                        raise ValueError("Selection trace or input changed")
                    selections[key] = (call["retrieved_sources"], call)

        for arm in arms:
            condition = data["conditions"][arm]
            coverage = (sum(any(_official_multihoprag_fact_match(f, n["text"]) for n in condition["ordered"])
                            for f in facts) / len(facts)) if facts else None
            base_ids = set(audits[qid]["base"])
            base = [n for n in condition["ordered"] if n["id"] in base_ids]
            base_coverage = (sum(any(_official_multihoprag_fact_match(f, n["text"]) for n in base)
                                 for f in facts) / len(facts)) if facts else None
            for selector in selectors:
                selected, call = selections[arm, selector]
                base_selected = selections[BASE_ARM, selector][0] if BASE_ARM in arms else None
                metrics = multihop_metrics(selected, facts)
                if metrics is not None:
                    metrics["complete_evidence@10"] = float(metrics["Recall@10"] == 1.0)
                scores[f"{arm}/{selector}"] = {"metrics": metrics, "candidate_fact_recall": coverage,
                                               "added_fact_recall": coverage - base_coverage if facts else None,
                                               "failed": bool(call and call["status"] != "completed"),
                                               "usage": call["usage"] if call else None,
                                               "funnel": evidence_funnel(facts, base, condition["ordered"],
                                                                         selected, base_selected),
                                               "candidate_count": len(condition["ordered"])}
        if not facts:
            null_ids.append(qid)
        details.append({"query_id": qid, "original_query_id": q.get("original_query_id", qid), "scores": scores})
    eligible = [r for r in details if r["query_id"] not in null_ids]
    groups = [r["original_query_id"] for r in eligible]
    labels = list(details[0]["scores"])
    metrics = ["MAP@10", "Recall@10", "complete_evidence@10"]
    conditions = {label: {m: cluster_interval([r["scores"][label]["metrics"][m] for r in eligible], groups)
                          for m in metrics} for label in labels}
    coverage = {arm: {m: cluster_interval([r["scores"][f"{arm}/fused"][m] for r in eligible], groups)
                     for m in ("candidate_fact_recall", "added_fact_recall")} for arm in arms}
    funnels = {label: funnel_summary([r["scores"][label]["funnel"] for r in eligible]) for label in labels}
    contrasts = {}
    for selector in selectors:
        for left, right in combinations(arms, 2):
            a, b = f"{left}/{selector}", f"{right}/{selector}"
            contrasts[f"{a}_minus_{b}"] = {
                m: cluster_interval([r["scores"][a]["metrics"][m] - r["scores"][b]["metrics"][m]
                                     for r in eligible], groups) for m in metrics}
    interactions = {}
    if not fused_only:
        for left, right in combinations(arms, 2):
            interactions[f"{left}_minus_{right}:llm_minus_fused"] = {
                m: cluster_interval([
                    (r["scores"][f"{left}/llm"]["metrics"][m] - r["scores"][f"{left}/fused"]["metrics"][m])
                    - (r["scores"][f"{right}/llm"]["metrics"][m] - r["scores"][f"{right}/fused"]["metrics"][m])
                    for r in eligible], groups) for m in metrics}
    name = "fused-evaluation.json" if fused_only else "evaluation.json"
    atomic_json(output / name, {"protocol_sha256": digest(output / "protocol.json"), "conditions": conditions,
                                "evaluation_script_sha256": digest(Path(__file__)), "candidate_coverage": coverage,
                                "funnels": funnels,
                                "contrasts": contrasts, "selector_interactions": interactions,
                                "eligible": len(eligible), "null_query_ids": null_ids,
                                "failures": {label: sum(r["scores"][label]["failed"] for r in details) for label in labels},
                                "details": details})
    print(json.dumps({label: value["MAP@10"] for label, value in conditions.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--limit", type=int, default=256)
    prep.add_argument("--seed", type=int, default=42)
    base = sub.add_parser("add-base", help="Extend a frozen four-arm pilot offline into a new directory")
    base.add_argument("--prepared", type=Path, required=True)
    base.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("select")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--limit", type=int, help="First N frozen queries for a retained canary")
    score = sub.add_parser("evaluate")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--fused-only", action="store_true")
    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if getattr(args, "workers", 1) <= 0:
        parser.error("--workers must be positive")
    if args.command != "evaluate":
        from scripts.runner_environment import _load_runner_environment
        _load_runner_environment()
    if args.command == "prepare":
        asyncio.run(prepare(args))
    elif args.command == "add-base":
        asyncio.run(add_base(args.prepared, args.output))
    elif args.command == "select":
        with lock(args.output / "selection.lock"):
            failures = asyncio.run(select(args.output, args.workers, args.limit))
        if failures:
            raise SystemExit(f"{failures} terminal selector failures; retained for evaluation")
    else:
        evaluate(args.output, fused_only=args.fused_only)


if __name__ == "__main__":
    main()
