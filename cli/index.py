import asyncio
import json
import logging
import multiprocessing as _mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from core.config import RAGConfig
from models.prehop.graphrag import GraphRAG
from models.prehop.indexing.chunking import parse_pages_offline
from models.naive.naive_rag import NaiveRAG


# Spawn-based context for the parsing worker pool. Using the default `fork`
# context corrupts the parent process's httpx/openai async clients (vLLM
# requests stop being dispatched after the pool shuts down), which manifests
# as 100% CPU on the main thread but 0 reqs at the vLLM serve endpoint.
_PARSE_MP_CTX = _mp.get_context("spawn")


logger = logging.getLogger("Prehop")


async def _collect_graph_stats(engine, strategy: str) -> Optional[dict]:
    """Query the live graph for structural statistics after indexing.

    Queries the graph directly (not counters threaded through indexing)
    so this is always consistent with what actually landed in Neo4j, the
    same way CLAUDE.md's "Neo4j data layout" integrity probes work. Only
    meaningful for `prehop` (Q-/Q+/HOP structure); other strategies don't
    have this graph shape.
    """
    if strategy != "prehop" or not hasattr(engine, "chunk_label"):
        return None
    chunk_label = engine.chunk_label
    doc_label = engine.doc_label

    chunk_rows = await engine.neo4j.execute_query(f"""
        MATCH (c:{chunk_label})
        RETURN count(c) AS total_chunks,
               count(c.q_minus_embedding) AS q_minus_chunks,
               count(c.q_plus_embedding) AS q_plus_chunks
    """)
    chunk_stats = chunk_rows[0] if chunk_rows else {}

    doc_rows = await engine.neo4j.execute_query(f"MATCH (d:{doc_label}) RETURN count(d) AS total_docs")
    doc_stats = doc_rows[0] if doc_rows else {}

    hop_rows = await engine.neo4j.execute_query(f"""
        MATCH (:{chunk_label})-[r:HOP]->(:{chunk_label})
        RETURN count(r) AS total_hop_edges
    """)
    hop_stats = hop_rows[0] if hop_rows else {}

    total_chunks = chunk_stats.get("total_chunks", 0) or 0
    total_docs = doc_stats.get("total_docs", 0) or 0
    q_minus_chunks = chunk_stats.get("q_minus_chunks", 0) or 0
    q_plus_chunks = chunk_stats.get("q_plus_chunks", 0) or 0
    total_hop_edges = hop_stats.get("total_hop_edges", 0) or 0

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": (total_chunks / total_docs) if total_docs else 0.0,
        "q_minus_coverage": (q_minus_chunks / total_chunks) if total_chunks else 0.0,
        "q_plus_coverage": (q_plus_chunks / total_chunks) if total_chunks else 0.0,
        "total_hop_edges": total_hop_edges,
        # Out-degree among HOP-eligible (Q+-surviving) source chunks only —
        # matches the "wrote N HOP edges over M Q+ chunks" indexing log line.
        "avg_hop_out_degree_per_eligible_chunk": (total_hop_edges / q_plus_chunks) if q_plus_chunks else 0.0,
        # Out-degree across the whole chunk population (most chunks have none
        # since only Q+-surviving chunks can be a HOP source) — overall graph
        # density.
        "avg_hop_out_degree_per_chunk": (total_hop_edges / total_chunks) if total_chunks else 0.0,
    }


async def rebuild_hop_edges(corpus_tag: str, strategy: str = "prehop") -> Optional[dict]:
    """Delete existing HOP edges for a corpus tag and rebuild them under the
    current `RAGConfig.HOP_THRESHOLD` (env `RAG_HOP_THRESHOLD`).

    Chunks/Q-/Q+/embeddings are untouched — HOP-edge construction is a
    post-processing step over already-embedded chunks (see CLAUDE.md
    "Re-index note"), so this is far cheaper than a full reindex. Used by
    `--mode hop_rebuild` and `scripts/threshold_sweep.py` for τ_hop
    sensitivity sweeps, where each threshold value only needs its edges
    rebuilt, not the whole corpus re-chunked/re-embedded.
    """
    if strategy != "prehop":
        logger.error("rebuild_hop_edges only supports strategy=prehop (got %s)", strategy)
        return None
    engine = GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
    chunk_label = engine.chunk_label
    await engine.neo4j.execute_query(f"MATCH (:{chunk_label})-[r:HOP]->(:{chunk_label}) DELETE r")
    await engine.build_all_hop_edges()
    stats = await _collect_graph_stats(engine, strategy)
    logger.info("HOP rebuild complete for corpus_tag=%s (threshold=%s): %s", corpus_tag, RAGConfig.HOP_THRESHOLD, stats)
    return stats


async def run_indexing(
    dataset_path: str,
    strategy: str,
    model_id: str,
    corpus_tag: Optional[str] = None,
    save_intermediate: bool = False,
):
    """Index files using selected strategy with parallel processing."""
    logger.info(
        "Indexing strategy: %s | Dataset: %s | Corpus: %s",
        strategy,
        dataset_path,
        corpus_tag or "default",
    )

    if strategy == "ms_graphrag":
        # Official MS GraphRAG pipeline (extract_graph + Leiden + community
        # reports), routed through LiteLLM → local vLLM. Outputs parquet
        # under data/ms_graphrag_output/<corpus_tag>/. Skips our chunking/
        # HOP/summary stages — MS does its own.
        from models.ms_graphrag.official_indexer import run_official_index as run_ms_index
        await run_ms_index(
            dataset_path=dataset_path,
            corpus_tag=corpus_tag or "default",
        )
        return

    if strategy == "hoprag":
        # Official HopRAG indexing (QABuilder.create_nodes + grouped
        # create_edge + create_index). Routed through OpenAI client → local
        # vLLM, embeddings via vLLM HTTP. Writes nodes/edges directly to
        # Neo4j under HO_<corpus_tag>_* labels.
        from models.hoprag.official_indexer import run_official_index as run_hop_index
        await run_hop_index(
            dataset_path=dataset_path,
            corpus_tag=corpus_tag or "default",
        )
        return

    if strategy == "prehop":
        engine = GraphRAG(
            strategy=strategy,
            indexing_model_id=model_id,
            corpus_tag=corpus_tag,
            save_intermediate=save_intermediate,
        )
        is_graph = True
    elif strategy == "naive":
        engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
        is_graph = False
    else:
        logger.error("Unknown strategy: %s", strategy)
        return

    if not os.path.exists(dataset_path):
        logger.error("Path %s not found.", dataset_path)
        return

    files = sorted([file for file in os.listdir(dataset_path) if file.endswith((".txt", ".md"))])

    # Cap how many files can sit in the post-parse chunking+LLM pipeline
    # simultaneously. Each file's chunker fans out one LLM task per page
    # (page-summary stage is `gather([get_page_summary(p) for p in pages])`),
    # bypassing the per-call semaphore. With 16 files × ~200 pages we'd
    # schedule ~3,200 concurrent coroutines — page-summary fan-out is now
    # gated inside chunking.py via `page_summary_sem` (RAG_MAX_PARALLEL_PAGES)
    # and chunk-level fan-out via `chunk_sem` (MAX_CONCURRENT_LLM_CALLS), so
    # the outer file_semaphore is now the only file-level gate.
    file_concurrency = max(1, int(os.environ.get("RAG_MAX_PARALLEL_FILES", "4")))
    file_semaphore = asyncio.Semaphore(file_concurrency)
    progress = {"started": 0, "completed": 0, "lock": asyncio.Lock()}
    failed_files = []
    stats = {"succeeded": 0}
    progress_step = max(1, int(os.environ.get("RAG_PROGRESS_LOG_STEP", "1")))

    async def _log_progress(stage: str, filename: str):
        async with progress["lock"]:
            done = progress["completed"]
            started = progress["started"]
            failed = len(failed_files)
            total = len(files)
            remaining = total - done
            if stage == "start":
                if started % progress_step == 0 or started == total:
                    logger.info(
                        "Indexing progress | started=%d/%d completed=%d failed=%d remaining=%d | now: %s",
                        started, total, done, failed, remaining, filename,
                    )
            else:
                if done % progress_step == 0 or done == total:
                    logger.info(
                        "Indexing progress | completed=%d/%d failed=%d remaining=%d | finished: %s",
                        done, total, failed, remaining, filename,
                    )

    async def process_file(filename: str, content: str, prepared_pages: Optional[dict] = None):
        async with file_semaphore:
            async with progress["lock"]:
                progress["started"] += 1
            await _log_progress("start", filename)

            try:
                if is_graph:
                    knowledge = await engine.extract_knowledge(
                        content, prepared_pages=prepared_pages
                    )
                    doc_meta = {"title": knowledge["title"]}
                    doc_id = await engine.create_document_node(filename, doc_meta)
                    await engine.build_graph(knowledge, source=filename, document_filename=doc_id)
                else:
                    await engine.index_document(filename, content)
                async with progress["lock"]:
                    stats["succeeded"] += 1
                    progress["completed"] += 1
            except Exception as exc:
                logger.error("Failed to index file %s: %s", filename, exc)
                async with progress["lock"]:
                    failed_files.append({"item": filename, "stage": "index", "error": str(exc)})
                    progress["completed"] += 1
            await _log_progress("done", filename)

    file_contents = []
    for filename in files:
        try:
            with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as file:
                file_contents.append((filename, file.read()))
        except Exception as exc:
            logger.error("Failed to read file %s: %s", filename, exc)
            failed_files.append({"item": filename, "stage": "read", "error": str(exc)})

    # Page parsing is pure-CPU regex/string work — offload to a process pool
    # so it runs in parallel with the GPU pipeline rather than serializing on
    # the GIL. Only used for graph strategies (naive doesn't need pages).
    prepared_lookup: dict[str, dict] = {}
    if is_graph and file_contents:
        worker_count = max(1, min(len(file_contents), os.cpu_count() or 4))
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=_PARSE_MP_CTX) as parse_pool:
            parse_tasks = [
                loop.run_in_executor(parse_pool, parse_pages_offline, fn, ct)
                for fn, ct in file_contents
            ]
            parsed_results = await asyncio.gather(*parse_tasks, return_exceptions=True)
        for (fn, _ct), result in zip(file_contents, parsed_results):
            if isinstance(result, Exception):
                logger.warning("Page parsing failed for %s; will re-parse in main process: %s", fn, result)
                continue
            prepared_lookup[fn] = result
        logger.info(
            "Parallel page parsing complete: %d/%d files prepared (workers=%d).",
            len(prepared_lookup), len(file_contents), worker_count,
        )

    gather_results = await asyncio.gather(
        *[process_file(fn, ct, prepared_lookup.get(fn)) for fn, ct in file_contents],
        return_exceptions=True,
    )
    for idx, result in enumerate(gather_results):
        if isinstance(result, Exception):
            filename = file_contents[idx][0]
            logger.error("Unhandled indexing task error in %s: %s", filename, result)
            failed_files.append({"item": filename, "stage": "task", "error": str(result)})

    if is_graph:
        await engine.flush_graph_batch()

        # One-shot HOP edge construction over the complete graph (paper
        # §3.1.4). Done after all chunks/embeddings are written so every
        # source chunk has the same candidate pool. Strategies that don't
        # use HOP (e.g., naive_rag) won't have this method.
        if hasattr(engine, "build_all_hop_edges"):
            try:
                await engine.build_all_hop_edges()
            except Exception as exc:
                logger.error("HOP edge construction failed: %s", exc)
                failed_files.append({"item": "__hop_edges__", "stage": "hop_edges", "error": str(exc)})

    logger.info(
        "Indexing complete for %d files. Success: %d | Failed: %d",
        len(files),
        stats["succeeded"],
        len(failed_files),
    )
    if failed_files:
        preview = ", ".join([f["item"] for f in failed_files[:10]])
        logger.warning("Indexing failures (up to 10): %s", preview)

        # Persist the full failure list (not just the log preview) so a
        # partial run's errors are queryable/reviewable afterward instead of
        # only living in scrollback. One JSON per (strategy, corpus_tag, run).
        failures_dir = Path("data/index_failures")
        failures_dir.mkdir(parents=True, exist_ok=True)
        failures_path = failures_dir / f"{strategy}_{corpus_tag or 'default'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(failures_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "strategy": strategy,
                    "corpus_tag": corpus_tag or "default",
                    "dataset_path": dataset_path,
                    "total_files": len(files),
                    "succeeded": stats["succeeded"],
                    "failed": len(failed_files),
                    "failures": failed_files,
                }, fh, indent=2, ensure_ascii=False)
            logger.warning("Full failure list written to %s", failures_path)
        except Exception as exc:
            logger.error("Could not write failure log to %s: %s", failures_path, exc)

    # Structured graph/corpus statistics for the paper's dataset/graph
    # tables (chunk/HOP-edge counts, Q-/Q+ coverage) — queried from the live
    # graph so it's always consistent with what actually landed in Neo4j.
    # Best-effort: a stats-collection failure shouldn't fail an otherwise
    # successful indexing run.
    graph_stats = None
    try:
        graph_stats = await _collect_graph_stats(engine, strategy)
    except Exception as exc:
        logger.error("Graph stats collection failed: %s", exc)
    if graph_stats is not None:
        logger.info("Graph stats: %s", graph_stats)
        stats_dir = Path("data/index_stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = stats_dir / f"{strategy}_{corpus_tag or 'default'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "strategy": strategy,
                    "corpus_tag": corpus_tag or "default",
                    "dataset_path": dataset_path,
                    **graph_stats,
                }, fh, indent=2, ensure_ascii=False)
            logger.info("Graph stats written to %s", stats_path)
        except Exception as exc:
            logger.error("Could not write graph stats to %s: %s", stats_path, exc)
