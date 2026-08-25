import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j.exceptions import TransientError

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


_DEFAULT_DATASET = "data/multihoprag_corpus"
_DEFAULT_QUERIES_FILE = "data/multihoprag_sample200_queries.json"


def _ensure_run_id() -> str:
    """Return one filesystem-safe identifier shared by every child artifact."""
    raw = os.environ.get("RAG_RUN_ID")
    if not raw:
        raw = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{os.getpid()}"
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-") or f"run_{os.getpid()}"
    os.environ["RAG_RUN_ID"] = run_id
    return run_id


from cli.benchmark import reconcile_pending_judges, run_benchmark_multi_seed
from cli.index import rebuild_hop_edges, run_indexing
from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# The OpenAI/httpx client logs one line per request at INFO.  Full indexing
# emits millions of successful calls, obscuring progress and real warnings.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("Prehop")


async def _clear_graph_and_schema(neo4j: Neo4jService) -> None:
    """Remove graph data and stale application schema from the selected DB."""
    # Drop application schema first. Besides ensuring a cold rebuild, this
    # releases vector/full-text maintenance state before large node deletes.
    constraints = await neo4j.execute_query("SHOW CONSTRAINTS YIELD name RETURN name")
    for row in constraints:
        name = str(row.get("name", ""))
        if name:
            escaped = name.replace("`", "``")
            await neo4j.execute_query(f"DROP CONSTRAINT `{escaped}` IF EXISTS")

    # Preserve Neo4j's built-in token lookup indexes; every vector/full-text/
    # property index in this dedicated project database is rebuilt by the
    # selected strategy with the current embedding dimension.
    indexes = await neo4j.execute_query("SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name")
    for row in indexes:
        name = str(row.get("name", ""))
        if name:
            escaped = name.replace("`", "``")
            await neo4j.execute_query(f"DROP INDEX `{escaped}` IF EXISTS")

    # Delete in committed batches to stay within Neo4j transaction memory.
    # Reduce only the batch that raises a transient memory error; rollback
    # keeps each retry atomic.
    clear_batch = int(os.environ.get("RAG_NEO4J_CLEAR_BATCH_SIZE", "1000"))
    if clear_batch < 1:
        raise ValueError("RAG_NEO4J_CLEAR_BATCH_SIZE must be positive")
    deleted_total = 0
    while True:
        try:
            rows = await neo4j.execute_query(
                "MATCH (n) WITH n LIMIT $batch DETACH DELETE n RETURN count(*) AS deleted",
                {"batch": clear_batch},
            )
        except TransientError as exc:
            if "MemoryPoolOutOfMemoryError" not in str(exc) or clear_batch == 1:
                raise
            clear_batch = max(1, clear_batch // 2)
            logger.warning("Neo4j clear hit transaction memory limit; retrying with batch=%d", clear_batch)
            continue
        deleted = int(rows[0].get("deleted", 0)) if rows else 0
        deleted_total += deleted
        if deleted == 0:
            break
    logger.info("Neo4j clear removed %d nodes in bounded transactions", deleted_total)

    verification = await neo4j.execute_query("MATCH (n) RETURN count(n) AS node_count")
    remaining = await neo4j.execute_query(
        "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN count(*) AS index_count"
    )
    if (verification[0]["node_count"] if verification else -1) != 0:
        raise RuntimeError("Neo4j clear verification failed: nodes remain")
    if (remaining[0]["index_count"] if remaining else -1) != 0:
        raise RuntimeError("Neo4j clear verification failed: application indexes remain")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["index", "benchmark", "benchmark_all", "hop_rebuild", "clear_graph"],
        required=True,
    )
    parser.add_argument("--strategy", choices=["naive", "prehop", "hoprag", "ms_graphrag"], default="prehop")
    parser.add_argument("--model", default="default")
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--queries_file", default=_DEFAULT_QUERIES_FILE)
    parser.add_argument(
        "--clear-graph", action="store_true", help="Clear all Neo4j data before indexing to prevent duplicates"
    )
    parser.add_argument(
        "--corpus-tag", default=None, help="Tag to identify corpus in Neo4j. Different tags prevent data conflicts."
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediates under data/debug/<run-id>/<strategy>/<corpus>/",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of benchmark queries to evaluate")
    return parser


async def main():
    parser = _build_parser()
    args = parser.parse_args()
    run_id = _ensure_run_id()
    logger.info("Run ID: %s", run_id)

    try:
        if args.mode == "clear_graph":
            neo4j = Neo4jService()
            logger.warning("Clearing all Neo4j data and application schema...")
            await _clear_graph_and_schema(neo4j)
            logger.info("Neo4j graph and application schema cleared successfully.")
        elif args.mode == "index":
            RAGConfig.validate()
            if args.clear_graph:
                neo4j = Neo4jService()
                logger.warning("Clearing all Neo4j data and application schema before indexing...")
                await _clear_graph_and_schema(neo4j)
                logger.info("Neo4j graph and application schema cleared successfully.")
            await run_indexing(args.dataset, args.strategy, args.model, args.corpus_tag, args.save_intermediate)
        elif args.mode == "hop_rebuild":
            RAGConfig.validate()
            await rebuild_hop_edges(args.corpus_tag or "default", args.strategy)
        elif args.mode == "benchmark":
            RAGConfig.validate()
            corpus_tag = args.corpus_tag or "default"
            env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
            timestamp = env_ts if env_ts else run_id
            os.environ["RAG_BENCHMARK_TIMESTAMP"] = timestamp
            benchmark_failure = None
            try:
                await run_benchmark_multi_seed(
                    args.queries_file, args.strategy, args.model, corpus_tag=corpus_tag, limit=args.limit
                )
            except Exception as exc:  # noqa: BLE001 - reconcile any batches submitted by earlier seeds
                benchmark_failure = exc
            reconcile_failure = None
            try:
                await reconcile_pending_judges(Path("data/results") / timestamp)
            except Exception as exc:  # noqa: BLE001 - report both benchmark and judge failures
                reconcile_failure = exc
            if benchmark_failure or reconcile_failure:
                failures = [
                    f"benchmark: {type(benchmark_failure).__name__}: {benchmark_failure}" if benchmark_failure else "",
                    f"judge_reconcile: {type(reconcile_failure).__name__}: {reconcile_failure}"
                    if reconcile_failure
                    else "",
                ]
                raise RuntimeError("; ".join(item for item in failures if item))
        elif args.mode == "benchmark_all":
            RAGConfig.validate()
            corpus_tag = args.corpus_tag or "default"
            env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
            timestamp = env_ts if env_ts else run_id
            results_dir = Path("data/results") / timestamp
            results_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Batch benchmark results will be saved to: %s", results_dir)
            strategy_failures = []
            for strategy in ["naive", "prehop", "hoprag", "ms_graphrag"]:
                print(f"\n>>> Running Benchmark for: {strategy.upper()}")
                try:
                    await run_benchmark_multi_seed(
                        args.queries_file,
                        strategy,
                        args.model,
                        is_batch=True,
                        corpus_tag=corpus_tag,
                        output_dir=results_dir,
                        limit=args.limit,
                    )
                except Exception as exc:  # noqa: BLE001 - aggregate independent strategy failures
                    logger.error("Benchmark strategy %s failed: %s", strategy, exc)
                    strategy_failures.append(f"{strategy}: {type(exc).__name__}: {exc}")
            try:
                await reconcile_pending_judges(results_dir)
            except Exception as exc:  # noqa: BLE001 - preserve strategy failures and report judge failure too
                logger.error("Judge batch reconciliation failed: %s", exc)
                strategy_failures.append(f"judge_reconcile: {type(exc).__name__}: {exc}")
            if strategy_failures:
                raise RuntimeError("benchmark_all completed with strategy failures: " + "; ".join(strategy_failures))
    finally:
        try:
            await Neo4jService.global_close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the primary failure
            logger.warning("Failed to close Neo4j driver cleanly: %s", exc)
        try:
            await VLLMClient.global_close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the primary failure
            logger.warning("Failed to close VLLM clients cleanly: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
