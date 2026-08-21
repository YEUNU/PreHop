import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _preset_rag_domain() -> None:
    """Set RAG_DOMAIN (financial|news) BEFORE the prompt modules import, so the
    model-side prompts (hypothetical-query gen, rewrite, rerank, synthesis)
    pick the right framing. Explicit RAG_DOMAIN / --domain win; otherwise infer
    from the queries file's `dataset` marker or the dataset path. Must run
    before `from cli.* import ...` because those transitively import the
    prompt constants, which are selected at import time."""
    if os.environ.get("RAG_DOMAIN", "").strip():
        return

    argv = sys.argv

    def _argval(flag: str):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                return argv[idx + 1]
        for tok in argv:
            if tok.startswith(flag + "="):
                return tok.split("=", 1)[1]
        return None

    domain = _argval("--domain")
    if not domain:
        marker = ""
        queries_file = _argval("--queries_file")
        if queries_file and os.path.exists(queries_file):
            try:
                import json
                with open(queries_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data:
                    marker = str(data[0].get("dataset", "")).strip().lower()
            except Exception:
                marker = ""
        dataset = _argval("--dataset") or ""
        _NEWS_MARKERS = ("multihoprag", "hotpotqa", "musique")
        domain = "news" if (marker in _NEWS_MARKERS or any(m in dataset for m in _NEWS_MARKERS)) else "financial"

    os.environ["RAG_DOMAIN"] = domain


_preset_rag_domain()

from cli.benchmark import run_benchmark_multi_seed, reconcile_pending_judges
from cli.index import run_indexing
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Prehop")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "benchmark", "benchmark_all"], required=True)
    parser.add_argument("--strategy", choices=["naive", "prehop", "hoprag", "ms_graphrag"], default="prehop")
    parser.add_argument("--domain", choices=["financial", "news"], default=None,
                        help="Prompt domain for model-side prompts. Default: auto-detected as 'news' for "
                             "every supported dataset (multihoprag/hotpotqa/musique). Set RAG_DOMAIN to override.")
    parser.add_argument("--model", default="local")
    parser.add_argument("--dataset", default="data/multihoprag_corpus")
    parser.add_argument("--queries_file", default="data/multihoprag_sample200_queries.json")
    parser.add_argument("--clear-graph", action="store_true", help="Clear all Neo4j data before indexing to prevent duplicates")
    parser.add_argument("--corpus-tag", default=None, help="Tag to identify corpus in Neo4j. Different tags prevent data conflicts.")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results to data/debug/ for inspection")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of benchmark queries to evaluate")
    return parser


async def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "index":
            if args.clear_graph:
                neo4j = Neo4jService()
                logger.warning("Clearing all Neo4j data before indexing...")
                await neo4j.execute_query("MATCH (n) DETACH DELETE n")
                logger.info("Neo4j graph cleared successfully.")
            await run_indexing(args.dataset, args.strategy, args.model, args.corpus_tag, args.save_intermediate)
        elif args.mode == "benchmark":
            corpus_tag = args.corpus_tag or "default"
            # Pin the run dir so async judge batches can be reconciled afterwards.
            env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
            timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
            os.environ["RAG_BENCHMARK_TIMESTAMP"] = timestamp
            results_dir = Path("data/results") / timestamp
            await run_benchmark_multi_seed(args.queries_file, args.strategy, args.model, corpus_tag=corpus_tag, limit=args.limit)
            await reconcile_pending_judges(results_dir)
        elif args.mode == "benchmark_all":
            corpus_tag = args.corpus_tag or "default"
            env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
            timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
            results_dir = Path("data/results") / timestamp
            results_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Batch benchmark results will be saved to: %s", results_dir)
            for strategy in ["naive", "prehop", "hoprag", "ms_graphrag"]:
                print(f"\n>>> Running Benchmark for: {strategy.upper()}")
                await run_benchmark_multi_seed(args.queries_file, strategy, args.model, is_batch=True, corpus_tag=corpus_tag, output_dir=results_dir, limit=args.limit)
            # All strategies done. In async batch-judge mode each left a pending
            # manifest; resolve every batch in parallel now (one wait, not four).
            await reconcile_pending_judges(results_dir)
    finally:
        try:
            await Neo4jService.global_close()
        except Exception as exc:
            logger.warning("Failed to close Neo4j driver cleanly: %s", exc)
        try:
            await VLLMClient.global_close()
        except Exception as exc:
            logger.warning("Failed to close VLLM clients cleanly: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
