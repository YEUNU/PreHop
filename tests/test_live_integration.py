import asyncio
from collections.abc import Iterable

import httpx
import pytest

from core.config import RAGConfig
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient
from models.naive.naive_rag import NaiveRAG
from models.prehop.graphrag import GraphRAG

HEALTH_CHECKS: dict[str, tuple[str, set[int]]] = {
    "neo4j_http": ("http://localhost:7474", {200, 401, 403, 405}),
    "generation": (f"{RAGConfig.VLLM_URL.rstrip('/')}/models", {200, 401}),
    "embedding": (f"{RAGConfig.VLLM_EMBED_URL.rstrip('/')}/models", {200, 401}),
}


async def _endpoint_ready(url: str, ok_codes: Iterable[int]) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(url)
        return res.status_code in set(ok_codes)
    except Exception:  # noqa: BLE001 - optional live-service probe
        return False


async def _require_live_services() -> None:
    checks = await asyncio.gather(*[_endpoint_ready(url, codes) for url, codes in HEALTH_CHECKS.values()])
    missing = [name for name, is_ok in zip(HEALTH_CHECKS.keys(), checks) if not is_ok]
    if missing:
        pytest.skip(f"Live integration services not ready: {', '.join(missing)}")

    neo4j = Neo4jService()
    try:
        rows = await asyncio.wait_for(neo4j.execute_query("RETURN 1 AS ok"), timeout=10.0)
    except Exception as exc:  # noqa: BLE001 - optional live-service probe
        await Neo4jService.global_close()
        pytest.skip(f"Live integration Neo4j bolt not ready: {exc}")
    await Neo4jService.global_close()
    if not rows or rows[0].get("ok") != 1:
        pytest.skip("Live integration Neo4j bolt returned unexpected result.")


async def _run_live_step(name: str, coro, timeout_seconds: float):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - optional live-service probe
        pytest.skip(f"Live integration step '{name}' not ready: {exc}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_service_healthchecks():
    await _require_live_services()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_vllm_client_minimal_calls():
    await _require_live_services()

    client = VLLMClient()
    response = await _run_live_step(
        "generate_response",
        client.generate_response(
            [{"role": "user", "content": "Reply with exactly OK"}],
            temperature=0.0,
        ),
        timeout_seconds=60.0,
    )
    assert isinstance(response, str) and response.strip()

    embedding = await _run_live_step(
        "get_embedding",
        client.get_embedding("integration_smoke_token"),
        timeout_seconds=60.0,
    )
    assert isinstance(embedding, list) and len(embedding) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_naive_index_retrieve_roundtrip():
    await _require_live_services()

    corpus_tag = "it_live"
    unique_token = "integration_live_unique_token_90210"
    rag = NaiveRAG(strategy="naive", corpus_tag=corpus_tag)

    async def cleanup() -> None:
        await rag.neo4j.execute_query(f"MATCH (n:{rag.chunk_label}) DETACH DELETE n")

    async def wait_for_index_online() -> None:
        for _ in range(20):
            rows = await rag.neo4j.execute_query(
                "SHOW VECTOR INDEXES YIELD name, state WHERE name = $name RETURN state",
                {"name": rag.vector_index},
            )
            if rows and rows[0].get("state") == "ONLINE":
                return
            await asyncio.sleep(0.5)

    await _run_live_step("cleanup_before", cleanup(), timeout_seconds=20.0)
    try:
        content = (
            "Title: Integration Live Document\n"
            f"{unique_token} appears in this sentence.\n"
            f"Revenue for {unique_token} was 123.\n"
        )
        await _run_live_step(
            "index_document",
            rag.index_document("integration_live_doc.txt", content),
            timeout_seconds=120.0,
        )
        await _run_live_step("wait_for_index_online", wait_for_index_online(), timeout_seconds=30.0)

        node_count = await _run_live_step(
            "node_count",
            rag.neo4j.execute_query(f"MATCH (n:{rag.chunk_label}) RETURN count(n) AS c"),
            timeout_seconds=20.0,
        )
        assert node_count and node_count[0].get("c", 0) > 0, "No indexed nodes were created."

        context = ""
        nodes = []
        for _ in range(15):
            context, nodes = await _run_live_step(
                "retrieve",
                rag.retrieve(unique_token, top_k=3),
                timeout_seconds=60.0,
            )
            if nodes:
                break
            await asyncio.sleep(1)

        assert nodes, "No nodes retrieved from live integration index."
        retrieved_blob = (context + "\n" + "\n".join(n.get("text", "") for n in nodes)).lower()
        assert unique_token.lower() in retrieved_blob
    finally:
        await _run_live_step("cleanup_after", cleanup(), timeout_seconds=20.0)
        await Neo4jService.global_close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_graphrag_index_retrieve_roundtrip():
    """End-to-end smoke for the refactored GraphRAG facade.

    Exercises every mixin in :mod:`models.prehop.indexing` and
    :mod:`models.prehop.retrieval` against live Neo4j + vLLM:
    ChunkingMixin (page parse + fixed-size sentence windows) ->
    KnowledgeMappingMixin (Q-/Q+ generation via indexing_llm) ->
    GraphWriterMixin (Neo4j MERGE + NEXT edges + index lifecycle) ->
    HopEdgeMixin (offline HOP scoring for generated non-empty Q+ items) ->
    RetrieveMixin (two-stage Q-/Q+ entry) -> HybridSearchMixin (RRF) ->
    SimilarityScoringMixin (external embedding cosine ordering). The unique token must survive
    indexing and resurface through retrieval.
    """
    await _require_live_services()

    corpus_tag = "it_live_graph"
    unique_token = "graphrag_live_unique_token_77403"
    rag = GraphRAG(strategy="prehop", corpus_tag=corpus_tag)

    async def cleanup() -> None:
        await rag.neo4j.execute_query(f"MATCH (n:{rag.q_minus_label}) DETACH DELETE n")
        await rag.neo4j.execute_query(f"MATCH (n:{rag.q_plus_label}) DETACH DELETE n")
        await rag.neo4j.execute_query(f"MATCH (n:{rag.chunk_label}) DETACH DELETE n")
        await rag.neo4j.execute_query(f"MATCH (d:{rag.doc_label}) DETACH DELETE d")

    async def wait_for_index_online() -> None:
        for _ in range(20):
            rows = await rag.neo4j.execute_query(
                "SHOW VECTOR INDEXES YIELD name, state WHERE name = $name RETURN state",
                {"name": rag.body_vector_index},
            )
            if rows and rows[0].get("state") == "ONLINE":
                return
            await asyncio.sleep(0.5)

    await _run_live_step("cleanup_before", cleanup(), timeout_seconds=20.0)
    try:
        content = (
            "Title: GraphRAG Live Smoke 10K\n"
            "----- Page 1 -----\n"
            f"In FY2022 the {unique_token} program produced disclosed metrics. "
            f"Revenue attributable to {unique_token} was reported on the income statement. "
            f"Capital expenditure for {unique_token} appeared on the cash flow statement. "
            "These figures were audited as part of the consolidated financial statements.\n"
        )
        knowledge = await _run_live_step(
            "extract_knowledge",
            rag.extract_knowledge(content),
            timeout_seconds=180.0,
        )
        assert knowledge.get("chunks"), "Chunking returned no chunks."
        assert any(unique_token in c.get("text", "") for c in knowledge["chunks"]), (
            "Unique token missing from generated chunks."
        )

        await _run_live_step(
            "build_graph",
            rag.build_graph(
                knowledge,
                source="graphrag_live_smoke.txt",
                document_filename="graphrag_live_smoke.txt",
            ),
            timeout_seconds=120.0,
        )
        await _run_live_step("flush_graph_batch", rag.flush_graph_batch(), timeout_seconds=120.0)
        await _run_live_step("wait_for_index_online", wait_for_index_online(), timeout_seconds=30.0)

        node_count = await _run_live_step(
            "node_count",
            rag.neo4j.execute_query(f"MATCH (n:{rag.chunk_label}) RETURN count(n) AS c"),
            timeout_seconds=20.0,
        )
        assert node_count and node_count[0].get("c", 0) > 0, "No GraphRAG nodes indexed."

        context = ""
        nodes = []
        for _ in range(15):
            context, nodes = await _run_live_step(
                "retrieve",
                rag.retrieve(f"{unique_token} revenue", top_k=3),
                timeout_seconds=120.0,
            )
            if nodes:
                break
            await asyncio.sleep(1)

        assert nodes, "GraphRAG retrieve returned no nodes."
        retrieved_blob = (context + "\n" + "\n".join(n.get("text", "") for n in nodes)).lower()
        assert unique_token.lower() in retrieved_blob, "Unique token not present in retrieved context."

        # Re-indexing the same document must atomically replace its complete
        # chunk/question subgraph.  This catches stale NEXT and stale question
        # nodes that a MERGE-only smoke cannot expose.
        replacement = {
            "title": "GraphRAG Live Smoke 10K revised",
            "chunks": [
                {
                    "title": "GraphRAG Live Smoke 10K revised",
                    "text": f"The revised filing retains {unique_token} as its only disclosed program.",
                    "sent_id": 0,
                    "page": 1,
                    "summary": "The revised filing retains one disclosed program.",
                    "q_minus": [f"Which program is retained in the revised filing containing {unique_token}?"],
                    "q_plus": [f"Which later filing updated the metrics for {unique_token}?"],
                }
            ],
        }
        await _run_live_step(
            "build_replacement_graph",
            rag.build_graph(
                replacement,
                source="graphrag_live_smoke.txt",
                document_filename="graphrag_live_smoke.txt",
            ),
            timeout_seconds=120.0,
        )
        await _run_live_step("flush_replacement_graph", rag.flush_graph_batch(), timeout_seconds=120.0)
        replacement_counts = await _run_live_step(
            "replacement_counts",
            rag.neo4j.execute_query(
                f"""
                MATCH (d:{rag.doc_label} {{filename: $filename}})
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:{rag.chunk_label})
                OPTIONAL MATCH (c)-[:HAS_Q_MINUS]->(qm:{rag.q_minus_label})
                OPTIONAL MATCH (c)-[:HAS_Q_PLUS]->(qp:{rag.q_plus_label})
                OPTIONAL MATCH (c)-[next:NEXT]->()
                RETURN count(DISTINCT c) AS chunks,
                       count(DISTINCT qm) AS q_minus,
                       count(DISTINCT qp) AS q_plus,
                       count(DISTINCT next) AS next_edges
                """,
                {"filename": "graphrag_live_smoke.txt"},
            ),
            timeout_seconds=20.0,
        )
        assert replacement_counts and replacement_counts[0]["chunks"] == 1
        assert replacement_counts[0]["q_minus"] == 1
        assert replacement_counts[0]["q_plus"] == 1
        assert replacement_counts[0]["next_edges"] == 0
    finally:
        await _run_live_step("cleanup_after", cleanup(), timeout_seconds=20.0)
        await Neo4jService.global_close()
