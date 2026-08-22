"""Rank-Based HOP Edge Pre-Construction (core-only rewrite).

For each source chunk c_i with Q+ embedding q+_i:
1. Retrieve top-15 candidates from the Q+ vector index by ANN. The Neo4j
   vector index already returns a cosine-similarity score per candidate.
2. Keep edges where that score >= tau_hop (RAGConfig.HOP_THRESHOLD),
   retaining the top L_hop=RAGConfig.HOP_LINK_LIMIT edges by score.

Multi-hop discovery happens once, at indexing time, entirely from embeddings
already computed for Q+ indexing — no extra scoring model or LLM call is
needed. (Earlier iteration used a cross-encoder reranker for step 2; that
model is no longer part of the served model set, so edge scoring now reuses
the ANN similarity score directly.) The same tau_hop is used at retrieval
(§3.2.3 graph traversal, runtime HOP mode) so HOP edges follow the same
scoring criterion the system applies when reading them.
"""
import asyncio
import logging
import os
from typing import Any

from core.config import RAGConfig


logger = logging.getLogger(__name__)


class HopEdgeMixin:
    async def _find_hop_candidates(self, hop_src: dict[str, Any]) -> list[dict[str, Any]]:
        if not hop_src.get("q_plus_embed"):
            return []

        # Same-company filter: in v14 we observed 18% of HOP edges crossed
        # company boundaries (e.g., AES → AMAZON, ADOBE → ACTIVISIONBLIZZARD)
        # because the cross-encoder reranker confused structurally similar
        # finance tables across unrelated tickers. FinanceBench queries are
        # company-anchored, so cross-company HOPs add retrieval noise without
        # answering the actual question. We restrict candidates to the same
        # company prefix; same-source is still excluded to keep edges
        # cross-document (paper §3.1.4 multi-hop discovery).
        #
        # For news/multi-hop corpora (RAGConfig.COMPANY_ANCHORING == False) the
        # "company" derived from a news filename is just its leading token
        # (e.g. "THE", "2023", "NFL"), so this filter would confine HOP edges to
        # articles that coincidentally share a first token — collapsing the
        # multi-hop graph. Disable it there; same-source exclusion still keeps
        # edges cross-document.
        src_company = (hop_src.get("company") or "") if RAGConfig.COMPANY_ANCHORING else ""

        query = """
            CALL db.index.vector.queryNodes($index, 15, $embed)
            YIELD node, score
            WHERE node.id <> $src_id
              AND node.source <> $src_source
              AND ($src_company = '' OR node.company = $src_company)
              AND node.q_plus_embedding IS NOT NULL
            RETURN node.id as id, node.text as text, score
        """
        params = {
            "index": self.q_plus_vector_index,
            "embed": hop_src["q_plus_embed"],
            "src_id": hop_src["id"],
            "src_source": hop_src["source"],
            "src_company": src_company,
        }
        return await self.retry_query(query, params)

    async def _process_hop_wave(
        self,
        wave: list[dict[str, Any]],
        hop_sem: asyncio.Semaphore,
    ) -> list[dict[str, Any]]:
        """Score one wave of hop_src dicts concurrently, return their edges.

        Edge score is the ANN cosine-similarity score Neo4j's vector index
        already returns for each candidate — no extra model call needed.
        """
        async def _process_hop_src(hop_src: dict[str, Any]) -> list[dict[str, Any]]:
            async with hop_sem:
                candidates = await self._find_hop_candidates(hop_src)
                hop_src.pop("q_plus_embed", None)  # free embedding ASAP
                if not candidates:
                    return []

                valid_edges = [
                    {
                        "src_id": hop_src["id"],
                        "tgt_id": candidate["id"],
                        "score": candidate["score"],
                    }
                    for candidate in candidates
                    if candidate["score"] >= RAGConfig.HOP_THRESHOLD
                ]
                valid_edges.sort(key=lambda item: item["score"], reverse=True)
                return valid_edges[: RAGConfig.HOP_LINK_LIMIT]

        edge_groups = await asyncio.gather(
            *[_process_hop_src(src) for src in wave],
            return_exceptions=False,
        )
        return [edge for group in edge_groups for edge in group]

    async def _flush_hop_edges(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return
        await self.retry_query(f"""
            UNWIND $edges AS edge
            MATCH (src:{self.chunk_label} {{id: edge.src_id}})
            MATCH (tgt:{self.chunk_label} {{id: edge.tgt_id}})
            MERGE (src)-[r:HOP]->(tgt)
            SET r.score = edge.score, r.type = 'pruned'
        """, {"edges": edges})

    async def build_all_hop_edges(self) -> None:
        """One-shot HOP edge pre-construction over the COMPLETE graph
        (paper §3.1.4: 'Multi-hop discovery happens once, at indexing time').

        Streams Q+ chunks from Neo4j in pages of HOP_PAGE_SIZE so peak RSS
        stays bounded regardless of corpus size. Each page is split into
        gather waves of HOP_GATHER_WAVE coroutines and edges are flushed to
        Neo4j after every wave (no global accumulator). Restarts after a
        crash continue cleanly because MERGE on (src)-[:HOP]->(tgt) is
        idempotent — already-scored Q+ chunks just re-write identical edges.

        Asymmetry concern: every chunk still sees the full corpus as ANN
        candidates because the Q+ vector index covers the whole label, not
        just the current page.
        """
        if (RAGConfig.HOP_MODE != "offline") or (not RAGConfig.ABLATION_Q_PLUS):
            logger.info(
                "Skipping offline HOP edge construction (HOP_MODE=%s, ABLATION_Q_PLUS=%s).",
                RAGConfig.HOP_MODE,
                RAGConfig.ABLATION_Q_PLUS,
            )
            return

        page_size = max(100, int(os.environ.get("RAG_HOP_PAGE_SIZE", "5000")))
        wave_size = max(1, int(os.environ.get("RAG_HOP_GATHER_WAVE", "1000")))
        hop_sem = asyncio.Semaphore(
            max(1, int(os.environ.get("RAG_HOP_RERANK_CONCURRENCY", "64")))
        )

        total_proc = 0
        total_edges = 0
        skip = 0
        while True:
            rows = await self.retry_query(f"""
                MATCH (c:{self.chunk_label})
                WHERE c.q_plus_embedding IS NOT NULL
                  AND c.q_plus_text IS NOT NULL AND c.q_plus_text <> ''
                RETURN c.id AS id, c.source AS source, c.company AS company,
                       c.q_plus_embedding AS q_plus_embed, c.q_plus_text AS q_plus_text
                ORDER BY c.id
                SKIP $skip LIMIT $limit
            """, {"skip": skip, "limit": page_size})
            if not rows:
                break

            page_items = [
                {
                    "id": r["id"],
                    "source": r["source"],
                    "company": r.get("company") or "",
                    "q_plus_embed": r["q_plus_embed"],
                    "q_plus": [r["q_plus_text"]],
                }
                for r in rows
            ]
            page_n = len(page_items)
            del rows  # release the original Neo4j result list

            if total_proc == 0:
                logger.info(
                    "build_all_hop_edges: streaming HOP scoring (page_size=%d, wave=%d, sem=%d).",
                    page_size, wave_size,
                    int(os.environ.get("RAG_HOP_RERANK_CONCURRENCY", "64")),
                )

            for wave_start in range(0, page_n, wave_size):
                wave = page_items[wave_start : wave_start + wave_size]
                edges = await self._process_hop_wave(wave, hop_sem)
                await self._flush_hop_edges(edges)
                total_edges += len(edges)

            total_proc += page_n
            del page_items
            skip += page_size
            logger.info(
                "build_all_hop_edges: progress %d processed / %d edges so far.",
                total_proc, total_edges,
            )

            if page_n < page_size:
                break

        logger.info("build_all_hop_edges: wrote %d HOP edges over %d Q+ chunks.", total_edges, total_proc)
