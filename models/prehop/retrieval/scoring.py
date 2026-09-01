"""Parameter-free fusion of indexed representation and body semantics."""

import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from core.config import RAGConfig
from models.prehop.llm_json import generate_json_or_raise
from utils.prompts.query_rewrite import build_evidence_ranking_prompt
from utils.similarity import cosine_similarity

_CANDIDATE_ORDER_TRACE_LOCK = asyncio.Lock()


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


class SimilarityScoringMixin:
    @staticmethod
    def _document_identity(node: dict[str, Any]) -> str:
        """Return the logical document identity used for evidence diversity.

        Some prepared corpora store one paragraph per file, so ``source`` is
        a chunk container rather than a document boundary. The indexed title
        is the shared, dataset-neutral document identity in that case. Keep
        the filename as a fallback for sources without a title.
        """
        return str(node.get("title") or node.get("doc") or node.get("source") or "")

    @staticmethod
    def _validated_similarity(query_embedding: list[float], document_embedding: Any) -> float:
        if not isinstance(document_embedding, list) or not document_embedding:
            raise ValueError("Retrieved candidate is missing its indexed embedding")
        if len(document_embedding) != len(query_embedding):
            raise ValueError("Query and indexed candidate embedding dimensions do not match")
        return cosine_similarity(query_embedding, document_embedding)

    async def _score_and_select(
        self,
        query_embedding: list[float],
        candidates: list[dict[str, Any]],
        top_k: int,
        query_text: str = "",
        selection_variant: str | None = None,
        timing_sink: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fuse representation ranks with body and stored bridge semantics.

        The best matching individual source Q+ represents a traversed
        dependency bridge; linked continuation paths use the matched Q− in the
        same role. The conservative default takes its minimum with direct body
        relevance. A query-time ablation uses the bridge alone because the
        offline graph already selected the target body. Equal reciprocal ranks
        then combine the resulting semantic order with the Q-/body/Q+ retrieval
        order. Neither path compares backend-specific raw scores or introduces
        a fitted interpolation weight or threshold.
        """
        score_started = time.perf_counter()
        if not candidates:
            return [], []
        if not query_embedding:
            raise ValueError("Final scoring received an empty query embedding")

        for candidate in candidates:
            body_score = self._validated_similarity(query_embedding, candidate.get("embedding"))
            bridge_embeddings = candidate.get("bridge_embeddings") or []
            bridge_scores = [
                self._validated_similarity(query_embedding, embedding) for embedding in bridge_embeddings if embedding
            ]
            bridge_score = max(bridge_scores) if bridge_scores else None
            if bridge_score is None or RAGConfig.HOP_SEMANTIC_VARIANT == "body_only":
                final_score = body_score
            elif RAGConfig.HOP_SEMANTIC_VARIANT == "bridge_only":
                final_score = bridge_score
            else:
                final_score = min(body_score, bridge_score)
            candidate["similarity_score"] = body_score
            if bridge_score is not None:
                candidate["bridge_similarity_score"] = bridge_score
            candidate["final_score"] = final_score

        semantic_order = sorted(
            candidates,
            key=lambda item: (item.get("final_score", 0.0), self._node_identity(item)),
            reverse=True,
        )
        representation_order = sorted(
            candidates,
            key=lambda item: (float(item.get("representation_score", 0.0)), self._node_identity(item)),
            reverse=True,
        )
        semantic_ranks = {self._node_identity(candidate): rank for rank, candidate in enumerate(semantic_order)}
        representation_ranks = {
            self._node_identity(candidate): rank
            for rank, candidate in enumerate(representation_order)
            if float(candidate.get("representation_score", 0.0)) > 0.0
        }
        for candidate in candidates:
            node_id = self._node_identity(candidate)
            score = 1.0 / (semantic_ranks[node_id] + 1)
            if node_id in representation_ranks:
                score += 1.0 / (representation_ranks[node_id] + 1)
            candidate["rank_fusion_score"] = score

        fused_order = sorted(
            candidates,
            key=lambda item: (
                float(item.get("rank_fusion_score", 0.0)),
                float(item.get("final_score", 0.0)),
                self._node_identity(item),
            ),
            reverse=True,
        )
        if RAGConfig.FINAL_RANK_VARIANT == "semantic_only":
            ordered = semantic_order
        elif RAGConfig.FINAL_RANK_VARIANT == "representation_only":
            ordered = representation_order
        else:
            ordered = fused_order
        if timing_sink is not None:
            timing_sink["deterministic_score_ms"] = (time.perf_counter() - score_started) * 1000
        ordering_started = time.perf_counter()
        active_selection_variant = selection_variant or RAGConfig.SOURCE_SELECTION_VARIANT
        if active_selection_variant == "global":
            selected = ordered[:top_k]
        elif active_selection_variant == "round_robin":
            selected = self._source_round_robin(ordered, top_k)
        elif active_selection_variant == "source_balanced":
            selected = self._source_balanced(ordered, top_k)
        elif active_selection_variant == "graph_pairs":
            selected = self._graph_pairs(ordered, top_k)
        elif active_selection_variant == "source_balanced_graph_pairs":
            selected = self._graph_pairs(self._source_balanced_order(ordered), top_k)
        elif active_selection_variant == "role_body_owners":
            selected = self._role_body_owners(ordered, top_k)
        elif active_selection_variant == "role_body_rounds":
            selected = self._role_body_rounds(ordered, top_k)
        elif active_selection_variant == "role_body_list_ranking":
            selected = await self._role_body_list_ranking(query_text, ordered, top_k)
        else:
            raise ValueError(f"Unsupported source selection variant: {active_selection_variant!r}")
        if timing_sink is not None:
            timing_sink["candidate_order_ms"] = (time.perf_counter() - ordering_started) * 1000
        return selected, ordered

    async def _role_body_list_ranking(
        self,
        query_text: str,
        ordered: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rank the complete established candidate pool by opaque paragraph ID."""
        if not query_text.strip():
            raise ValueError("Body evidence candidate ordering requires the original query text")

        pool: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        for node in ordered:
            node_id = self._node_identity(node)
            if node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                pool.append(node)
        if not pool:
            return ordered[:top_k]

        canonical_pool = list(pool)
        if RAGConfig.CANDIDATE_ORDER_INPUT_ORDER == "reverse":
            pool.reverse()
        elif RAGConfig.CANDIDATE_ORDER_INPUT_ORDER == "hash_shuffle":
            seed = RAGConfig.CANDIDATE_ORDER_SHUFFLE_SEED

            def shuffle_key(node: dict[str, Any]) -> tuple[bytes, str]:
                node_id = self._node_identity(node)
                payload = f"{seed}\0{query_text}\0{node_id}".encode()
                return hashlib.sha256(payload).digest(), node_id

            pool.sort(key=shuffle_key)

        candidate_ids = [f"C{index:03d}" for index in range(len(pool))]
        node_by_candidate_id = dict(zip(candidate_ids, pool, strict=True))
        prompt = build_evidence_ranking_prompt(
            query_text,
            [
                (
                    candidate_id,
                    str(node.get("title") or node.get("doc") or ""),
                    "; ".join(
                        f"{label}: {node[key]}"
                        for key, label in (
                            ("publisher", "Publisher"),
                            ("published_at", "Published"),
                            ("author", "Author"),
                            ("category", "Category"),
                        )
                        if node.get(key)
                    ),
                    str(node.get("text") or ""),
                )
                for candidate_id, node in node_by_candidate_id.items()
            ],
            top_k,
        )
        payload = await generate_json_or_raise(
            self.llm,
            [{"role": "user", "content": prompt}],
            "evidence ranking",
            f"query={query_text!r}",
            required_fields={"ranking": list},
            temperature=0.0,
            max_tokens=1024,
        )

        model_ranking: list[str] = []
        for value in payload["ranking"]:
            candidate_id = str(value)
            if candidate_id not in node_by_candidate_id:
                continue
            if candidate_id not in model_ranking:
                model_ranking.append(candidate_id)
        ranking = list(model_ranking)
        ranking.extend(candidate_id for candidate_id in candidate_ids if candidate_id not in ranking)

        selected = [node_by_candidate_id[candidate_id] for candidate_id in ranking[:top_k]]
        trace_path = os.environ.get("RAG_CANDIDATE_ORDER_TRACE_PATH", "").strip()
        if trace_path:
            record = {
                "query": query_text,
                "top_k": top_k,
                "input_order": RAGConfig.CANDIDATE_ORDER_INPUT_ORDER,
                "shuffle_seed": RAGConfig.CANDIDATE_ORDER_SHUFFLE_SEED,
                "canonical_node_ids": [self._node_identity(node) for node in canonical_pool],
                "candidates": [
                    {
                        "node_id": self._node_identity(node),
                        "title": str(node.get("title") or node.get("doc") or ""),
                        "source": str(node.get("source") or ""),
                        "paragraph_id": str(node.get("paragraph_id") or ""),
                        "metadata": {
                            key: str(node[key])
                            for key in ("publisher", "published_at", "author", "category")
                            if node.get(key)
                        },
                        "text": str(node.get("text") or ""),
                        "similarity_score": float(node.get("similarity_score", 0.0)),
                        "bridge_similarity_score": (
                            float(node["bridge_similarity_score"]) if node.get("bridge_similarity_score") is not None else None
                        ),
                        "final_score": float(node.get("final_score", 0.0)),
                        "representation_score": float(node.get("representation_score", 0.0)),
                        "representation_scores": {
                            str(key): float(value)
                            for key, value in (node.get("representation_scores") or {}).items()
                        },
                        "rank_fusion_score": float(node.get("rank_fusion_score", 0.0)),
                        "retrieval_paths": node.get("retrieval_paths") or [],
                    }
                    for node in canonical_pool
                ],
                "model_returned_node_ids": [
                    self._node_identity(node_by_candidate_id[candidate_id]) for candidate_id in model_ranking
                ],
                "selected_node_ids": [self._node_identity(node) for node in selected],
            }
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            async with _CANDIDATE_ORDER_TRACE_LOCK:
                await asyncio.to_thread(_append_jsonl, Path(trace_path), line)
        selected_ids = {self._node_identity(node) for node in selected}
        for node in ordered:
            node_id = self._node_identity(node)
            if len(selected) >= top_k:
                break
            if node_id and node_id not in selected_ids:
                selected.append(node)
                selected_ids.add(node_id)
        return selected

    @classmethod
    def _role_body_rounds(
        cls,
        ordered: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Give every generated role view one body rank before the next rank."""
        by_rank: dict[int, dict[str, tuple[dict[str, Any], float]]] = defaultdict(dict)
        for node in ordered:
            node_id = cls._node_identity(node)
            for entry in node.get("role_body_round_entries") or []:
                rank = int(entry["rank"])
                score = float(entry.get("score", 0.0))
                current = by_rank[rank].get(node_id)
                if current is None or score > current[1]:
                    by_rank[rank][node_id] = (node, score)

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def append(node: dict[str, Any]) -> None:
            node_id = cls._node_identity(node)
            if node_id and node_id not in selected_ids and len(selected) < top_k:
                selected.append(node)
                selected_ids.add(node_id)

        for rank in sorted(by_rank):
            wave = sorted(
                by_rank[rank].values(),
                key=lambda item: (-item[1], cls._node_identity(item[0])),
            )
            for node, _score in wave:
                append(node)
        for node in ordered:
            append(node)
        return selected

    @classmethod
    def _role_body_owners(
        cls,
        ordered: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Retain each generated role view's first body result, then global order."""
        owner_by_order: dict[int, dict[str, Any]] = {}
        for node in ordered:
            for owner_order in node.get("role_body_owner_orders") or []:
                owner_by_order.setdefault(int(owner_order), node)

        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def append(node: dict[str, Any]) -> None:
            node_id = cls._node_identity(node)
            if node_id and node_id not in selected_ids and len(selected) < top_k:
                selected.append(node)
                selected_ids.add(node_id)

        for owner_order in sorted(owner_by_order):
            append(owner_by_order[owner_order])
        for node in ordered:
            append(node)
        return selected

    @staticmethod
    def _source_round_robin(ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Take one ranked chunk per source per round, then repeat."""
        groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        source_order: list[str] = []
        for node in ordered:
            source = SimilarityScoringMixin._document_identity(node)
            if source not in groups:
                source_order.append(source)
            groups[source].append(node)

        selected: list[dict[str, Any]] = []
        while len(selected) < top_k:
            progressed = False
            for source in source_order:
                if groups[source] and len(selected) < top_k:
                    selected.append(groups[source].popleft())
                    progressed = True
            if not progressed:
                break
        return selected

    @staticmethod
    def _source_balanced(ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Discount repeated-source candidates by their selected occurrence."""
        return SimilarityScoringMixin._source_balanced_order(ordered)[:top_k]

    @staticmethod
    def _source_balanced_order(ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return every candidate in soft document-diversified order."""
        remaining = list(ordered)
        source_counts: dict[str, int] = defaultdict(int)
        selected: list[dict[str, Any]] = []
        while remaining:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    float(remaining[index].get("rank_fusion_score", 0.0))
                    / (source_counts[SimilarityScoringMixin._document_identity(remaining[index])] + 1),
                    float(remaining[index].get("final_score", 0.0)),
                    -index,
                ),
            )
            node = remaining.pop(best_index)
            source = SimilarityScoringMixin._document_identity(node)
            selected.append(node)
            source_counts[source] += 1
        return selected

    @classmethod
    def _graph_pairs(cls, ordered: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Keep a high-ranked HOP target with its best-ranked source owner."""
        by_id = {cls._node_identity(node): node for node in ordered}
        rank = {cls._node_identity(node): index for index, node in enumerate(ordered)}
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def append(node: dict[str, Any]) -> None:
            node_id = cls._node_identity(node)
            if node_id and node_id not in selected_ids and len(selected) < top_k:
                selected.append(node)
                selected_ids.add(node_id)

        for node in ordered:
            if len(selected) >= top_k:
                break
            hop_sources = {
                str(path.get("source_chunk_id") or "")
                for path in (node.get("retrieval_paths") or [])
                if path.get("kind") == "hop" and str(path.get("source_chunk_id") or "") in by_id
            }
            if not hop_sources or len(selected) == top_k - 1:
                append(node)
                continue
            best_source_id = min(hop_sources, key=lambda source_id: rank[source_id])
            pair = sorted((node, by_id[best_source_id]), key=lambda item: rank[cls._node_identity(item)])
            for pair_node in pair:
                append(pair_node)
        return selected
