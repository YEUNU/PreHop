"""Embed non-empty indexing text with a revision-safe local cache."""

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

from core.config import RAGConfig

_EMBEDDING_CACHE_LOCK = threading.Lock()


class SparseEmbeddingMixin:
    @staticmethod
    def _embedding_cache_enabled() -> bool:
        value = os.environ.get("RAG_EMBEDDING_CACHE", os.environ.get("RAG_CHUNK_CACHE", "on"))
        return value.lower() not in {"off", "false", "0", "no"}

    @staticmethod
    def _embedding_cache_path() -> Path:
        root = Path(os.environ.get("RAG_EMBEDDING_CACHE_DIR", "data/embedding_cache"))
        return root / "embeddings.sqlite3"

    def _embedding_cache_key(self, text: str, encoding_type: str) -> str:
        identity = {
            "model": RAGConfig.EMBEDDING_MODEL,
            "dimensions": self.vector_dimensions,
            "max_input_tokens": RAGConfig.MAX_EMBEDDING_LENGTH,
            "encoding_type": encoding_type,
            "endpoint": RAGConfig.VLLM_EMBED_URL,
            "revision": os.environ.get("RAG_EMBEDDING_REVISION", ""),
            "query_instruction": RAGConfig.EMBEDDING_QUERY_INSTRUCTION,
            "text": text,
        }
        payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_embedding_cache(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys or not self._embedding_cache_enabled():
            return {}
        path = self._embedding_cache_path()
        if not path.exists():
            return {}
        with _EMBEDDING_CACHE_LOCK, sqlite3.connect(path, timeout=30) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector_json TEXT NOT NULL)"
            )
            rows: list[tuple[str, str]] = []
            for offset in range(0, len(keys), 500):
                batch = keys[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    connection.execute(
                        f"SELECT cache_key, vector_json FROM embeddings WHERE cache_key IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
        return {key: json.loads(vector_json) for key, vector_json in rows}

    def _write_embedding_cache(self, values: dict[str, list[float]]) -> None:
        if not values or not self._embedding_cache_enabled():
            return
        path = self._embedding_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _EMBEDDING_CACHE_LOCK, sqlite3.connect(path, timeout=30) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector_json TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, vector_json) VALUES (?, ?)",
                [(key, json.dumps(vector, separators=(",", ":"))) for key, vector in values.items()],
            )

    async def _embed_sparse_texts(
        self,
        texts: list[str],
        *,
        encoding_type: str = "document",
    ) -> list[list[float]]:
        if not texts:
            return []
        positions: list[int] = []
        payload: list[str] = []
        for index, text in enumerate(texts):
            normalized = str(text or "").strip()
            if normalized:
                positions.append(index)
                payload.append(normalized)

        result: list[list[float]] = [[] for _ in texts]
        if not payload:
            return result

        keys = [self._embedding_cache_key(text, encoding_type) for text in payload]
        cached = await asyncio.to_thread(self._read_embedding_cache, keys)
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for key, text in zip(keys, payload):
            vector = cached.get(key)
            if not vector or len(vector) != self.vector_dimensions:
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            generated = await self.llm.get_embeddings(missing_texts, encoding_type=encoding_type)
            if len(generated) != len(missing_texts) or any(
                not vector or len(vector) != self.vector_dimensions for vector in generated
            ):
                raise ValueError(
                    f"Sparse-text embedding failure: expected {len(missing_texts)} vectors "
                    f"with dimension {self.vector_dimensions}"
                )
            new_values = dict(zip(missing_keys, generated))
            await asyncio.to_thread(self._write_embedding_cache, new_values)
            cached.update(new_values)

        for key, output_index in zip(keys, positions):
            result[output_index] = cached[key]
        return result
