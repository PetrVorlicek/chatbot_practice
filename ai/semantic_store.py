from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL_NAME = "Qwen-3-30B"


@dataclass(slots=True)
class SemanticChunk:
    source_file: str
    chunk_index: int
    text: str
    vector: np.ndarray


@dataclass(slots=True)
class ChunkMatch:
    source_file: str
    chunk_index: int
    text: str
    score: float


class SemanticStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_chunks_source_chunk
                ON semantic_chunks(source_file, chunk_index)
                """
            )
            conn.commit()

    def reset(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS semantic_chunks")
            conn.commit()
        self.ensure_schema()

    def insert_many(self, chunks: Iterable[SemanticChunk]) -> None:
        payload = []
        for chunk in chunks:
            vector32 = np.asarray(chunk.vector, dtype=np.float32).reshape(-1)
            payload.append(
                (
                    chunk.source_file,
                    chunk.chunk_index,
                    chunk.text,
                    vector32.tobytes(),
                    int(vector32.shape[0]),
                )
            )

        if not payload:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO semantic_chunks (
                    source_file,
                    chunk_index,
                    text,
                    vector_blob,
                    vector_dim
                ) VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()

    def load_all(self) -> list[SemanticChunk]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT source_file, chunk_index, text, vector_blob
                FROM semantic_chunks
                ORDER BY source_file, chunk_index
                """
            ).fetchall()

        return [
            SemanticChunk(
                source_file=source_file,
                chunk_index=int(chunk_index),
                text=text,
                vector=np.frombuffer(vector_blob, dtype=np.float32),
            )
            for source_file, chunk_index, text, vector_blob in rows
        ]

    def row_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM semantic_chunks").fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        client: OpenAI,
        query: str,
        *,
        top_k: int = 3,
        max_chars_per_chunk: int = 850,
    ) -> list[ChunkMatch]:
        chunks = self.load_all()
        if not chunks:
            return []

        query_vector = embed_text(client, query)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return []

        candidates = [chunk for chunk in chunks if chunk.vector.shape[0] == query_vector.shape[0]]
        if not candidates:
            return []

        matrix = np.vstack([chunk.vector for chunk in candidates]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 0
        if not np.any(valid):
            return []

        scores = np.full(matrix.shape[0], -1.0, dtype=np.float32)
        scores[valid] = (matrix[valid] @ query_vector) / (norms[valid] * query_norm)
        best_indices = np.argsort(-scores)[:top_k]

        results: list[ChunkMatch] = []
        for idx in best_indices:
            score = float(scores[int(idx)])
            if score < 0:
                continue
            chunk = candidates[int(idx)]
            results.append(
                ChunkMatch(
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    text=truncate_text(chunk.text, max_chars_per_chunk),
                    score=score,
                )
            )
        return results


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_embedding(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        if "data" in payload and payload["data"]:
            first = payload["data"][0]
            if isinstance(first, dict) and "embedding" in first:
                return first["embedding"]
        if "embedding" in payload and isinstance(payload["embedding"], list):
            return payload["embedding"]

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, (float, int)):
            return payload
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]

    raise ValueError(f"Unexpected embedding payload type: {type(payload)}")


def embed_text(client: OpenAI, text: str) -> np.ndarray:
    raw_response = client.embeddings.with_raw_response.create(
        model=EMBEDDING_MODEL_NAME,
        input=text,
    )
    payload = json.loads(raw_response.text)
    return np.asarray(extract_embedding(payload), dtype=np.float32).reshape(-1)
