from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

EMBEDDING_MODEL_NAME = "Qwen-3-30B"


@dataclass(slots=True)
class SemanticChunk:
    source_file: str
    chunk_index: int
    text: str
    vector: np.ndarray


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
