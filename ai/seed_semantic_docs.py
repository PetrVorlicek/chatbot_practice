from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI

from ai.semantic_store import EMBEDDING_MODEL_NAME, SemanticChunk, SemanticStore
from settings import settings

CHUNK_SIZE = 750  # We have small context window. One chunk should be ~200 tokens.
CHUNK_OVERLAP = 120
MAX_WORKERS = 2  # We have 2 models now!


@dataclass(slots=True)
class SeedChunk:
    source_file: str
    chunk_index: int
    text: str


def build_embedding_client() -> OpenAI:
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.api_url,
        default_headers={"X-API-Key": settings.api_key},
    )


def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def load_seed_chunks(documents_dir: Path) -> list[SeedChunk]:
    all_chunks: list[SeedChunk] = []
    for document in sorted(documents_dir.glob("*.txt")):
        raw_text = document.read_text(encoding="utf-8")
        normalized = normalize_text(raw_text)
        chunks = chunk_text(normalized)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                SeedChunk(source_file=document.name, chunk_index=idx, text=chunk)
            )
    return all_chunks


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


def embed_one(client: OpenAI, text: str) -> list[float]:
    raw_response = client.embeddings.with_raw_response.create(
        model=EMBEDDING_MODEL_NAME,
        input=text,
    )
    data = json.loads(raw_response.text)
    return extract_embedding(data)


def embed_chunk(
    client: OpenAI,
    chunk: SeedChunk,
    *,
    idx: int,
    total: int,
) -> list[float]:
    print(
        f"[seed] embedding {idx}/{total} start "
        f"source={chunk.source_file} chunk={chunk.chunk_index} chars={len(chunk.text)}",
        flush=True,
    )
    started = perf_counter()
    embedding = embed_one(client, chunk.text)
    elapsed = perf_counter() - started
    print(
        f"[seed] embedding {idx}/{total} done "
        f"source={chunk.source_file} chunk={chunk.chunk_index} seconds={elapsed:.2f}",
        flush=True,
    )
    return embedding


def seed_documents() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    documents_dir = root_dir / "documents"
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    seed_chunks = load_seed_chunks(documents_dir)
    if not seed_chunks:
        raise RuntimeError("No chunks produced from documents.")
    print(f"[seed] prepared chunks={len(seed_chunks)}", flush=True)

    client = build_embedding_client()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        indexed_chunks = list(enumerate(seed_chunks, start=1))
        embeddings = list(
            executor.map(
                lambda item: embed_chunk(
                    client,
                    item[1],
                    idx=item[0],
                    total=len(indexed_chunks),
                ),
                indexed_chunks,
            )
        )

    store = SemanticStore(settings.semantic_db_path)
    store.reset()

    rows = [
        SemanticChunk(
            source_file=chunk.source_file,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            vector=embedding,
        )
        for chunk, embedding in zip(seed_chunks, embeddings, strict=True)
    ]
    store.insert_many(rows)

    print(
        "Seed complete:",
        f"chunks={len(rows)}",
        f"db={store.db_path}",
        f"model={EMBEDDING_MODEL_NAME}",
    )


if __name__ == "__main__":
    seed_documents()
