import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.rag.embeddings import EmbeddingProvider


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any]


class SQLiteVectorStore:
    """Persistent local vector store with exact cosine-similarity search."""

    def __init__(self, db_path: str | Path, embedding_provider: EmbeddingProvider) -> None:
        self.db_path = Path(db_path)
        self.embedding_provider = embedding_provider
        self._memory_connection = (
            sqlite3.connect(":memory:") if str(self.db_path) == ":memory:" else None
        )
        if self._memory_connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        return sqlite3.connect(str(self.db_path))

    def upsert(
        self,
        records: list[tuple[str, str, str, dict[str, Any]]],
        *,
        replace_source: str | None = None,
    ) -> None:
        """Insert records and optionally replace every existing chunk of a source."""
        if not records and replace_source is None:
            return
        if any(not record_id or not text.strip() for record_id, text, _, _ in records):
            raise ValueError("Each vector record must have a non-empty id and text.")

        existing_vectors: dict[str, tuple[str, str, str, str]] = {}
        record_ids = [record[0] for record in records]
        for offset in range(0, len(record_ids), 500):
            batch_ids = record_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch_ids)
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, text, source, metadata, embedding FROM vectors "
                    f"WHERE id IN ({placeholders})",
                    batch_ids,
                ).fetchall()
            existing_vectors.update(
                (record_id, (text, source, metadata, embedding))
                for record_id, text, source, metadata, embedding in rows
            )

        row_data: list[tuple[str, str, str, str, str] | None] = [None] * len(records)
        changed_indexes = []
        changed_texts = []
        for index, (record_id, text, source, metadata) in enumerate(records):
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            existing = existing_vectors.get(record_id)
            if existing is not None and existing[:3] == (text, source, metadata_json):
                row_data[index] = (
                    record_id,
                    text,
                    source,
                    metadata_json,
                    existing[3],
                )
            else:
                changed_indexes.append(index)
                changed_texts.append(text)

        embeddings = self.embedding_provider.embed_many(changed_texts)
        if len(embeddings) != len(changed_indexes):
            raise ValueError("Embedding provider returned an unexpected number of vectors.")

        for index, embedding in zip(changed_indexes, embeddings):
            record_id, text, source, metadata = records[index]
            if not embedding or not all(math.isfinite(value) for value in embedding):
                raise ValueError(f"Embedding for record {record_id!r} must contain finite values.")
            row_data[index] = (
                record_id,
                text,
                source,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(embedding),
            )

        rows = [row for row in row_data if row is not None]
        if len(rows) != len(records):
            raise RuntimeError("Vector upsert failed to prepare all records.")

        with self._connect() as connection:
            if replace_source is not None:
                connection.execute("DELETE FROM vectors WHERE source = ?", (replace_source,))
            connection.executemany(
                """
                INSERT INTO vectors (id, text, source, metadata, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text = excluded.text,
                    source = excluded.source,
                    metadata = excluded.metadata,
                    embedding = excluded.embedding
                """,
                rows,
            )

    def search(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top matching vectors ranked by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = self.embedding_provider.embed(query)
        query_norm = math.sqrt(sum(value * value for value in query_embedding))
        if query_norm == 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, text, source, metadata, embedding FROM vectors"
            ).fetchall()

        results: list[SearchResult] = []
        for record_id, text, source, metadata_json, embedding_json in rows:
            metadata = json.loads(metadata_json)
            if metadata_filter and any(
                metadata.get(key) != expected for key, expected in metadata_filter.items()
            ):
                continue
            embedding = json.loads(embedding_json)
            if len(embedding) != len(query_embedding):
                raise ValueError(
                    f"Stored embedding for {record_id!r} has dimension {len(embedding)}; "
                    f"query embedding has dimension {len(query_embedding)}."
                )
            vector_norm = math.sqrt(sum(value * value for value in embedding))
            if vector_norm == 0:
                continue
            score = sum(a * b for a, b in zip(query_embedding, embedding)) / (
                query_norm * vector_norm
            )
            results.append(
                SearchResult(
                    id=record_id,
                    text=text,
                    source=source,
                    score=score,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda result: (-result.score, result.id))
        return results[:top_k]

    def count(self) -> int:
        """Return the number of indexed chunks."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(row[0])
