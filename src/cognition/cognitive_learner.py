import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from src.utils.logger import logger
from src.utils.wiki_extractor import LEARNING_LEVELS


class MemoryStore(Protocol):
    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...


class SQLiteCognitiveMemory:
    """Small persistent key-value store for locally consolidated interactions."""

    def __init__(self, database_path: str | Path = "data/cognitive_memory.sqlite3") -> None:
        self._uri = str(database_path) == ":memory:"
        if self._uri:
            self.database_path = f"file:cognitive_{uuid4().hex}?mode=memory&cache=shared"
            self._anchor_connection = sqlite3.connect(self.database_path, uri=True)
        else:
            self.database_path = str(database_path)
            self._anchor_connection = None
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cognitive_memory (
                    memory_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expires_at REAL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, uri=self._uri)

    def close(self) -> None:
        """Close the anchor connection that keeps a shared in-memory store alive."""
        if self._anchor_connection is not None:
            self._anchor_connection.close()
            self._anchor_connection = None

    def get(self, key: str) -> Any | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT payload, expires_at FROM cognitive_memory WHERE memory_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            payload, expires_at = row
            if expires_at is not None and expires_at <= time.time():
                connection.execute(
                    "DELETE FROM cognitive_memory WHERE memory_key = ?",
                    (key,),
                )
                return None
        return json.loads(payload)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not key:
            raise ValueError("memory key must not be empty.")
        if ttl is not None and ttl <= 0:
            raise ValueError("memory ttl must be greater than zero.")
        expires_at = time.time() + ttl if ttl is not None else None
        payload = json.dumps(value, ensure_ascii=False)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO cognitive_memory (memory_key, payload, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(memory_key) DO UPDATE SET
                    payload = excluded.payload,
                    expires_at = excluded.expires_at
                """,
                (key, payload, expires_at),
            )


class CognitiveLearnerSkill:
    """Heuristic perception, context-gap assessment, and durable local recall."""

    DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self.memory: MemoryStore = (
            memory_store if memory_store is not None else SQLiteCognitiveMemory()
        )
        logger.info("Cognitive learner initialized.")

    def perceive(self, user_query: str) -> dict[str, Any]:
        """Estimate a curriculum level from simple, inspectable query heuristics."""
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("user_query must not be empty.")

        normalized_query = user_query.casefold()
        word_count = len(user_query.split())
        complexity = "basico"
        if word_count > 15 or any(
            term in normalized_query
            for term in ("teoria", "mecanismo", "avançado", "avancado", "como funciona")
        ):
            complexity = "intermediario"
        if word_count > 30 or any(
            term in normalized_query
            for term in ("arquitetura", "otimização", "otimizacao", "matemática", "matematica", "exceção", "excecao")
        ):
            complexity = "avancado"

        return {
            "query": user_query,
            "detected_complexity": complexity,
            "timestamp": time.time(),
        }

    def evaluate_cognition(
        self,
        perception: dict[str, Any],
        retrieved_context: str | None = None,
        prior_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Estimate evidence sufficiency; this heuristic is not a factuality guarantee."""
        if not isinstance(perception, dict) or not isinstance(
            perception.get("query"), str
        ):
            raise ValueError("perception must contain the original query.")

        has_gap = not retrieved_context or len(retrieved_context.strip()) < 50
        return {
            "has_knowledge_gap": has_gap,
            "confidence": 0.4 if has_gap else 0.9,
            "recommended_action": "search_wikipedia" if has_gap else "answer_with_sources",
            "prior_memory_found": prior_memory is not None,
        }

    def recall(self, query: str, level: str | None = None) -> dict[str, Any] | None:
        """Return a previously stored answer for the same normalized question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")
        levels = (level,) if level is not None else LEARNING_LEVELS
        if any(candidate not in LEARNING_LEVELS for candidate in levels):
            raise ValueError(f"level must be one of: {', '.join(LEARNING_LEVELS)}.")

        digest = hashlib.sha256(query.strip().casefold().encode("utf-8")).hexdigest()
        for candidate in levels:
            memory = self.memory.get(f"learned_{candidate}_{digest}")
            if memory is not None:
                return memory
        return None

    def learn_and_store(
        self,
        query: str,
        response: str,
        level: str,
        sources: list[str] | None = None,
        assessment: dict[str, Any] | None = None,
    ) -> None:
        """Persist a sourced interaction for later local recall."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must not be empty.")
        if level not in LEARNING_LEVELS:
            raise ValueError(f"level must be one of: {', '.join(LEARNING_LEVELS)}.")
        if sources is not None and any(not isinstance(source, str) or not source for source in sources):
            raise ValueError("sources must contain non-empty strings.")

        digest = hashlib.sha256(query.strip().casefold().encode("utf-8")).hexdigest()
        payload = {
            "query": query.strip(),
            "synthesized_answer": response,
            "level": level,
            "sources": sources or [],
            "assessment": assessment or {},
            "learned_at": time.time(),
        }
        self.memory.set(
            f"learned_{level}_{digest}",
            payload,
            ttl=self.DEFAULT_TTL_SECONDS,
        )
        logger.info("Consolidated a sourced interaction at level %s.", level)
