import hashlib
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from src.utils.logger import logger


class EmbeddingProvider(ABC):
    """Interface for text embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a vector representation of text."""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free lexical embeddings for local development and tests."""

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        features = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider for semantic retrieval."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.model = model

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from openai import OpenAI

        response = OpenAI(api_key=self.api_key).embeddings.create(
            model=self.model,
            input=texts,
        )
        vectors: list[list[float] | None] = [None] * len(texts)
        for item in response.data:
            vectors[item.index] = item.embedding

        if any(vector is None for vector in vectors):
            raise RuntimeError("OpenAI returned an incomplete set of embeddings.")
        return [vector for vector in vectors if vector is not None]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embeddings with separate retrieval task types for documents and queries."""

    MAX_BATCH_SIZE = 100
    MIN_REQUEST_INTERVAL_SECONDS = 1.0
    MAX_RATE_LIMIT_RETRIES = 4

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
        self.model = model
        self._request_lock = threading.Lock()
        self._last_request_at: float | None = None

    def embed(self, text: str) -> list[float]:
        return self._embed_contents([text], task_type="RETRIEVAL_QUERY")[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self._embed_contents(texts, task_type="RETRIEVAL_DOCUMENT")

    def _embed_contents(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start : start + self.MAX_BATCH_SIZE]
            response = self._embed_batch(client, types, batch, task_type)
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch) or any(not item.values for item in embeddings):
                raise RuntimeError("Gemini returned an incomplete set of embeddings.")
            vectors.extend(list(item.values) for item in embeddings)
        return vectors

    def _embed_batch(
        self,
        client: Any,
        types: Any,
        batch: list[str],
        task_type: str,
    ) -> Any:
        for attempt in range(self.MAX_RATE_LIMIT_RETRIES + 1):
            self._wait_for_request_slot()
            try:
                return client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            except Exception as error:
                if not self._is_rate_limit_error(error) or attempt == self.MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = self._retry_delay(error, attempt)
                logger.warning(
                    "Gemini embedding quota reached for model %s; retrying in %.1f seconds.",
                    self.model,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("Gemini embedding retry loop ended unexpectedly.")

    def _wait_for_request_slot(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                wait_seconds = (
                    self._last_request_at + self.MIN_REQUEST_INTERVAL_SECONDS - now
                )
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        return status_code == 429 or str(status_code) == "429"

    @staticmethod
    def _retry_delay(error: Exception, attempt: int) -> float:
        error_text = str(error)
        retry_match = re.search(
            r"retry in\s+([\d.]+)\s*s|retryDelay['\"]?\s*:\s*['\"]([\d.]+)s",
            error_text,
            flags=re.IGNORECASE,
        )
        if retry_match:
            retry_after = next(
                (float(value) for value in retry_match.groups() if value is not None),
                0.0,
            )
            return max(1.0, retry_after)
        return min(60.0, 2.0 ** (attempt + 1))


def create_embedding_provider(config: dict[str, Any]) -> EmbeddingProvider:
    """Build the configured embedding provider."""
    provider = config.get("embedding_provider", "hash")
    if provider == "hash":
        return HashEmbeddingProvider(dimensions=config.get("dimensions", 512))
    if provider == "openai":
        return OpenAIEmbeddingProvider(model=config.get("embedding_model", "text-embedding-3-small"))
    if provider == "gemini":
        return GeminiEmbeddingProvider(model=config.get("embedding_model", "gemini-embedding-001"))
    raise ValueError(
        f"Unsupported embedding provider {provider!r}; choose 'hash', 'openai', or 'gemini'."
    )
