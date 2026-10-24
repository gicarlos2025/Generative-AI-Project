import time
from typing import Any


class SimpleCache:
    """In-memory cache with per-entry time-to-live expiration."""

    def __init__(self, default_ttl: int = 3600) -> None:
        if default_ttl < 0:
            raise ValueError("default_ttl must be zero or greater")
        self._cache: dict[str, tuple[Any, float]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        item = self._cache.get(key)
        if item is None:
            return None

        value, expires_at = item
        if time.monotonic() < expires_at:
            return value
        del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = self.default_ttl if ttl is None else ttl
        if effective_ttl < 0:
            raise ValueError("ttl must be zero or greater")
        self._cache[key] = (value, time.monotonic() + effective_ttl)
