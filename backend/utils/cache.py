from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    """Cache key components."""

    model: str
    prompt: str
    extra: str = ""


class FileCache:
    """File-based cache for prompt/response pairs."""

    def __init__(self, cache_dir: Path, ttl_seconds: int) -> None:
        """Create a file cache.

        Args:
            cache_dir: Directory where cache files are stored.
            ttl_seconds: Time-to-live for cache entries.
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash(self, key: CacheKey) -> str:
        raw = f"{key.model}\n{key.extra}\n{key.prompt}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _path(self, key: CacheKey) -> Path:
        return self.cache_dir / f"{self._hash(key)}.json"

    def get(self, key: CacheKey) -> str | None:
        """Get a cached value by key.

        Args:
            key: Cache lookup key.

        Returns:
            Cached value or None if missing/expired.
        """
        path = self._path(key)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        created_at = float(payload.get("created_at", 0.0))
        if (time.time() - created_at) > self.ttl_seconds:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            return None

        value = payload.get("value")
        return value if isinstance(value, str) else None

    def set(self, key: CacheKey, value: str) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to store.
        """
        payload: dict[str, Any] = {"created_at": time.time(), "value": value}
        self._path(key).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def clear(self) -> None:
        """Delete all cache entries."""
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass
