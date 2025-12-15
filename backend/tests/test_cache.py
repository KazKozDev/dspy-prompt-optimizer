from __future__ import annotations

from pathlib import Path

from utils.cache import CacheKey, FileCache


def test_file_cache_set_get(tmp_path: Path) -> None:
    cache = FileCache(cache_dir=tmp_path, ttl_seconds=60)
    key = CacheKey(model="m", prompt="p", extra="e")

    assert cache.get(key) is None
    cache.set(key, "value")
    assert cache.get(key) == "value"


def test_file_cache_ttl_expiry(tmp_path: Path) -> None:
    cache = FileCache(cache_dir=tmp_path, ttl_seconds=0)
    key = CacheKey(model="m", prompt="p")

    cache.set(key, "value")
    assert cache.get(key) is None
