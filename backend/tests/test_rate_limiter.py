from __future__ import annotations

import time

import pytest

from utils.rate_limiter import RateLimiter


def test_rate_limiter_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        RateLimiter(requests_per_minute=0)


def test_rate_limiter_acquire_tracks_usage() -> None:
    limiter = RateLimiter(requests_per_minute=1000)
    usage_before = limiter.get_current_usage().requests_in_window

    limiter.acquire()
    usage_after = limiter.get_current_usage().requests_in_window

    assert usage_after == usage_before + 1


def test_rate_limiter_window_purges() -> None:
    limiter = RateLimiter(requests_per_minute=1000)
    limiter.acquire()

    limiter.window_seconds = 0.001
    time.sleep(0.01)

    assert limiter.get_current_usage().requests_in_window == 0
