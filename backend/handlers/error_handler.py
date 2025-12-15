from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any


class APIError(Exception):
    """Base exception for API errors."""


class RateLimitError(APIError):
    """Raised when a provider rate limit is hit."""


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy configuration."""

    max_retries: int = 3
    delay_seconds: float = 1.0
    backoff: float = 2.0


def retry_on_error(
    *,
    config: RetryConfig,
    exceptions: tuple[type[BaseException], ...] = (APIError, Exception),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry decorator supporting sync and async callables."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: object, **kwargs: object) -> object:
                attempt = 0
                delay = config.delay_seconds
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions:
                        attempt += 1
                        if attempt > config.max_retries:
                            raise
                        await asyncio.sleep(delay)
                        delay *= config.backoff

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: object, **kwargs: object) -> object:
            attempt = 0
            delay = config.delay_seconds
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    attempt += 1
                    if attempt > config.max_retries:
                        raise
                    time.sleep(delay)
                    delay *= config.backoff

        return sync_wrapper

    return decorator
