from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class Usage:
    """Usage statistics."""

    requests_in_window: int


class RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, requests_per_minute: int) -> None:
        """Create a rate limiter.

        Args:
            requests_per_minute: Allowed number of events per rolling minute.
        """
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self._events: deque[float] = deque()

    def _purge(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def get_current_usage(self) -> Usage:
        """Return current usage statistics."""
        now = time.time()
        self._purge(now)
        return Usage(requests_in_window=len(self._events))

    def acquire(self) -> None:
        """Block until a token is available and record an event."""
        while True:
            now = time.time()
            self._purge(now)
            if len(self._events) < self.requests_per_minute:
                self._events.append(now)
                return

            earliest = self._events[0]
            sleep_for = max(0.0, (earliest + self.window_seconds) - now)
            time.sleep(sleep_for)

    async def acquire_async(self) -> None:
        """Async version of acquire()."""
        while True:
            now = time.time()
            self._purge(now)
            if len(self._events) < self.requests_per_minute:
                self._events.append(now)
                return

            earliest = self._events[0]
            sleep_for = max(0.0, (earliest + self.window_seconds) - now)
            await asyncio.sleep(sleep_for)
