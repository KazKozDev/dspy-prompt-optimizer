from __future__ import annotations

from utils.token_counter import TokenCounter


def test_token_counter_fallback_counts_words() -> None:
    counter = TokenCounter(model="unknown-model")
    assert counter.count("hello world") == 2


def test_token_counter_truncate_fallback() -> None:
    counter = TokenCounter(model="unknown-model")
    assert counter.truncate("a b c d", max_tokens=2) == "a b"
