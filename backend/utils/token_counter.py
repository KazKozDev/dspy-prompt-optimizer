from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCounter:
    """Token counting utility with a best-effort tokenizer."""

    model: str

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit max_tokens."""
        if max_tokens <= 0:
            return ""

        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception:
            words = text.split()
            return " ".join(words[:max_tokens])

    def split_by_tokens(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks by tokens."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not text:
            return []

        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
            tokens = enc.encode(text)
            chunks = [
                tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)
            ]
            return [enc.decode(c) for c in chunks]
        except Exception:
            words = text.split()
            return [
                " ".join(words[i : i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]
