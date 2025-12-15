"""Base Retriever - Abstract base class for RAG retrievers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    passages: list[str]
    scores: list[float] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.passages)

    def __iter__(self):
        return iter(self.passages)

    def top_k(self, k: int) -> "RetrievalResult":
        """Get top k results."""
        return RetrievalResult(
            passages=self.passages[:k],
            scores=self.scores[:k] if self.scores else [],
            metadata=self.metadata[:k] if self.metadata else [],
        )


class BaseRetriever(ABC):
    """Abstract base class for document retrievers.

    Retrievers are used in RAG pipelines to find relevant context.
    """

    name: str = "base_retriever"

    @abstractmethod
    def index(self, documents: list[str], metadata: list[dict] | None = None) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of document texts to index
            metadata: Optional metadata for each document
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> RetrievalResult:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            RetrievalResult with passages and scores
        """
        pass

    def __call__(self, query: str, k: int = 5) -> RetrievalResult:
        """Call search."""
        return self.search(query, k)

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        pass

    @property
    @abstractmethod
    def num_documents(self) -> int:
        """Number of indexed documents."""
        pass

    def to_dspy_rm(self):
        """Convert to DSPy retrieval model format.

        Returns a callable that DSPy can use as a retrieval model.
        """

        def retrieve(query: str, k: int = 5) -> list[str]:
            result = self.search(query, k)
            return result.passages

        return retrieve
