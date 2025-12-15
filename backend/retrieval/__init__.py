"""DSPy Retrieval package.
Provides retrievers for RAG pipelines.
"""

from .base import BaseRetriever, RetrievalResult
from .chroma_retriever import ChromaRetriever
from .faiss_retriever import FAISSRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "FAISSRetriever",
    "ChromaRetriever",
]
