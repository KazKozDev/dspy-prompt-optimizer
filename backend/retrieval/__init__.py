"""
DSPy Retrieval package.
Provides retrievers for RAG pipelines.
"""

from .base import BaseRetriever, RetrievalResult
from .faiss_retriever import FAISSRetriever
from .chroma_retriever import ChromaRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "FAISSRetriever",
    "ChromaRetriever",
]
