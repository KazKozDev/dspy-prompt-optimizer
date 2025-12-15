"""
Tests for retrieval module.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.base import BaseRetriever, RetrievalResult


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""
    
    def test_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            passages=["doc1", "doc2", "doc3"],
            scores=[0.9, 0.8, 0.7],
            metadata=[{"id": 1}, {"id": 2}, {"id": 3}]
        )
        
        assert len(result) == 3
        assert result.passages[0] == "doc1"
        assert result.scores[0] == 0.9
    
    def test_iteration(self):
        """Test iterating over results."""
        result = RetrievalResult(passages=["a", "b", "c"])
        
        passages = list(result)
        
        assert passages == ["a", "b", "c"]
    
    def test_top_k(self):
        """Test getting top k results."""
        result = RetrievalResult(
            passages=["doc1", "doc2", "doc3", "doc4"],
            scores=[0.9, 0.8, 0.7, 0.6],
            metadata=[{"id": i} for i in range(4)]
        )
        
        top2 = result.top_k(2)
        
        assert len(top2) == 2
        assert top2.passages == ["doc1", "doc2"]
        assert top2.scores == [0.9, 0.8]
    
    def test_empty_result(self):
        """Test empty result."""
        result = RetrievalResult(passages=[])
        
        assert len(result) == 0
        assert list(result) == []


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""
    
    name = "mock"
    
    def __init__(self):
        self._documents = []
        self._metadata = []
    
    def index(self, documents, metadata=None):
        self._documents.extend(documents)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{}] * len(documents))
    
    def search(self, query, k=5):
        # Simple substring matching for testing
        results = []
        scores = []
        meta = []
        
        for i, doc in enumerate(self._documents):
            if query.lower() in doc.lower():
                results.append(doc)
                scores.append(1.0 - i * 0.1)
                meta.append(self._metadata[i] if i < len(self._metadata) else {})
        
        return RetrievalResult(
            passages=results[:k],
            scores=scores[:k],
            metadata=meta[:k]
        )
    
    def clear(self):
        self._documents = []
        self._metadata = []
    
    @property
    def num_documents(self):
        return len(self._documents)


class TestBaseRetriever:
    """Tests for BaseRetriever interface."""
    
    def test_index_and_search(self):
        """Test basic index and search."""
        retriever = MockRetriever()
        
        retriever.index([
            "The quick brown fox",
            "The lazy dog",
            "A quick rabbit"
        ])
        
        results = retriever.search("quick")
        
        assert len(results) == 2
        assert "quick" in results.passages[0].lower()
    
    def test_search_with_k(self):
        """Test search with k limit."""
        retriever = MockRetriever()
        
        retriever.index([
            "apple pie",
            "apple juice",
            "apple sauce",
            "apple cider"
        ])
        
        results = retriever.search("apple", k=2)
        
        assert len(results) == 2
    
    def test_clear(self):
        """Test clearing index."""
        retriever = MockRetriever()
        retriever.index(["doc1", "doc2"])
        
        retriever.clear()
        
        assert retriever.num_documents == 0
    
    def test_callable_interface(self):
        """Test __call__ interface."""
        retriever = MockRetriever()
        retriever.index(["test document"])
        
        results = retriever("test")
        
        assert len(results) == 1
    
    def test_to_dspy_rm(self):
        """Test conversion to DSPy retrieval model."""
        retriever = MockRetriever()
        retriever.index(["hello world", "goodbye world"])
        
        rm = retriever.to_dspy_rm()
        passages = rm("hello")
        
        assert isinstance(passages, list)
        assert len(passages) == 1
        assert "hello" in passages[0]
    
    def test_index_with_metadata(self):
        """Test indexing with metadata."""
        retriever = MockRetriever()
        
        retriever.index(
            ["doc1", "doc2"],
            metadata=[{"source": "a"}, {"source": "b"}]
        )
        
        assert retriever.num_documents == 2


# Skip FAISS and Chroma tests if dependencies not installed
class TestFAISSRetriever:
    """Tests for FAISSRetriever (requires faiss and sentence-transformers)."""
    
    @pytest.fixture
    def faiss_retriever(self):
        """Create FAISS retriever if available."""
        try:
            from retrieval import FAISSRetriever
            return FAISSRetriever(model_name="all-MiniLM-L6-v2")
        except ImportError:
            pytest.skip("FAISS or sentence-transformers not installed")
    
    def test_index_and_search(self, faiss_retriever):
        """Test FAISS index and search."""
        faiss_retriever.index([
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing handles text"
        ])
        
        results = faiss_retriever.search("neural networks", k=2)
        
        assert len(results) <= 2
        assert len(results.scores) == len(results.passages)
    
    def test_clear(self, faiss_retriever):
        """Test clearing FAISS index."""
        faiss_retriever.index(["test"])
        faiss_retriever.clear()
        
        assert faiss_retriever.num_documents == 0


class TestChromaRetriever:
    """Tests for ChromaRetriever (requires chromadb)."""
    
    @pytest.fixture
    def chroma_retriever(self):
        """Create Chroma retriever if available."""
        try:
            from retrieval import ChromaRetriever
            retriever = ChromaRetriever(
                collection_name="test_collection",
                persist_directory=None  # In-memory
            )
            yield retriever
            retriever.clear()
        except ImportError:
            pytest.skip("ChromaDB not installed")
    
    def test_index_and_search(self, chroma_retriever):
        """Test Chroma index and search."""
        chroma_retriever.index([
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is memory safe"
        ])
        
        results = chroma_retriever.search("programming", k=2)
        
        assert len(results) <= 2
    
    def test_num_documents(self, chroma_retriever):
        """Test document count."""
        chroma_retriever.index(["doc1", "doc2", "doc3"])
        
        assert chroma_retriever.num_documents == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
