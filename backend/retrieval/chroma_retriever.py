"""
Chroma Retriever - Vector database using ChromaDB.
"""

import uuid
from typing import Any, Dict, List, Optional

from .base import BaseRetriever, RetrievalResult


class ChromaRetriever(BaseRetriever):
    """
    ChromaDB-based retriever.
    
    Easy to use, persistent storage, good for development.
    """
    
    name = "chroma"
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize Chroma retriever.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage (None for in-memory)
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        self._client = None
        self._collection = None
        self._embedding_function = None
    
    def _get_client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.persist_directory:
                    self._client = chromadb.PersistentClient(
                        path=self.persist_directory
                    )
                else:
                    self._client = chromadb.Client()
                    
            except ImportError:
                raise ImportError(
                    "chromadb required for ChromaRetriever. "
                    "Install with: pip install chromadb"
                )
        return self._client
    
    def _get_embedding_function(self):
        """Get embedding function for ChromaDB."""
        if self._embedding_function is None:
            try:
                from chromadb.utils import embedding_functions
                
                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedding_function
    
    def _get_collection(self):
        """Get or create collection."""
        if self._collection is None:
            client = self._get_client()
            embedding_function = self._get_embedding_function()
            
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def index(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not documents:
            return
        
        collection = self._get_collection()
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{"index": i} for i in range(len(documents))]
        else:
            # Ensure metadata is list of dicts with string values
            metadata = [
                {k: str(v) for k, v in (m or {}).items()}
                for m in metadata
            ]
            # Add index if not present
            for i, m in enumerate(metadata):
                if "index" not in m:
                    m["index"] = str(i)
        
        # Add to collection
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadata
        )
    
    def search(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            RetrievalResult with passages and scores
        """
        collection = self._get_collection()
        
        if collection.count() == 0:
            return RetrievalResult(passages=[], scores=[], metadata=[])
        
        k = min(k, collection.count())
        
        results = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )
        
        passages = results["documents"][0] if results["documents"] else []
        
        # Convert distances to similarity scores (Chroma uses L2 by default)
        distances = results["distances"][0] if results["distances"] else []
        scores = [1.0 / (1.0 + d) for d in distances]
        
        metadata = results["metadatas"][0] if results["metadatas"] else []
        
        return RetrievalResult(passages=passages, scores=scores, metadata=metadata)
    
    def clear(self) -> None:
        """Clear the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
    
    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        collection = self._get_collection()
        collection.delete(ids=ids)
    
    def update(self, ids: List[str], documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Update existing documents."""
        collection = self._get_collection()
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadata
        )
