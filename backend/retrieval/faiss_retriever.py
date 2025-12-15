"""FAISS Retriever - Vector similarity search using FAISS and sentence-transformers."""

import numpy as np

from utils.settings import get_settings

from .base import BaseRetriever, RetrievalResult


class FAISSRetriever(BaseRetriever):
    """FAISS-based retriever using sentence-transformers for embeddings.

    Fast and efficient for medium-sized document collections.
    """

    name = "faiss"

    def __init__(
        self,
        model_name: str | None = None,
        index_type: str = "flat",  # "flat" or "ivf"
    ):
        """Initialize FAISS retriever.

        Args:
            model_name: Sentence transformer model for embeddings
            index_type: FAISS index type ("flat" for exact, "ivf" for approximate)
        """
        self.model_name = model_name or get_settings().model_defaults.semantic_model
        self.index_type = index_type

        self._model = None
        self._index = None
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._dimension: int | None = None

    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for FAISSRetriever. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def _get_faiss(self):
        """Import FAISS."""
        try:
            import faiss

            return faiss
        except ImportError:
            raise ImportError(
                "faiss required for FAISSRetriever. "
                "Install with: pip install faiss-cpu"
            )

    def _create_index(self, dimension: int):
        """Create FAISS index."""
        faiss = self._get_faiss()

        if self.index_type == "ivf":
            # IVF index for larger collections
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, max(1, len(self._documents) // 10))
            self._index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            # Flat index for exact search
            self._index = faiss.IndexFlatL2(dimension)

        self._dimension = dimension

    def index(self, documents: list[str], metadata: list[dict] | None = None) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not documents:
            return

        model = self._get_model()

        # Generate embeddings
        embeddings = model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        # Create index if needed
        if self._index is None:
            self._create_index(embeddings.shape[1])

        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(embeddings)

        # Add to index
        self._index.add(embeddings)

        # Store documents and metadata
        self._documents.extend(documents)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{}] * len(documents))

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            RetrievalResult with passages and scores
        """
        if self._index is None or len(self._documents) == 0:
            return RetrievalResult(passages=[], scores=[], metadata=[])

        model = self._get_model()

        # Encode query
        query_embedding = model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")

        # Search
        k = min(k, len(self._documents))
        distances, indices = self._index.search(query_embedding, k)

        # Collect results
        passages = []
        scores = []
        metadata = []

        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._documents):
                passages.append(self._documents[idx])
                # Convert L2 distance to similarity score (higher is better)
                scores.append(1.0 / (1.0 + distances[0][i]))
                metadata.append(
                    self._metadata[idx] if idx < len(self._metadata) else {}
                )

        return RetrievalResult(passages=passages, scores=scores, metadata=metadata)

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._documents = []
        self._metadata = []
        self._dimension = None

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def save(self, path: str) -> None:
        """Save index to disk."""
        import json
        import os

        faiss = self._get_faiss()

        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        if self._index is not None:
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))

        # Save documents and metadata
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(
                {
                    "documents": self._documents,
                    "metadata": self._metadata,
                    "model_name": self.model_name,
                    "dimension": self._dimension,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load index from disk."""
        import json
        import os

        faiss = self._get_faiss()

        # Load FAISS index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self._index = faiss.read_index(index_path)

        # Load documents and metadata
        docs_path = os.path.join(path, "documents.json")
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                data = json.load(f)
                self._documents = data["documents"]
                self._metadata = data["metadata"]
                self._dimension = data.get("dimension")
