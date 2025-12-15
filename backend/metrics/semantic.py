"""
Semantic Similarity Metric - Uses embeddings to compute semantic similarity.
"""

from typing import Any, List, Optional

from .base import BaseMetric


class SemanticSimilarityMetric(BaseMetric):
    """
    Semantic similarity metric using embeddings.
    
    Computes cosine similarity between embeddings of expected and predicted outputs.
    """
    
    name = "semantic_similarity"
    description = "Semantic similarity using embeddings"
    
    def __init__(
        self,
        output_field: str = "result",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize SemanticSimilarityMetric.
        
        Args:
            output_field: Name of the output field to compare
            model_name: Sentence transformer model name
        """
        self.output_field = output_field
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for SemanticSimilarityMetric. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Optional[Any] = None
    ) -> float:
        """
        Compute semantic similarity between expected and predicted.
        
        Args:
            example: Ground truth with expected output
            prediction: Model prediction
            trace: Optional trace (unused)
            
        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        expected = self.get_field_value(example, self.output_field)
        predicted = self.get_field_value(prediction, self.output_field)
        
        if not expected and not predicted:
            return 1.0
        if not expected or not predicted:
            return 0.0
        
        try:
            model = self._get_model()
            
            embeddings = model.encode([expected, predicted])
            
            similarity = self._cosine_similarity(
                embeddings[0].tolist(),
                embeddings[1].tolist()
            )
            
            return max(0.0, similarity)
            
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return 0.0
    
    def compute_batch(
        self,
        expected_texts: List[str],
        predicted_texts: List[str]
    ) -> List[float]:
        """
        Compute semantic similarity for a batch of texts.
        
        More efficient than calling __call__ multiple times.
        """
        if len(expected_texts) != len(predicted_texts):
            raise ValueError("Lists must have same length")
        
        try:
            model = self._get_model()
            
            all_texts = expected_texts + predicted_texts
            embeddings = model.encode(all_texts)
            
            n = len(expected_texts)
            expected_emb = embeddings[:n]
            predicted_emb = embeddings[n:]
            
            similarities = []
            for exp, pred in zip(expected_emb, predicted_emb):
                sim = self._cosine_similarity(exp.tolist(), pred.tolist())
                similarities.append(max(0.0, sim))
            
            return similarities
            
        except Exception as e:
            print(f"Batch semantic similarity error: {e}")
            return [0.0] * len(expected_texts)
