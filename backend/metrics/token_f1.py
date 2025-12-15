"""
Token F1 Metric - Computes token-level F1 score between prediction and expected output.
"""

from typing import Any, Optional, Set

from .base import BaseMetric


class TokenF1Metric(BaseMetric):
    """
    Token-level F1 metric.
    
    Computes precision, recall, and F1 score based on token overlap.
    Useful for tasks where partial matches are valuable.
    """
    
    name = "token_f1"
    description = "Token-level F1 score between prediction and expected output"
    
    def __init__(
        self,
        output_field: str = "result",
        case_sensitive: bool = False
    ):
        """
        Initialize TokenF1Metric.
        
        Args:
            output_field: Name of the output field to compare
            case_sensitive: Whether comparison is case-sensitive
        """
        self.output_field = output_field
        self.case_sensitive = case_sensitive
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into set of tokens."""
        if not self.case_sensitive:
            text = text.lower()
        return set(text.split())
    
    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Optional[Any] = None
    ) -> float:
        """
        Compute token F1 score.
        
        Args:
            example: Ground truth with expected output
            prediction: Model prediction
            trace: Optional trace (unused)
            
        Returns:
            F1 score between 0.0 and 1.0
        """
        expected = self.get_field_value(example, self.output_field)
        predicted = self.get_field_value(prediction, self.output_field)
        
        expected_tokens = self._tokenize(expected)
        predicted_tokens = self._tokenize(predicted)
        
        if not expected_tokens and not predicted_tokens:
            return 1.0
        if not expected_tokens or not predicted_tokens:
            return 0.0
        
        intersection = len(expected_tokens & predicted_tokens)
        
        if intersection == 0:
            return 0.0
        
        precision = intersection / len(predicted_tokens)
        recall = intersection / len(expected_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def compute_detailed(
        self,
        example: Any,
        prediction: Any
    ) -> dict:
        """
        Compute detailed token F1 metrics.
        
        Returns dict with precision, recall, f1, and token details.
        """
        expected = self.get_field_value(example, self.output_field)
        predicted = self.get_field_value(prediction, self.output_field)
        
        expected_tokens = self._tokenize(expected)
        predicted_tokens = self._tokenize(predicted)
        
        if not expected_tokens and not predicted_tokens:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "matched_tokens": [],
                "missed_tokens": [],
                "extra_tokens": [],
            }
        
        intersection = expected_tokens & predicted_tokens
        missed = expected_tokens - predicted_tokens
        extra = predicted_tokens - expected_tokens
        
        precision = len(intersection) / len(predicted_tokens) if predicted_tokens else 0.0
        recall = len(intersection) / len(expected_tokens) if expected_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_tokens": list(intersection),
            "missed_tokens": list(missed),
            "extra_tokens": list(extra),
        }
