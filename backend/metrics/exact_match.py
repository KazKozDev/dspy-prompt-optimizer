"""
Exact Match Metric - Checks if prediction exactly matches expected output.
"""

from typing import Any, Optional

from .base import BaseMetric


class ExactMatchMetric(BaseMetric):
    """
    Exact match metric.
    
    Returns 1.0 if prediction exactly matches expected output, 0.0 otherwise.
    Supports case-insensitive matching and whitespace normalization.
    """
    
    name = "exact_match"
    description = "Exact string match between prediction and expected output"
    
    def __init__(
        self,
        output_field: str = "result",
        case_sensitive: bool = False,
        normalize_whitespace: bool = True
    ):
        """
        Initialize ExactMatchMetric.
        
        Args:
            output_field: Name of the output field to compare
            case_sensitive: Whether comparison is case-sensitive
            normalize_whitespace: Whether to normalize whitespace
        """
        self.output_field = output_field
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.normalize_whitespace:
            text = " ".join(text.split())
        if not self.case_sensitive:
            text = text.lower()
        return text.strip()
    
    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: Optional[Any] = None
    ) -> float:
        """
        Check if prediction exactly matches expected output.
        
        Args:
            example: Ground truth with expected output
            prediction: Model prediction
            trace: Optional trace (unused)
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        expected = self.get_field_value(example, self.output_field)
        predicted = self.get_field_value(prediction, self.output_field)
        
        expected_norm = self._normalize(expected)
        predicted_norm = self._normalize(predicted)
        
        if not expected_norm and not predicted_norm:
            return 1.0
        
        return 1.0 if expected_norm == predicted_norm else 0.0
