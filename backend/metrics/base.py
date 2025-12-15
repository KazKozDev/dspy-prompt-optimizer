"""Base Metric - Abstract base class for all DSPy metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Result of metric evaluation."""

    score: float  # 0.0 to 1.0
    name: str
    details: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    All metrics should implement __call__ to evaluate a prediction against ground truth.
    """

    name: str = "base_metric"
    description: str = "Base metric class"

    @abstractmethod
    def __call__(
        self, example: Any, prediction: Any, trace: Any | None = None
    ) -> float:
        """Evaluate prediction against example.

        Args:
            example: Ground truth example with expected output
            prediction: Model prediction
            trace: Optional trace information from DSPy

        Returns:
            Score between 0.0 and 1.0
        """
        pass

    def evaluate_batch(
        self, examples: list[Any], predictions: list[Any]
    ) -> MetricResult:
        """Evaluate a batch of predictions.

        Args:
            examples: List of ground truth examples
            predictions: List of model predictions

        Returns:
            MetricResult with average score and details
        """
        if len(examples) != len(predictions):
            raise ValueError("Examples and predictions must have same length")

        scores = []
        for example, pred in zip(examples, predictions):
            try:
                score = self(example, pred)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return MetricResult(
            score=avg_score,
            name=self.name,
            details={
                "individual_scores": scores,
                "num_examples": len(examples),
                "num_perfect": sum(1 for s in scores if s == 1.0),
                "num_zero": sum(1 for s in scores if s == 0.0),
            },
        )

    def get_field_value(self, obj: Any, field_name: str) -> str:
        """Extract field value from example or prediction."""
        if hasattr(obj, field_name):
            return str(getattr(obj, field_name, ""))
        if isinstance(obj, dict):
            return str(obj.get(field_name, ""))
        return ""
