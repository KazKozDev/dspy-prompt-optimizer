"""DSPy Metrics package.
Provides evaluation metrics including LLM-as-Judge.
"""

from .base import BaseMetric, MetricResult
from .exact_match import ExactMatchMetric
from .llm_judge import (
    CoherenceJudge,
    CorrectnessJudge,
    FaithfulnessJudge,
    LLMJudgeMetric,
)
from .semantic import SemanticSimilarityMetric
from .token_f1 import TokenF1Metric

__all__ = [
    "BaseMetric",
    "MetricResult",
    "ExactMatchMetric",
    "TokenF1Metric",
    "LLMJudgeMetric",
    "CorrectnessJudge",
    "FaithfulnessJudge",
    "CoherenceJudge",
    "SemanticSimilarityMetric",
]
