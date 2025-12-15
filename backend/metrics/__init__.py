"""
DSPy Metrics package.
Provides evaluation metrics including LLM-as-Judge.
"""

from .base import BaseMetric, MetricResult
from .exact_match import ExactMatchMetric
from .token_f1 import TokenF1Metric
from .llm_judge import LLMJudgeMetric, CorrectnessJudge, FaithfulnessJudge, CoherenceJudge
from .semantic import SemanticSimilarityMetric

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
