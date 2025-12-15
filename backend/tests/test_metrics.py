"""Tests for metrics module."""

import pytest
from dataclasses import dataclass

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import ExactMatchMetric, TokenF1Metric
from metrics.base import MetricResult


@dataclass
class MockExample:
    """Mock example for testing."""

    text: str = ""
    result: str = ""
    output: str = ""


@dataclass
class MockPrediction:
    """Mock prediction for testing."""

    result: str = ""
    output: str = ""


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        metric = ExactMatchMetric(output_field="result")
        example = MockExample(result="positive")
        pred = MockPrediction(result="positive")

        score = metric(example, pred)
        assert score == 1.0

    def test_exact_match_different(self):
        """Test exact match with different strings."""
        metric = ExactMatchMetric(output_field="result")
        example = MockExample(result="positive")
        pred = MockPrediction(result="negative")

        score = metric(example, pred)
        assert score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching (default)."""
        metric = ExactMatchMetric(output_field="result", case_sensitive=False)
        example = MockExample(result="Positive")
        pred = MockPrediction(result="positive")

        score = metric(example, pred)
        assert score == 1.0

    def test_exact_match_case_sensitive(self):
        """Test case-sensitive matching."""
        metric = ExactMatchMetric(output_field="result", case_sensitive=True)
        example = MockExample(result="Positive")
        pred = MockPrediction(result="positive")

        score = metric(example, pred)
        assert score == 0.0

    def test_exact_match_whitespace_normalization(self):
        """Test whitespace normalization."""
        metric = ExactMatchMetric(output_field="result", normalize_whitespace=True)
        example = MockExample(result="hello  world")
        pred = MockPrediction(result="hello world")

        score = metric(example, pred)
        assert score == 1.0

    def test_exact_match_empty_strings(self):
        """Test with empty strings."""
        metric = ExactMatchMetric(output_field="result")
        example = MockExample(result="")
        pred = MockPrediction(result="")

        score = metric(example, pred)
        assert score == 1.0

    def test_exact_match_with_dict(self):
        """Test with dict inputs."""
        metric = ExactMatchMetric(output_field="result")
        example = {"result": "positive"}
        pred = {"result": "positive"}

        score = metric(example, pred)
        assert score == 1.0


class TestTokenF1Metric:
    """Tests for TokenF1Metric."""

    def test_token_f1_identical(self):
        """Test F1 with identical strings."""
        metric = TokenF1Metric(output_field="result")
        example = MockExample(result="the quick brown fox")
        pred = MockPrediction(result="the quick brown fox")

        score = metric(example, pred)
        assert score == 1.0

    def test_token_f1_partial_overlap(self):
        """Test F1 with partial overlap."""
        metric = TokenF1Metric(output_field="result")
        example = MockExample(result="the quick brown fox")
        pred = MockPrediction(result="the quick red fox")

        score = metric(example, pred)
        # 3 tokens match out of 4 each
        # precision = 3/4, recall = 3/4, F1 = 0.75
        assert 0.7 < score < 0.8

    def test_token_f1_no_overlap(self):
        """Test F1 with no overlap."""
        metric = TokenF1Metric(output_field="result")
        example = MockExample(result="hello world")
        pred = MockPrediction(result="foo bar")

        score = metric(example, pred)
        assert score == 0.0

    def test_token_f1_empty_strings(self):
        """Test F1 with empty strings."""
        metric = TokenF1Metric(output_field="result")
        example = MockExample(result="")
        pred = MockPrediction(result="")

        score = metric(example, pred)
        assert score == 1.0

    def test_token_f1_one_empty(self):
        """Test F1 with one empty string."""
        metric = TokenF1Metric(output_field="result")
        example = MockExample(result="hello world")
        pred = MockPrediction(result="")

        score = metric(example, pred)
        assert score == 0.0

    def test_token_f1_detailed(self):
        """Test detailed F1 computation."""
        metric = TokenF1Metric(output_field="result")
        example = MockExample(result="the quick brown fox")
        pred = MockPrediction(result="the slow brown dog")

        details = metric.compute_detailed(example, pred)

        assert "precision" in details
        assert "recall" in details
        assert "f1" in details
        assert "matched_tokens" in details
        assert set(details["matched_tokens"]) == {"the", "brown"}


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test MetricResult creation."""
        result = MetricResult(score=0.85, name="test_metric", details={"key": "value"})

        assert result.score == 0.85
        assert result.name == "test_metric"
        assert result.details == {"key": "value"}


class TestBaseMetricBatch:
    """Tests for BaseMetric batch evaluation."""

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        metric = ExactMatchMetric(output_field="result")

        examples = [
            MockExample(result="a"),
            MockExample(result="b"),
            MockExample(result="c"),
        ]
        predictions = [
            MockPrediction(result="a"),  # match
            MockPrediction(result="x"),  # no match
            MockPrediction(result="c"),  # match
        ]

        result = metric.evaluate_batch(examples, predictions)

        assert result.score == pytest.approx(2 / 3, rel=0.01)
        assert result.details["num_examples"] == 3
        assert result.details["num_perfect"] == 2
        assert result.details["num_zero"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
