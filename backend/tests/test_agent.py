"""Tests for agent module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import (
    AgentConfig,
    PipelineType,
    MetricType,
    OptimizerType,
    TaskType,
    ComplexityLevel,
    TaskAnalysis,
)
from agent.selectors.task_analyzer import TaskAnalyzer
from agent.selectors.metric_selector import MetricSelector
from agent.selectors.optimizer_selector import OptimizerSelector
from agent.selectors.pipeline_selector import PipelineSelector


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()

        assert config.mode == "auto"
        assert config.quality_profile == "BALANCED"

    def test_to_dict(self):
        """Test serialization to dict."""
        config = AgentConfig(
            mode="manual", business_task="Test task", target_model="openai/gpt-5"
        )

        data = config.to_dict()

        assert data["mode"] == "manual"
        assert data["business_task"] == "Test task"
        assert data["target_model"] == "openai/gpt-5"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "mode": "manual",
            "business_task": "Test task",
            "target_model": "openai/gpt-5",
            "task_analysis": {
                "task_type": "classification",
                "domain": "support",
                "complexity": "medium",
                "input_fields": ["text"],
                "output_fields": ["label"],
                "needs_retrieval": False,
                "needs_chain_of_thought": True,
                "needs_tools": False,
                "needs_multi_stage": False,
                "suggested_tools": [],
                "confidence": 0.9,
                "reasoning": "Test",
            },
        }

        config = AgentConfig.from_dict(data)

        assert config.mode == "manual"
        assert config.task_analysis.task_type == TaskType.CLASSIFICATION
        assert config.task_analysis.domain == "support"


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer."""

    def setup_method(self):
        """Setup analyzer for each test."""
        self.analyzer = TaskAnalyzer()

    def test_classification_detection(self):
        """Test classification task detection."""
        analysis = self.analyzer.analyze(
            "Classify customer reviews as positive or negative"
        )

        assert analysis.task_type == TaskType.CLASSIFICATION

    def test_extraction_detection(self):
        """Test extraction task detection."""
        analysis = self.analyzer.analyze("Extract named entities from the text")

        assert analysis.task_type == TaskType.EXTRACTION

    def test_summarization_detection(self):
        """Test summarization task detection."""
        analysis = self.analyzer.analyze("Summarize the following article")

        assert analysis.task_type == TaskType.SUMMARIZATION

    def test_rag_detection(self):
        """Test RAG task detection."""
        analysis = self.analyzer.analyze(
            "Answer questions using retrieval from knowledge base"
        )

        assert analysis.task_type == TaskType.RAG
        assert analysis.needs_retrieval is True

    def test_reasoning_default(self):
        """Test default to reasoning task."""
        analysis = self.analyzer.analyze("Solve this problem")

        assert analysis.task_type == TaskType.REASONING

    def test_domain_detection_legal(self):
        """Test legal domain detection."""
        analysis = self.analyzer.analyze("Analyze legal contracts for compliance")

        assert analysis.domain == "legal"

    def test_domain_detection_finance(self):
        """Test finance domain detection."""
        analysis = self.analyzer.analyze(
            "Analyze financial reports and investment data"
        )

        assert analysis.domain == "finance"

    def test_complexity_high(self):
        """Test high complexity detection."""
        long_task = "This is a very complex multi-step task " * 20
        analysis = self.analyzer.analyze(long_task)

        assert analysis.complexity == ComplexityLevel.HIGH

    def test_complexity_low(self):
        """Test low complexity detection."""
        analysis = self.analyzer.analyze("Simple label task")

        assert analysis.complexity == ComplexityLevel.LOW

    def test_needs_cot_for_reasoning(self):
        """Test CoT needed for reasoning tasks."""
        analysis = self.analyzer.analyze("Solve this math problem step by step")

        assert analysis.needs_chain_of_thought is True

    def test_tool_suggestion(self):
        """Test tool suggestion for calculation tasks."""
        analysis = self.analyzer.analyze("Calculate the average of these numbers")

        assert analysis.needs_tools is True
        assert "calculator" in analysis.suggested_tools


class TestMetricSelector:
    """Tests for MetricSelector."""

    def setup_method(self):
        """Setup selector for each test."""
        self.selector = MetricSelector()

    def test_classification_metric(self):
        """Test metric selection for classification."""
        analysis = TaskAnalysis(task_type=TaskType.CLASSIFICATION)
        config = self.selector.select(analysis)

        assert config.metric_type == MetricType.EXACT_MATCH

    def test_extraction_metric(self):
        """Test metric selection for extraction."""
        analysis = TaskAnalysis(task_type=TaskType.EXTRACTION)
        config = self.selector.select(analysis)

        assert config.metric_type == MetricType.TOKEN_F1

    def test_generation_metric(self):
        """Test metric selection for generation."""
        analysis = TaskAnalysis(task_type=TaskType.GENERATION)
        config = self.selector.select(analysis)

        assert config.metric_type == MetricType.LLM_JUDGE

    def test_fast_cheap_avoids_llm_judge(self):
        """Test FAST_CHEAP profile avoids LLM judge."""
        analysis = TaskAnalysis(task_type=TaskType.GENERATION)
        config = self.selector.select(analysis, quality_profile="FAST_CHEAP")

        assert config.metric_type == MetricType.TOKEN_F1

    def test_high_quality_uses_llm_judge(self):
        """Test HIGH_QUALITY profile uses LLM judge for QA."""
        analysis = TaskAnalysis(task_type=TaskType.QA)
        config = self.selector.select(analysis, quality_profile="HIGH_QUALITY")

        assert config.metric_type == MetricType.LLM_JUDGE


class TestOptimizerSelector:
    """Tests for OptimizerSelector."""

    def setup_method(self):
        """Setup selector for each test."""
        self.selector = OptimizerSelector()

    def test_small_dataset_uses_bootstrap(self):
        """Test small dataset uses BootstrapFewShot."""
        analysis = TaskAnalysis()
        config = self.selector.select(analysis, dataset_size=15)

        assert config.optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT

    def test_medium_dataset_uses_random_search(self):
        """Test medium dataset uses RandomSearch."""
        analysis = TaskAnalysis()
        config = self.selector.select(
            analysis, dataset_size=35, quality_profile="BALANCED"
        )

        assert config.optimizer_type == OptimizerType.BOOTSTRAP_RANDOM_SEARCH

    def test_large_dataset_high_quality_uses_mipro(self):
        """Test large dataset with HIGH_QUALITY uses MIPROv2."""
        analysis = TaskAnalysis()
        config = self.selector.select(
            analysis, dataset_size=60, quality_profile="HIGH_QUALITY"
        )

        assert config.optimizer_type == OptimizerType.MIPRO_V2

    def test_fast_cheap_always_bootstrap(self):
        """Test FAST_CHEAP always uses BootstrapFewShot."""
        analysis = TaskAnalysis()
        config = self.selector.select(
            analysis, dataset_size=100, quality_profile="FAST_CHEAP"
        )

        assert config.optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT


class TestPipelineSelector:
    """Tests for PipelineSelector."""

    def setup_method(self):
        """Setup selector for each test."""
        self.selector = PipelineSelector()

    def test_simple_task_uses_predict(self):
        """Test simple task uses Predict."""
        analysis = TaskAnalysis(
            needs_chain_of_thought=False,
            needs_tools=False,
            needs_retrieval=False,
            needs_multi_stage=False,
        )
        config = self.selector.select(analysis)

        assert config.pipeline_type == PipelineType.PREDICT

    def test_cot_task(self):
        """Test CoT task uses ChainOfThought."""
        analysis = TaskAnalysis(needs_chain_of_thought=True)
        config = self.selector.select(analysis)

        assert config.pipeline_type == PipelineType.CHAIN_OF_THOUGHT

    def test_tool_task_uses_react(self):
        """Test tool task uses ReAct."""
        analysis = TaskAnalysis(needs_tools=True, suggested_tools=["calculator"])
        config = self.selector.select(analysis, available_tools=["calculator"])

        assert config.pipeline_type == PipelineType.REACT
        assert "calculator" in config.tools

    def test_retrieval_task_uses_rag(self):
        """Test retrieval task uses RAG."""
        analysis = TaskAnalysis(needs_retrieval=True)
        config = self.selector.select(analysis)

        assert config.pipeline_type == PipelineType.RAG
        assert config.retriever_type is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
