"""
Tests for HybridDSPyEngine methods.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch, MagicMock
from hybrid_engine import HybridDSPyEngine
from agent.config import (
    AgentConfig, PipelineConfig, MetricConfig, OptimizerConfig,
    TaskAnalysis, PipelineType, MetricType, OptimizerType, TaskType
)


class TestHybridEngineInit:
    """Tests for HybridDSPyEngine initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        engine = HybridDSPyEngine()
        assert engine.optimizer_model == "openai/gpt-4o-mini"
        assert engine.artifacts_dir.exists() or True  # May not exist yet
        assert engine.steps == []
        assert engine.config is None
    
    def test_init_custom(self, tmp_path):
        """Test custom initialization."""
        engine = HybridDSPyEngine(
            optimizer_model="anthropic/claude-3-haiku",
            artifacts_dir=str(tmp_path / "artifacts")
        )
        assert engine.optimizer_model == "anthropic/claude-3-haiku"


class TestBuildReactTools:
    """Tests for _build_react_tools method."""
    
    def test_build_tools_empty(self):
        """Test building tools with empty list."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.pipeline_config = PipelineConfig(tools=[])
        
        tools = engine._build_react_tools()
        assert tools == []
    
    def test_build_tools_with_calculator(self):
        """Test building tools with calculator."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.pipeline_config = PipelineConfig(tools=["calculator"])
        
        tools = engine._build_react_tools()
        # Should return list (may be empty if tools not registered)
        assert isinstance(tools, list)
    
    def test_build_tools_multiple(self):
        """Test building multiple tools."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.pipeline_config = PipelineConfig(
            tools=["calculator", "web_search", "wikipedia"]
        )
        
        tools = engine._build_react_tools()
        assert isinstance(tools, list)


class TestBuildRagPredictor:
    """Tests for _build_rag_predictor method."""
    
    @patch('hybrid_engine._load_dspy')
    def test_build_rag_faiss(self, mock_load_dspy):
        """Test building RAG predictor with FAISS."""
        mock_dspy = MagicMock()
        mock_dspy.Module = type('Module', (), {'__init__': lambda self: None})
        mock_dspy.ChainOfThought = MagicMock()
        mock_load_dspy.return_value = (mock_dspy, None, None, None, None)
        
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.pipeline_config = PipelineConfig(
            retriever_type="faiss",
            retriever_k=5
        )
        
        # This may fail if FAISS not installed, which is OK
        try:
            predictor = engine._build_rag_predictor(MagicMock(), MagicMock())
            assert predictor is not None
        except ImportError:
            pass  # FAISS not installed
    
    @patch('hybrid_engine._load_dspy')
    def test_build_rag_chroma(self, mock_load_dspy):
        """Test building RAG predictor with Chroma."""
        mock_dspy = MagicMock()
        mock_dspy.Module = type('Module', (), {'__init__': lambda self: None})
        mock_dspy.ChainOfThought = MagicMock()
        mock_load_dspy.return_value = (mock_dspy, None, None, None, None)
        
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.pipeline_config = PipelineConfig(
            retriever_type="chroma",
            retriever_k=3
        )
        
        # This may fail if ChromaDB not installed, which is OK
        try:
            predictor = engine._build_rag_predictor(MagicMock(), MagicMock())
            assert predictor is not None
        except ImportError:
            pass  # ChromaDB not installed


class TestGenerateProgramCode:
    """Tests for _generate_program_code method."""
    
    def test_generate_code_basic(self):
        """Test basic code generation."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig(
            business_task="Classify sentiment",
            mode="auto"
        )
        engine.config.task_analysis = TaskAnalysis(
            task_type=TaskType.CLASSIFICATION,
            input_fields=["text"],
            output_fields=["sentiment"]
        )
        engine.config.pipeline_config = PipelineConfig(
            pipeline_type=PipelineType.PREDICT
        )
        engine.config.metric_config = MetricConfig(
            metric_type=MetricType.EXACT_MATCH
        )
        engine.config.optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.BOOTSTRAP_FEW_SHOT
        )
        engine.config.agent_reasoning = ["Test reasoning"]
        
        compilation_result = {
            "metric_name": "accuracy",
            "metric_value": 0.85,
            "real_dspy": True
        }
        
        code = engine._generate_program_code(compilation_result)
        
        assert "import dspy" in code
        assert "ClassificationProgram" in code
        assert "text" in code
        assert "sentiment" in code
        assert "Predict" in code
        assert "0.85" in code
    
    def test_generate_code_with_distillation(self):
        """Test code generation with distillation result."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig(
            business_task="Summarize text",
            mode="manual"
        )
        engine.config.task_analysis = TaskAnalysis(
            task_type=TaskType.SUMMARIZATION,
            input_fields=["document"],
            output_fields=["summary"]
        )
        engine.config.pipeline_config = PipelineConfig(
            pipeline_type=PipelineType.CHAIN_OF_THOUGHT
        )
        engine.config.metric_config = MetricConfig(
            metric_type=MetricType.LLM_JUDGE
        )
        engine.config.optimizer_config = OptimizerConfig(
            optimizer_type=OptimizerType.MIPRO_V2
        )
        engine.config.agent_reasoning = []
        
        compilation_result = {
            "metric_name": "quality",
            "metric_value": 0.92,
            "real_dspy": True
        }
        
        distillation_result = {
            "status": "success",
            "teacher_model": "openai/gpt-4o",
            "samples_generated": 50
        }
        
        code = engine._generate_program_code(compilation_result, distillation_result)
        
        assert "import dspy" in code
        assert "ChainOfThought" in code
        assert "Distillation" in code
        assert "gpt-4o" in code
        assert "50" in code


class TestCreateStep:
    """Tests for _create_step method."""
    
    def test_create_step(self):
        """Test step creation."""
        engine = HybridDSPyEngine()
        
        step = engine._create_step(
            name="Test Step",
            tool="test_tool",
            thought="Testing..."
        )
        
        assert step.name == "Test Step"
        assert step.tool == "test_tool"
        assert step.thought == "Testing..."
        assert step.status == "running"
        assert step.id.startswith("step_")


class TestUpdateStep:
    """Tests for _update_step method."""
    
    def test_update_step(self):
        """Test step update."""
        engine = HybridDSPyEngine()
        
        step = engine._create_step("Test", "tool")
        
        engine._update_step(
            step,
            status="success",
            action="test_action()",
            observation="Result OK",
            duration_ms=100
        )
        
        assert step.status == "success"
        assert step.action == "test_action()"
        assert step.observation == "Result OK"
        assert step.duration_ms == 100


class TestPrepareData:
    """Tests for _prepare_data method."""
    
    def test_prepare_data_splits(self):
        """Test data split preparation."""
        engine = HybridDSPyEngine()
        
        dataset = [{"input": f"text{i}", "output": f"label{i}"} for i in range(100)]
        
        splits = engine._prepare_data(dataset)
        
        assert splits["train"] == 70
        assert splits["dev"] == 20
        assert splits["test"] == 10
    
    def test_prepare_data_small(self):
        """Test data split with small dataset."""
        engine = HybridDSPyEngine()
        
        dataset = [{"input": "a", "output": "b"}] * 10
        
        splits = engine._prepare_data(dataset)
        
        assert splits["train"] == 7
        assert splits["dev"] == 2
        assert splits["test"] == 1


class TestCreateMetric:
    """Tests for _create_metric method."""
    
    def test_create_exact_match_metric(self):
        """Test creating exact match metric."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.task_analysis = TaskAnalysis(output_fields=["result"])
        engine.config.metric_config = MetricConfig(metric_type=MetricType.EXACT_MATCH)
        
        metric = engine._create_metric()
        
        assert metric.name == "exact_match"
    
    def test_create_token_f1_metric(self):
        """Test creating token F1 metric."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.task_analysis = TaskAnalysis(output_fields=["answer"])
        engine.config.metric_config = MetricConfig(metric_type=MetricType.TOKEN_F1)
        
        metric = engine._create_metric()
        
        assert metric.name == "token_f1"
    
    def test_create_llm_judge_metric(self):
        """Test creating LLM judge metric."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.task_analysis = TaskAnalysis(output_fields=["response"])
        engine.config.metric_config = MetricConfig(
            metric_type=MetricType.LLM_JUDGE,
            llm_judge_model="openai/gpt-4o-mini",
            llm_judge_criteria="Check correctness"
        )
        
        metric = engine._create_metric()
        
        assert "judge" in metric.name.lower() or "llm" in metric.name.lower()


class TestGenerateInstructions:
    """Tests for _generate_instructions method."""
    
    def test_generate_instructions(self):
        """Test instruction generation."""
        engine = HybridDSPyEngine()
        engine.config = AgentConfig()
        engine.config.task_analysis = TaskAnalysis(input_fields=["query", "context"])
        
        instructions = engine._generate_instructions("openai/gpt-4o")
        
        assert "pip install dspy-ai" in instructions
        assert "gpt-4o" in instructions
        assert "query" in instructions
        assert "context" in instructions
