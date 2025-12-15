"""
Meta Agent - Main orchestrator that auto-configures DSPy pipelines.

Analyzes task, selects pipeline, metrics, optimizer, and tools automatically.
Supports both AUTO and MANUAL modes.
"""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from .config import (
    AgentConfig, TaskAnalysis, PipelineConfig, MetricConfig, OptimizerConfig,
    PipelineType, MetricType, OptimizerType
)
from .selectors import (
    TaskAnalyzer, PipelineSelector, MetricSelector, OptimizerSelector, ToolSelector
)


class MetaAgent:
    """
    DSPy Meta-Agent for intelligent pipeline configuration.
    
    In AUTO mode: Analyzes task and automatically configures everything.
    In MANUAL mode: Uses user-provided configuration with validation.
    """
    
    def __init__(
        self,
        optimizer_model: str = "openai/gpt-4o-mini",
        artifacts_dir: str = "data/artifacts"
    ):
        """
        Initialize Meta-Agent.
        
        Args:
            optimizer_model: Model to use for optimization
            artifacts_dir: Directory to store artifacts
        """
        self.optimizer_model = optimizer_model
        self.artifacts_dir = artifacts_dir
        
        self.task_analyzer = TaskAnalyzer()
        self.pipeline_selector = PipelineSelector()
        self.metric_selector = MetricSelector()
        self.optimizer_selector = OptimizerSelector()
        self.tool_selector = ToolSelector()
    
    def configure(
        self,
        business_task: str,
        target_model: str,
        dataset: List[Dict[str, str]],
        quality_profile: str = "BALANCED",
        mode: str = "auto",
        manual_overrides: Optional[Dict[str, Any]] = None
    ) -> AgentConfig:
        """
        Configure the DSPy pipeline.
        
        In AUTO mode, analyzes task and selects optimal configuration.
        In MANUAL mode, uses provided overrides with validation.
        
        Args:
            business_task: Description of the task
            target_model: Target model for inference
            dataset: Training dataset
            quality_profile: FAST_CHEAP, BALANCED, or HIGH_QUALITY
            mode: "auto" or "manual"
            manual_overrides: Manual configuration overrides
            
        Returns:
            AgentConfig with all settings
        """
        config = AgentConfig(
            mode=mode,
            business_task=business_task,
            target_model=target_model,
            optimizer_model=self.optimizer_model,
            quality_profile=quality_profile
        )
        
        dataset_sample = dataset[:5] if dataset else None
        task_analysis = self.task_analyzer.analyze(business_task, dataset_sample)
        config.task_analysis = task_analysis
        config.agent_reasoning.append(f"Task Analysis: {task_analysis.reasoning}")
        
        if mode == "auto":
            config = self._auto_configure(config, task_analysis, len(dataset), quality_profile)
        else:
            config = self._manual_configure(config, manual_overrides or {})
        
        return config
    
    def _auto_configure(
        self,
        config: AgentConfig,
        task_analysis: TaskAnalysis,
        dataset_size: int,
        quality_profile: str
    ) -> AgentConfig:
        """Auto-configure all components."""
        available_tools = ["calculator", "web_search", "python_repl", "wikipedia"]
        
        pipeline_config = self.pipeline_selector.select(task_analysis, available_tools)
        config.pipeline_config = pipeline_config
        config.agent_reasoning.append(
            f"Pipeline: {pipeline_config.pipeline_type.value} "
            f"(template: {pipeline_config.template_name})"
        )
        
        metric_config = self.metric_selector.select(
            task_analysis, 
            quality_profile,
            judge_model=self.optimizer_model
        )
        config.metric_config = metric_config
        config.agent_reasoning.append(f"Metric: {metric_config.metric_type.value}")
        
        optimizer_config = self.optimizer_selector.select(
            task_analysis,
            dataset_size,
            quality_profile,
            target_model=config.target_model,
            optimizer_model=self.optimizer_model
        )
        config.optimizer_config = optimizer_config
        config.agent_reasoning.append(f"Optimizer: {optimizer_config.optimizer_type.value}")
        
        if task_analysis.needs_tools:
            tools = self.tool_selector.select(task_analysis, available_tools)
            config.pipeline_config.tools = tools
            config.agent_reasoning.append(f"Tools: {', '.join(tools)}")
        
        return config
    
    def _manual_configure(
        self,
        config: AgentConfig,
        overrides: Dict[str, Any]
    ) -> AgentConfig:
        """Apply manual configuration overrides."""
        if "pipeline_type" in overrides and overrides["pipeline_type"]:
            config.pipeline_config.pipeline_type = PipelineType(overrides["pipeline_type"])
            config.agent_reasoning.append(f"Pipeline override: {overrides['pipeline_type']}")
        
        if "template_name" in overrides and overrides["template_name"]:
            config.pipeline_config.template_name = overrides["template_name"]
        
        if "tools" in overrides and overrides["tools"]:
            config.pipeline_config.tools = overrides["tools"]
            config.agent_reasoning.append(f"Tools override: {', '.join(overrides['tools'])}")
        
        if "metric_type" in overrides and overrides["metric_type"]:
            config.metric_config.metric_type = MetricType(overrides["metric_type"])
            config.agent_reasoning.append(f"Metric override: {overrides['metric_type']}")
        
        if "llm_judge_model" in overrides and overrides["llm_judge_model"]:
            config.metric_config.llm_judge_model = overrides["llm_judge_model"]
        
        if "llm_judge_criteria" in overrides and overrides["llm_judge_criteria"]:
            config.metric_config.llm_judge_criteria = overrides["llm_judge_criteria"]
        
        if "optimizer_type" in overrides and overrides["optimizer_type"]:
            config.optimizer_config.optimizer_type = OptimizerType(overrides["optimizer_type"])
        
        if "max_bootstrapped_demos" in overrides and overrides["max_bootstrapped_demos"]:
            config.optimizer_config.max_bootstrapped_demos = overrides["max_bootstrapped_demos"]
        
        if "max_labeled_demos" in overrides and overrides["max_labeled_demos"]:
            config.optimizer_config.max_labeled_demos = overrides["max_labeled_demos"]
        
        # RAG configuration
        if overrides.get("enable_rag") or overrides.get("pipeline_type") == "rag":
            config.pipeline_config.pipeline_type = PipelineType.RAG
            config.pipeline_config.retriever_type = overrides.get("retriever_type", "faiss")
            config.pipeline_config.retriever_k = overrides.get("retriever_k", 5)
            config.agent_reasoning.append(f"RAG enabled: {config.pipeline_config.retriever_type}, k={config.pipeline_config.retriever_k}")
        
        # Distillation configuration
        if overrides.get("enable_distillation"):
            config.enable_distillation = True
            config.teacher_model = overrides.get("teacher_model", "openai/gpt-4o")
            config.distillation_samples = overrides.get("distillation_samples", 100)
            config.agent_reasoning.append(f"Distillation enabled: teacher={config.teacher_model}, samples={config.distillation_samples}")
        
        return config
    
    def get_configuration_summary(self, config: AgentConfig) -> Dict[str, Any]:
        """
        Get a human-readable summary of the configuration.
        
        Args:
            config: AgentConfig to summarize
            
        Returns:
            Dict with summary information
        """
        return {
            "mode": config.mode,
            "task": {
                "type": config.task_analysis.task_type.value,
                "domain": config.task_analysis.domain,
                "complexity": config.task_analysis.complexity.value,
            },
            "pipeline": {
                "type": config.pipeline_config.pipeline_type.value,
                "template": config.pipeline_config.template_name,
                "tools": config.pipeline_config.tools,
                "has_retrieval": config.pipeline_config.retriever_type is not None,
            },
            "metric": {
                "type": config.metric_config.metric_type.value,
                "uses_llm_judge": config.metric_config.llm_judge_model is not None,
            },
            "optimizer": {
                "type": config.optimizer_config.optimizer_type.value,
                "max_demos": config.optimizer_config.max_labeled_demos,
            },
            "reasoning": config.agent_reasoning,
        }
    
    def validate_config(self, config: AgentConfig) -> List[str]:
        """
        Validate configuration and return any warnings.
        
        Args:
            config: AgentConfig to validate
            
        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []
        
        if config.pipeline_config.pipeline_type == PipelineType.REACT:
            if not config.pipeline_config.tools:
                warnings.append("ReAct pipeline selected but no tools configured")
        
        if config.pipeline_config.pipeline_type == PipelineType.RAG:
            if not config.pipeline_config.retriever_type:
                warnings.append("RAG pipeline selected but no retriever configured")
        
        if config.metric_config.metric_type == MetricType.LLM_JUDGE:
            if not config.metric_config.llm_judge_model:
                warnings.append("LLM Judge metric selected but no judge model specified")
        
        if config.optimizer_config.optimizer_type == OptimizerType.DISTILLATION:
            if not config.optimizer_config.teacher_model:
                warnings.append("Distillation selected but no teacher model specified")
        
        return warnings


def create_meta_agent(
    optimizer_model: str = "openai/gpt-4o-mini",
    artifacts_dir: str = "data/artifacts"
) -> MetaAgent:
    """
    Factory function to create a MetaAgent.
    
    Args:
        optimizer_model: Model for optimization
        artifacts_dir: Directory for artifacts
        
    Returns:
        Configured MetaAgent instance
    """
    return MetaAgent(
        optimizer_model=optimizer_model,
        artifacts_dir=artifacts_dir
    )
