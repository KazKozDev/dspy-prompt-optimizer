"""
DSPy Meta-Agent package.
Provides intelligent auto-configuration for DSPy pipelines.
"""

from .config import (
    AgentConfig, 
    PipelineType, 
    MetricType, 
    OptimizerType,
    TaskType,
    ComplexityLevel,
    TaskAnalysis,
    PipelineConfig,
    MetricConfig,
    OptimizerConfig,
)
from .meta_agent import MetaAgent, create_meta_agent

__all__ = [
    "AgentConfig",
    "PipelineType",
    "MetricType", 
    "OptimizerType",
    "TaskType",
    "ComplexityLevel",
    "TaskAnalysis",
    "PipelineConfig",
    "MetricConfig",
    "OptimizerConfig",
    "MetaAgent",
    "create_meta_agent",
]
