"""Pipeline Selector - Selects appropriate pipeline configuration."""

import os
import sys

from ..config import PipelineConfig, PipelineType, TaskAnalysis

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from pipelines.templates import get_template, suggest_template


class PipelineSelector:
    """Selects pipeline configuration based on task analysis."""

    def select(
        self, task_analysis: TaskAnalysis, available_tools: list[str] | None = None
    ) -> PipelineConfig:
        """Select pipeline configuration.

        Args:
            task_analysis: Result of task analysis
            available_tools: List of available tool names

        Returns:
            PipelineConfig with selected settings
        """
        pipeline_type = self._select_pipeline_type(task_analysis)
        template_name = self._select_template(task_analysis)
        tools = self._select_tools(task_analysis, available_tools)

        retriever_type = None
        retriever_k = 5
        if task_analysis.needs_retrieval:
            retriever_type = "faiss"
            retriever_k = 5 if task_analysis.complexity.value == "low" else 10

        stages = self._build_stages(template_name)

        return PipelineConfig(
            pipeline_type=pipeline_type,
            template_name=template_name,
            stages=stages,
            retriever_type=retriever_type,
            retriever_k=retriever_k,
            tools=tools,
        )

    def _select_pipeline_type(self, task_analysis: TaskAnalysis) -> PipelineType:
        """Select pipeline type based on task analysis."""
        if task_analysis.needs_tools:
            return PipelineType.REACT

        if task_analysis.needs_retrieval:
            return PipelineType.RAG

        if task_analysis.needs_multi_stage:
            return PipelineType.MULTI_STAGE

        if task_analysis.needs_chain_of_thought:
            return PipelineType.CHAIN_OF_THOUGHT

        return PipelineType.PREDICT

    def _select_template(self, task_analysis: TaskAnalysis) -> str:
        """Select pipeline template."""
        if task_analysis.suggested_pipeline_template:
            return task_analysis.suggested_pipeline_template

        return suggest_template(
            task_type=task_analysis.task_type.value,
            needs_retrieval=task_analysis.needs_retrieval,
            needs_tools=task_analysis.needs_tools,
        )

    def _select_tools(
        self, task_analysis: TaskAnalysis, available_tools: list[str] | None
    ) -> list[str]:
        """Select tools to use."""
        if not task_analysis.needs_tools:
            return []

        suggested = task_analysis.suggested_tools

        if available_tools:
            return [t for t in suggested if t in available_tools]

        return suggested

    def _build_stages(self, template_name: str) -> list[dict]:
        """Build stages from template."""
        template = get_template(template_name)
        if template:
            return template.get("stages", [])
        return []
