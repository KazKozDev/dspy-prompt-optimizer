"""Metric Selector - Selects appropriate evaluation metric."""

from utils.settings import get_settings

from ..config import MetricConfig, MetricType, TaskAnalysis, TaskType


class MetricSelector:
    """Selects evaluation metric based on task analysis."""

    TASK_METRIC_MAP = {
        TaskType.CLASSIFICATION: MetricType.EXACT_MATCH,
        TaskType.EXTRACTION: MetricType.TOKEN_F1,
        TaskType.SUMMARIZATION: MetricType.LLM_JUDGE,
        TaskType.QA: MetricType.TOKEN_F1,
        TaskType.RAG: MetricType.LLM_JUDGE,
        TaskType.REASONING: MetricType.LLM_JUDGE,
        TaskType.GENERATION: MetricType.LLM_JUDGE,
        TaskType.ROUTING: MetricType.EXACT_MATCH,
        TaskType.CODE: MetricType.LLM_JUDGE,
    }

    def select(
        self,
        task_analysis: TaskAnalysis,
        quality_profile: str = "BALANCED",
        judge_model: str | None = None,
    ) -> MetricConfig:
        """Select metric configuration.

        Args:
            task_analysis: Result of task analysis
            quality_profile: Quality profile (affects LLM judge usage)
            judge_model: Optional specific model for LLM judge

        Returns:
            MetricConfig with selected settings
        """
        metric_type = self._select_metric_type(task_analysis, quality_profile)

        llm_judge_model = None
        llm_judge_criteria = None

        if metric_type == MetricType.LLM_JUDGE:
            llm_judge_model = judge_model or self._select_judge_model(quality_profile)
            llm_judge_criteria = self._generate_criteria(task_analysis)

        semantic_model = None
        if metric_type == MetricType.SEMANTIC_SIMILARITY:
            semantic_model = get_settings().model_defaults.semantic_model

        return MetricConfig(
            metric_type=metric_type,
            llm_judge_model=llm_judge_model,
            llm_judge_criteria=llm_judge_criteria,
            semantic_model=semantic_model,
        )

    def _select_metric_type(
        self, task_analysis: TaskAnalysis, quality_profile: str
    ) -> MetricType:
        """Select metric type based on task and quality profile."""
        default_metric = self.TASK_METRIC_MAP.get(
            task_analysis.task_type, MetricType.TOKEN_F1
        )

        if quality_profile == "FAST_CHEAP":
            if default_metric == MetricType.LLM_JUDGE:
                return MetricType.TOKEN_F1

        if quality_profile == "HIGH_QUALITY":
            if task_analysis.task_type in [TaskType.QA, TaskType.EXTRACTION]:
                return MetricType.LLM_JUDGE

        return default_metric

    def _select_judge_model(self, quality_profile: str) -> str:
        """Select model for LLM judge."""
        if quality_profile == "HIGH_QUALITY":
            return "openai/gpt-5"
        return f"openai/{get_settings().model_defaults.openai_chat}"

    def _generate_criteria(self, task_analysis: TaskAnalysis) -> str:
        """Generate evaluation criteria for LLM judge."""
        task_type = task_analysis.task_type

        criteria_map = {
            TaskType.SUMMARIZATION: """
Evaluate the summary based on:
1. Coverage: Does it capture the main points?
2. Conciseness: Is it appropriately brief?
3. Accuracy: Is the information correct?
4. Coherence: Is it well-written and logical?
""",
            TaskType.GENERATION: """
Evaluate the generated content based on:
1. Relevance: Does it address the prompt?
2. Quality: Is it well-written?
3. Completeness: Does it fully address the task?
4. Creativity: Is it engaging and original?
""",
            TaskType.REASONING: """
Evaluate the reasoning based on:
1. Correctness: Is the final answer correct?
2. Logic: Is the reasoning sound?
3. Completeness: Are all steps shown?
4. Clarity: Is it easy to follow?
""",
            TaskType.CODE: """
Evaluate the code based on:
1. Correctness: Does it solve the problem?
2. Efficiency: Is it reasonably efficient?
3. Readability: Is it clean and well-structured?
4. Completeness: Does it handle edge cases?
""",
            TaskType.RAG: """
Evaluate the answer based on:
1. Correctness: Is the answer accurate?
2. Faithfulness: Is it grounded in the context?
3. Completeness: Does it fully answer the question?
4. Relevance: Does it focus on what was asked?
""",
        }

        return criteria_map.get(
            task_type,
            """
Evaluate the response based on:
1. Correctness: Is it accurate?
2. Completeness: Is it thorough?
3. Relevance: Does it address the task?
""",
        )
