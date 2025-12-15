"""LLM-as-Judge Metrics - Use LLM to evaluate predictions."""

import os
from typing import Any

from prompts.templates import get_prompt
from utils.settings import get_settings

from .base import BaseMetric


class LLMJudgeMetric(BaseMetric):
    """LLM-as-Judge metric.

    Uses an LLM to evaluate predictions based on custom criteria.
    Supports multiple judge types: correctness, faithfulness, coherence, custom.
    """

    name = "llm_judge"
    description = "LLM-based evaluation metric"

    def __init__(
        self,
        judge_type: str = "correctness",
        judge_model: str | None = None,
        output_field: str = "result",
        custom_criteria: str | None = None,
        temperature: float | None = None,
    ):
        """Initialize LLMJudgeMetric.

        Args:
            judge_type: Type of judge ("correctness", "faithfulness", "coherence", "custom")
            judge_model: Model to use for judging (e.g., "openai/gpt-5-mini")
            output_field: Name of the output field to evaluate
            custom_criteria: Custom evaluation criteria (required if judge_type="custom")
            temperature: Temperature for judge model
        """
        settings = get_settings()

        self.judge_type = judge_type
        self.judge_model = (
            judge_model or f"openai/{settings.model_defaults.openai_chat}"
        )
        self.output_field = output_field
        self.custom_criteria = custom_criteria
        self.temperature = (
            settings.judge.temperature if temperature is None else temperature
        )

        if judge_type == "custom" and not custom_criteria:
            raise ValueError("custom_criteria required when judge_type='custom'")

        self._llm = None

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            import dspy

            if "/" in self.judge_model:
                provider, model_name = self.judge_model.split("/", 1)
            else:
                provider = "openai"
                model_name = self.judge_model

            if provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                self._llm = dspy.LM(
                    f"openai/{model_name}",
                    api_key=api_key,
                    temperature=self.temperature,
                )
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                self._llm = dspy.LM(
                    f"anthropic/{model_name}",
                    api_key=api_key,
                    temperature=self.temperature,
                )
            elif provider == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self._llm = dspy.LM(
                    f"ollama_chat/{model_name}",
                    api_base=base_url,
                    temperature=self.temperature,
                )
            else:
                self._llm = dspy.LM(self.judge_model, temperature=self.temperature)

        return self._llm

    def _build_prompt(
        self, task: str, expected: str, predicted: str, context: str | None = None
    ) -> str:
        """Build the judge prompt."""
        try:
            template = get_prompt("llm_judge_prompts", self.judge_type)
        except Exception:
            template = get_prompt("llm_judge_prompts", "correctness")

        return template.format(
            task=task,
            expected=expected,
            predicted=predicted,
            context=context or "",
            question=task,
            criteria=self.custom_criteria or "",
        )

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM judge response."""
        import json
        import re

        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        score_match = re.search(r'score["\s:]+([0-9.]+)', response.lower())
        if score_match:
            return {"score": float(score_match.group(1)), "reasoning": response}

        return {
            "score": 0.5,
            "reasoning": f"Could not parse response: {response[:200]}",
        }

    def __call__(
        self, example: Any, prediction: Any, trace: Any | None = None
    ) -> float:
        """Evaluate prediction using LLM judge.

        Args:
            example: Ground truth with expected output and task
            prediction: Model prediction
            trace: Optional trace (unused)

        Returns:
            Score between 0.0 and 1.0
        """
        expected = self.get_field_value(example, self.output_field)
        predicted = self.get_field_value(prediction, self.output_field)

        task = self.get_field_value(example, "text") or self.get_field_value(
            example, "input"
        )
        context = self.get_field_value(example, "context")

        prompt = self._build_prompt(task, expected, predicted, context)

        try:
            llm = self._get_llm()
            response = llm(prompt)

            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            response = str(response)

            result = self._parse_response(response)
            score = float(result.get("score", 0.5))

            return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"LLM Judge error: {e}")
            return 0.5

    def evaluate_with_reasoning(self, example: Any, prediction: Any) -> dict[str, Any]:
        """Evaluate and return detailed result with reasoning.

        Returns dict with score and reasoning from the judge.
        """
        expected = self.get_field_value(example, self.output_field)
        predicted = self.get_field_value(prediction, self.output_field)

        task = self.get_field_value(example, "text") or self.get_field_value(
            example, "input"
        )
        context = self.get_field_value(example, "context")

        prompt = self._build_prompt(task, expected, predicted, context)

        try:
            llm = self._get_llm()
            response = llm(prompt)

            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            response = str(response)

            result = self._parse_response(response)
            result["score"] = max(0.0, min(1.0, float(result.get("score", 0.5))))
            result["raw_response"] = response

            return result

        except Exception as e:
            return {"score": 0.5, "reasoning": f"Error: {str(e)}", "error": str(e)}


class CorrectnessJudge(LLMJudgeMetric):
    """Convenience class for correctness evaluation."""

    name = "correctness_judge"
    description = "LLM judge for answer correctness"

    def __init__(self, judge_model: str | None = None, output_field: str = "result"):
        super().__init__(
            judge_type="correctness", judge_model=judge_model, output_field=output_field
        )


class FaithfulnessJudge(LLMJudgeMetric):
    """Convenience class for faithfulness evaluation (RAG)."""

    name = "faithfulness_judge"
    description = "LLM judge for RAG faithfulness"

    def __init__(self, judge_model: str | None = None, output_field: str = "result"):
        super().__init__(
            judge_type="faithfulness",
            judge_model=judge_model,
            output_field=output_field,
        )


class CoherenceJudge(LLMJudgeMetric):
    """Convenience class for coherence/quality evaluation."""

    name = "coherence_judge"
    description = "LLM judge for response coherence and quality"

    def __init__(self, judge_model: str | None = None, output_field: str = "result"):
        super().__init__(
            judge_type="coherence", judge_model=judge_model, output_field=output_field
        )
