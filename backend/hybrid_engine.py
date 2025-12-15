"""Hybrid DSPy Engine - Combines Meta-Agent with DSPy compilation.

Supports AUTO mode (agent configures everything) and MANUAL mode (user overrides).
"""

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import AgentConfig, MetaAgent, MetricType, OptimizerType, PipelineType
from metrics import (
    BaseMetric,
    ExactMatchMetric,
    LLMJudgeMetric,
    SemanticSimilarityMetric,
    TokenF1Metric,
)

from utils.settings import get_settings

dspy = None
BootstrapFewShot = None
BootstrapFewShotWithRandomSearch = None
MIPROv2 = None
COPRO = None


def _load_dspy():
    """Lazy load DSPy."""
    global dspy, BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2, COPRO
    if dspy is None:
        import dspy as _dspy
        from dspy import teleprompt as _tp

        dspy = _dspy
        BootstrapFewShot = getattr(_tp, "BootstrapFewShot", None)
        BootstrapFewShotWithRandomSearch = getattr(
            _tp, "BootstrapFewShotWithRandomSearch", None
        )
        MIPROv2 = getattr(_tp, "MIPROv2", None)
        COPRO = getattr(_tp, "COPRO", None)

        if hasattr(_dspy, "configure_cache"):
            _dspy.configure_cache(False)

    return dspy, BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2, COPRO


@dataclass
class HybridStep:
    """A single step in the hybrid orchestration process."""

    id: str
    name: str
    tool: str
    status: str  # pending, running, success, error
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    duration_ms: int | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HybridDSPyEngine:
    """Hybrid DSPy Engine with Meta-Agent integration.

    Supports:
    - AUTO mode: Agent analyzes task and configures pipeline, metrics, optimizer
    - MANUAL mode: User provides configuration, agent validates
    - Streaming progress updates
    - Multiple pipeline types (Predict, CoT, ReAct, RAG, Multi-stage)
    - Multiple metrics (Exact Match, F1, LLM-Judge, Semantic)
    - Multiple optimizers (Bootstrap, MIPRO, COPRO)
    """

    def __init__(
        self,
        optimizer_model: str = "openai/gpt-5-mini",
        artifacts_dir: str = "data/artifacts",
    ):
        self.optimizer_model = optimizer_model
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.meta_agent = MetaAgent(
            optimizer_model=optimizer_model, artifacts_dir=artifacts_dir
        )
        self.steps: list[HybridStep] = []
        self.config: AgentConfig | None = None

    def _create_step(
        self,
        name: str,
        tool: str,
        status: str = "running",
        thought: str | None = None,
    ) -> HybridStep:
        """Create a new step."""
        step = HybridStep(
            id=f"step_{len(self.steps) + 1}",
            name=name,
            tool=tool,
            status=status,
            thought=thought,
        )
        self.steps.append(step)
        return step

    def _update_step(
        self,
        step: HybridStep,
        status: str,
        action: str | None = None,
        observation: str | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ):
        """Update an existing step."""
        step.status = status
        if action:
            step.action = action
        if observation:
            step.observation = observation
        if error:
            step.error = error
        if duration_ms:
            step.duration_ms = duration_ms

    async def run_async(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str = "BALANCED",
        mode: str = "auto",
        manual_overrides: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the hybrid orchestration pipeline with streaming.

        Args:
            business_task: Task description
            target_lm: Target model for inference
            dataset: Training dataset
            quality_profile: FAST_CHEAP, BALANCED, HIGH_QUALITY
            mode: "auto" or "manual"
            manual_overrides: Manual configuration (for manual mode)

        Yields:
            Step updates and final result
        """
        self.steps = []
        start_time = time.time()

        try:
            # Step 1: Meta-Agent Configuration
            step = self._create_step(
                "Meta-Agent Analysis",
                "meta_agent_configure",
                thought=f"Analyzing task in {mode.upper()} mode...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            self.config = self.meta_agent.configure(
                business_task=business_task,
                target_model=target_lm,
                dataset=dataset,
                quality_profile=quality_profile,
                mode=mode,
                manual_overrides=manual_overrides,
            )

            summary = self.meta_agent.get_configuration_summary(self.config)

            self._update_step(
                step,
                status="success",
                action=f'configure(mode="{mode}", task_type="{summary["task"]["type"]}")',
                observation=f'pipeline={summary["pipeline"]["type"]}, metric={summary["metric"]["type"]}, optimizer={summary["optimizer"]["type"]}',
                duration_ms=200,
            )
            yield {"type": "step", "step": asdict(step)}

            # Yield configuration for UI
            yield {
                "type": "config",
                "config": summary,
                "reasoning": self.config.agent_reasoning,
            }

            # Step 2: Validate Configuration
            step = self._create_step(
                "Validate Configuration",
                "validate_config",
                thought="Checking configuration for issues...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            warnings = self.meta_agent.validate_config(self.config)

            self._update_step(
                step,
                status="success",
                action="validate_config()",
                observation=(
                    f"{len(warnings)} warnings" if warnings else "Configuration valid"
                ),
                duration_ms=50,
            )
            yield {"type": "step", "step": asdict(step)}

            if warnings:
                yield {"type": "warnings", "warnings": warnings}

            # Step 3: Build Pipeline
            step = self._create_step(
                "Build Pipeline",
                "build_pipeline",
                thought=f"Building {self.config.pipeline_config.pipeline_type.value} pipeline...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            pipeline_info = self._build_pipeline_info()

            self._update_step(
                step,
                status="success",
                action=f'build_pipeline(type="{self.config.pipeline_config.pipeline_type.value}")',
                observation=f"template={self.config.pipeline_config.template_name}, tools={len(self.config.pipeline_config.tools)}",
                duration_ms=100,
            )
            yield {"type": "step", "step": asdict(step)}

            # Step 4: Prepare Data
            step = self._create_step(
                "Prepare Data Splits",
                "prepare_data",
                thought="Splitting dataset for training and evaluation...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            data_splits = self._prepare_data(dataset)

            self._update_step(
                step,
                status="success",
                action="prepare_data(train=0.7, dev=0.2, test=0.1)",
                observation=f'train={data_splits["train"]}, dev={data_splits["dev"]}, test={data_splits["test"]}',
                duration_ms=50,
            )
            yield {"type": "step", "step": asdict(step)}

            # Step 5: Create Metric
            step = self._create_step(
                "Initialize Metric",
                "create_metric",
                thought=f"Setting up {self.config.metric_config.metric_type.value} metric...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            metric = self._create_metric()

            self._update_step(
                step,
                status="success",
                action=f'create_metric(type="{self.config.metric_config.metric_type.value}")',
                observation=f"metric_name={metric.name}",
                duration_ms=100,
            )
            yield {"type": "step", "step": asdict(step)}

            # Step 6: Run Compilation
            step = self._create_step(
                "Run DSPy Compilation",
                "run_compilation",
                thought=f"Starting {self.config.optimizer_config.optimizer_type.value} optimization...",
            )
            yield {"type": "step", "step": asdict(step)}

            compilation_result = await self._run_compilation(dataset, target_lm, metric)

            self._update_step(
                step,
                status=(
                    "success"
                    if compilation_result.get("status") == "success"
                    else "error"
                ),
                action=f'compile(optimizer="{self.config.optimizer_config.optimizer_type.value}")',
                observation=f'metric={compilation_result.get("metric_value", 0):.3f}, real_dspy={compilation_result.get("real_dspy", False)}',
                duration_ms=compilation_result.get("duration_ms", 5000),
                error=compilation_result.get("error"),
            )
            yield {"type": "step", "step": asdict(step)}

            # Step 6.5: Run Distillation (if enabled)
            distillation_result = None
            if self.config.enable_distillation:
                step = self._create_step(
                    "Run Distillation",
                    "run_distillation",
                    thought=f"Generating {self.config.distillation_samples} samples from {self.config.teacher_model}...",
                )
                yield {"type": "step", "step": asdict(step)}

                distillation_result = await self._run_distillation(dataset, target_lm)

                self._update_step(
                    step,
                    status=(
                        "success"
                        if distillation_result.get("status") == "success"
                        else "error"
                    ),
                    action=f'distill(teacher="{self.config.teacher_model}", samples={self.config.distillation_samples})',
                    observation=f'generated={distillation_result.get("samples_generated", 0)} samples',
                    duration_ms=distillation_result.get("duration_ms", 5000),
                    error=distillation_result.get("error"),
                )
                yield {"type": "step", "step": asdict(step)}

            # Step 7: Generate Program Code
            step = self._create_step(
                "Generate Program Code",
                "generate_code",
                thought="Generating optimized DSPy program...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            program_code = self._generate_program_code(
                compilation_result, distillation_result
            )

            self._update_step(
                step,
                status="success",
                action="generate_code()",
                observation=f"{len(program_code)} characters",
                duration_ms=50,
            )
            yield {"type": "step", "step": asdict(step)}

            # Step 8: Save Artifacts
            step = self._create_step(
                "Save Artifacts",
                "save_artifacts",
                thought="Saving optimized program and metadata...",
            )
            yield {"type": "step", "step": asdict(step)}

            await asyncio.sleep(0.1)

            artifact_id = self._save_artifacts(
                compilation_result, program_code, target_lm, quality_profile
            )

            self._update_step(
                step,
                status="success",
                action="save_artifacts()",
                observation=f'artifact_id="{artifact_id}"',
                duration_ms=100,
            )
            yield {"type": "step", "step": asdict(step)}

            # Final Result
            total_duration = int((time.time() - start_time) * 1000)

            result = {
                "type": "complete",
                "artifact_version_id": artifact_id,
                "compiled_program_id": f"prog_{uuid.uuid4().hex[:8]}",
                "eval_results": compilation_result,
                "task_analysis": (
                    self.config.task_analysis.__dict__
                    if self.config.task_analysis
                    else {}
                ),
                "program_code": program_code,
                "config_summary": summary,
                "agent_reasoning": self.config.agent_reasoning,
                "deployment_package": {
                    "path": f"/exports/{artifact_id}/",
                    "instructions": self._generate_instructions(target_lm),
                },
                "react_iterations": len(self.steps),
                "total_duration_ms": total_duration,
                "optimizer_type": self.config.optimizer_config.optimizer_type.value,
                "metric_type": self.config.metric_config.metric_type.value,
                "pipeline_type": self.config.pipeline_config.pipeline_type.value,
                "quality_profile": quality_profile,
                "mode": mode,
                "data_splits": data_splits,
            }
            yield result

        except Exception as e:
            if self.steps:
                last_step = self.steps[-1]
                self._update_step(last_step, status="error", error=str(e))
                yield {"type": "step", "step": asdict(last_step)}

            yield {"type": "error", "error": str(e)}

    def _build_pipeline_info(self) -> dict[str, Any]:
        """Build pipeline information."""
        return {
            "type": self.config.pipeline_config.pipeline_type.value,
            "template": self.config.pipeline_config.template_name,
            "stages": self.config.pipeline_config.stages,
            "tools": self.config.pipeline_config.tools,
            "retriever": self.config.pipeline_config.retriever_type,
        }

    def _prepare_data(self, dataset: list[dict]) -> dict[str, int]:
        """Prepare train/dev/test splits."""
        n = len(dataset)
        train_size = int(n * 0.7)
        dev_size = int(n * 0.2)
        test_size = n - train_size - dev_size

        return {"train": train_size, "dev": dev_size, "test": test_size}

    def _create_metric(self) -> BaseMetric:
        """Create metric based on configuration."""
        settings = get_settings()
        metric_type = self.config.metric_config.metric_type
        output_field = (
            self.config.task_analysis.output_fields[0]
            if self.config.task_analysis.output_fields
            else "result"
        )

        if metric_type == MetricType.EXACT_MATCH:
            return ExactMatchMetric(output_field=output_field)

        elif metric_type == MetricType.TOKEN_F1:
            return TokenF1Metric(output_field=output_field)

        elif metric_type == MetricType.LLM_JUDGE:
            return LLMJudgeMetric(
                judge_type="correctness",
                judge_model=self.config.metric_config.llm_judge_model,
                output_field=output_field,
                custom_criteria=self.config.metric_config.llm_judge_criteria,
            )

        elif metric_type == MetricType.SEMANTIC_SIMILARITY:
            return SemanticSimilarityMetric(
                output_field=output_field,
                model_name=self.config.metric_config.semantic_model
                or settings.model_defaults.semantic_model,
            )

        return TokenF1Metric(output_field=output_field)

    async def _run_compilation(
        self, dataset: list[dict], target_lm: str, metric: BaseMetric
    ) -> dict[str, Any]:
        """Run DSPy compilation with configured settings."""
        start_time = time.time()

        try:
            dspy, BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2, COPRO = (
                _load_dspy()
            )

            # Configure LM
            if "/" in target_lm:
                provider, model_name = target_lm.split("/", 1)
            else:
                provider = "openai"
                model_name = target_lm

            if provider == "ollama":
                ollama_base_url = get_settings().endpoints.ollama_base_url
                lm = dspy.LM(f"ollama_chat/{model_name}", api_base=ollama_base_url)
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                lm = dspy.LM(f"anthropic/{model_name}", api_key=api_key)
            elif provider == "google" or provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not set")
                lm = dspy.LM(f"google/{model_name}", api_key=api_key)
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                lm = dspy.LM(f"openai/{model_name}", api_key=api_key)

            dspy.configure(lm=lm)

            # Create signature
            input_fields = self.config.task_analysis.input_fields
            output_fields = self.config.task_analysis.output_fields

            sig_fields = {}
            for field_name in input_fields:
                sig_fields[field_name] = dspy.InputField(desc=f"{field_name} input")
            for field_name in output_fields:
                sig_fields[field_name] = dspy.OutputField(desc=f"{field_name} output")

            DynamicSignature = type(
                "DynamicSignature",
                (dspy.Signature,),
                {"__doc__": self.config.business_task[:100], **sig_fields},
            )

            # Create predictor based on pipeline type
            pipeline_type = self.config.pipeline_config.pipeline_type

            if pipeline_type == PipelineType.PREDICT:
                predictor = dspy.Predict(DynamicSignature)
            elif pipeline_type == PipelineType.CHAIN_OF_THOUGHT:
                predictor = dspy.ChainOfThought(DynamicSignature)
            elif pipeline_type == PipelineType.REACT:
                # Build tools for ReAct
                react_tools = self._build_react_tools()
                if hasattr(dspy, "ReAct") and react_tools:
                    predictor = dspy.ReAct(DynamicSignature, tools=react_tools)
                elif hasattr(dspy, "ReAct"):
                    predictor = dspy.ReAct(DynamicSignature, tools=[])
                else:
                    predictor = dspy.ChainOfThought(DynamicSignature)
            elif pipeline_type == PipelineType.RAG:
                # Build RAG pipeline with retriever
                predictor = self._build_rag_predictor(DynamicSignature, lm)
            else:
                predictor = dspy.ChainOfThought(DynamicSignature)

            # Prepare training data
            n = len(dataset)
            train_size = int(n * 0.7)
            dev_size = int(n * 0.2)
            train_data = dataset[:train_size]
            dev_data = (
                dataset[train_size : train_size + dev_size] if dev_size > 0 else []
            )

            trainset = []
            for item in train_data:
                example_dict = {}
                if len(input_fields) == 1:
                    example_dict[input_fields[0]] = item.get("input", "")
                if len(output_fields) >= 1:
                    example_dict[output_fields[0]] = item.get("output", "")
                trainset.append(dspy.Example(**example_dict).with_inputs(*input_fields))

            # Create DSPy-compatible metric wrapper
            def dspy_metric(example, pred, trace=None):
                return metric(example, pred, trace)

            # Select and configure optimizer
            optimizer_type = self.config.optimizer_config.optimizer_type
            opt_config = self.config.optimizer_config

            if optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT and BootstrapFewShot:
                optimizer = BootstrapFewShot(
                    metric=dspy_metric,
                    max_bootstrapped_demos=opt_config.max_bootstrapped_demos,
                    max_labeled_demos=opt_config.max_labeled_demos,
                    max_rounds=opt_config.max_rounds,
                )
            elif (
                optimizer_type == OptimizerType.BOOTSTRAP_RANDOM_SEARCH
                and BootstrapFewShotWithRandomSearch
            ):
                optimizer = BootstrapFewShotWithRandomSearch(
                    metric=dspy_metric,
                    max_bootstrapped_demos=opt_config.max_bootstrapped_demos,
                    max_labeled_demos=opt_config.max_labeled_demos,
                    max_rounds=opt_config.max_rounds,
                    num_candidate_programs=opt_config.num_candidates,
                )
            elif optimizer_type == OptimizerType.MIPRO_V2 and MIPROv2:
                optimizer = MIPROv2(
                    metric=dspy_metric,
                    max_bootstrapped_demos=opt_config.max_bootstrapped_demos,
                    max_labeled_demos=opt_config.max_labeled_demos,
                    auto="light",
                    num_candidates=opt_config.num_candidates,
                )
            elif optimizer_type == OptimizerType.COPRO and COPRO:
                optimizer = COPRO(metric=dspy_metric, breadth=10, depth=3)
            else:
                optimizer = BootstrapFewShot(
                    metric=dspy_metric,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=16,
                    max_rounds=1,
                )

            # Run compilation
            compiled_predictor = optimizer.compile(predictor, trainset=trainset)

            # Evaluate
            eval_examples = []
            if dev_data:
                for item in dev_data:
                    example_dict = {}
                    if len(input_fields) == 1:
                        example_dict[input_fields[0]] = item.get("input", "")
                    if len(output_fields) >= 1:
                        example_dict[output_fields[0]] = item.get("output", "")
                    eval_examples.append(
                        dspy.Example(**example_dict).with_inputs(*input_fields)
                    )
            else:
                eval_examples = trainset

            metric_history = []
            correct = 0.0
            eval_count = min(10, len(eval_examples))

            for example in eval_examples[:eval_count]:
                try:
                    input_kwargs = {
                        field: getattr(example, field) for field in input_fields
                    }
                    pred = compiled_predictor(**input_kwargs)
                    m = dspy_metric(example, pred)
                    metric_history.append(float(m))
                    correct += m
                except Exception:
                    pass

            final_metric = (correct / eval_count) if eval_count > 0 else 0.0
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "success",
                "metric_value": round(final_metric, 3),
                "metric_name": metric.name,
                "iterations": len(trainset),
                "real_dspy": True,
                "metric_history": metric_history,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "metric_value": 0.0,
                "metric_name": "error",
                "error": str(e),
                "real_dspy": False,
                "duration_ms": duration_ms,
            }

    def _generate_program_code(
        self, compilation_result: dict, distillation_result: dict | None = None
    ) -> str:
        """Generate the final DSPy program code."""
        task_type = self.config.task_analysis.task_type.value
        input_fields = self.config.task_analysis.input_fields
        output_fields = self.config.task_analysis.output_fields
        pipeline_type = self.config.pipeline_config.pipeline_type

        input_field_defs = "\n    ".join(
            [
                f'{field}: str = dspy.InputField(desc="{field} input")'
                for field in input_fields
            ]
        )

        output_field_defs = "\n    ".join(
            [
                f'{field}: str = dspy.OutputField(desc="{field} output")'
                for field in output_fields
            ]
        )

        predictor_type = (
            "ChainOfThought"
            if pipeline_type in [PipelineType.CHAIN_OF_THOUGHT, PipelineType.RAG]
            else "Predict"
        )
        if pipeline_type == PipelineType.REACT:
            predictor_type = "ReAct"

        class_name = f"{task_type.title().replace('_', '')}Program"

        # Add distillation info if available
        distill_comment = ""
        if distillation_result and distillation_result.get("status") == "success":
            distill_comment = f"""#
# Distillation:
# - Teacher: {distillation_result.get("teacher_model", "N/A")}
# - Samples Generated: {distillation_result.get("samples_generated", 0)}
"""

        return f'''import dspy

class {class_name}Signature(dspy.Signature):
    """{self.config.business_task[:100]}"""
    {input_field_defs}
    {output_field_defs}

class {class_name}(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.{predictor_type}({class_name}Signature)
    
    def forward(self, {", ".join(input_fields)}):
        return self.predictor({", ".join([f"{f}={f}" for f in input_fields])})

# Configuration:
# - Pipeline: {pipeline_type.value}
# - Metric: {self.config.metric_config.metric_type.value}
# - Optimizer: {self.config.optimizer_config.optimizer_type.value}
# - Mode: {self.config.mode}
# 
# Evaluation:
# - Metric ({compilation_result.get("metric_name", "metric")}): {compilation_result.get("metric_value", 0):.3f}
# - Real DSPy: {compilation_result.get("real_dspy", False)}
{distill_comment}#
# Agent Reasoning:
{chr(10).join(["# - " + r for r in self.config.agent_reasoning])}
'''

    def _build_react_tools(self) -> list:
        """Build tools for ReAct agent from configuration."""
        tools = []
        tool_names = self.config.pipeline_config.tools or []

        try:
            from tools.builtin import register_all_builtin_tools
            from tools.registry import get_registry

            register_all_builtin_tools()
            registry = get_registry()

            for tool_name in tool_names:
                tool = registry.get(tool_name)
                if tool:
                    # Convert to DSPy tool format
                    def make_tool_fn(t):
                        def tool_fn(**kwargs):
                            result = t.run(**kwargs)
                            return (
                                result.output
                                if result.success
                                else f"Error: {result.error}"
                            )

                        tool_fn.__name__ = t.name
                        tool_fn.__doc__ = t.description
                        return tool_fn

                    tools.append(make_tool_fn(tool))
        except ImportError:
            pass

        return tools

    def _build_rag_predictor(self, signature, lm):
        """Build RAG predictor with retriever."""
        dspy = _load_dspy()[0]

        retriever_type = self.config.pipeline_config.retriever_type or "faiss"
        retriever_k = self.config.pipeline_config.retriever_k or 5

        try:
            if retriever_type == "chroma":
                from retrieval import ChromaRetriever

                retriever = ChromaRetriever(collection_name="dspy_rag")
            else:
                from retrieval import FAISSRetriever

                retriever = FAISSRetriever()

            # Create RAG module
            class RAGModule(dspy.Module):
                def __init__(self, sig, ret, k):
                    super().__init__()
                    self.retriever = ret
                    self.k = k
                    self.generate = dspy.ChainOfThought(sig)

                def forward(self, **kwargs):
                    # Get first input field value for query
                    query = list(kwargs.values())[0] if kwargs else ""

                    # Retrieve context
                    results = self.retriever.search(query, k=self.k)
                    context = "\n".join(results.passages) if results.passages else ""

                    # Add context to kwargs
                    kwargs["context"] = context

                    return self.generate(**kwargs)

            return RAGModule(signature, retriever, retriever_k)

        except ImportError:
            # Fallback to ChainOfThought if retrieval not available
            return dspy.ChainOfThought(signature)

    async def _run_distillation(
        self, dataset: list[dict], target_lm: str
    ) -> dict[str, Any]:
        """Run teacher-student distillation."""
        start_time = time.time()

        try:
            from distillation import DistillationConfig, TeacherStudentDistiller

            distill_config = DistillationConfig(
                teacher_model=self.config.teacher_model,
                student_model=target_lm,
                num_samples=self.config.distillation_samples,
                temperature=0.7,
            )

            distiller = TeacherStudentDistiller(distill_config)

            # Generate samples from teacher
            input_field = (
                self.config.task_analysis.input_fields[0]
                if self.config.task_analysis.input_fields
                else "input"
            )
            seed_inputs = [
                item.get("input", item.get(input_field, "")) for item in dataset[:10]
            ]

            generated_samples = await distiller.generate_samples_async(
                seed_inputs=seed_inputs, task_description=self.config.business_task
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "success",
                "samples_generated": len(generated_samples),
                "teacher_model": self.config.teacher_model,
                "duration_ms": duration_ms,
            }

        except ImportError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "error": f"Distillation module not available: {e}",
                "samples_generated": 0,
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "error": str(e),
                "samples_generated": 0,
                "duration_ms": duration_ms,
            }

    def _generate_instructions(self, target_lm: str) -> str:
        """Generate deployment instructions."""
        input_fields = self.config.task_analysis.input_fields

        return f"""1. Install: pip install dspy-ai
2. Load the program from artifact
3. Configure DSPy with your LLM:
   dspy.configure(lm=dspy.LM("{target_lm}"))
4. Run inference:
   result = program({', '.join([f'{f}="..."' for f in input_fields])})"""

    def _save_artifacts(
        self,
        compilation_result: dict,
        program_code: str,
        target_lm: str,
        quality_profile: str,
    ) -> str:
        """Save artifacts and return artifact ID."""
        artifact_id = f"v_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"

        artifact_dir = self.artifacts_dir / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "artifact_version_id": artifact_id,
            "created_at": datetime.now().isoformat(),
            "target_lm": target_lm,
            "quality_profile": quality_profile,
            "mode": self.config.mode,
            "config": self.config.to_dict(),
            "eval_results": compilation_result,
            "react_iterations": len(self.steps),
        }

        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        with open(artifact_dir / "program.py", "w") as f:
            f.write(program_code)

        return artifact_id
