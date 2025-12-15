"""MCP-based LangChain Agent for DSPy Optimization.

Uses MCP server tools with LLM for intelligent decision making.
"""

import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

# Import MCP server tools directly (in-process)
from mcp_client import InProcessMCPServer

from prompts.templates import render_prompt
from utils.settings import get_settings


# Lazy imports for LLM providers
def _get_chat_model(
    provider: str, model_name: str, temperature: float = 0.1
) -> BaseChatModel:
    """Get chat model for agent."""
    settings = get_settings()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=settings.endpoints.ollama_base_url,
        )
    elif provider == "google" or provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@dataclass
class AgentStep:
    """Represents a step in agent execution."""

    id: str
    name: str
    tool: str
    status: str
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DSPyMCPAgent:
    """LangChain Agent that orchestrates DSPy via MCP server tools.

    Uses LLM for:
    1. Analyzing business task
    2. Making decisions about configuration

    Uses MCP tools for:
    1. Creating DSPy signatures
    2. Creating modules
    3. Configuring LM
    4. Preparing datasets
    5. Running optimization
    6. Evaluating and exporting
    """

    def __init__(self, agent_model: str | None = None):
        settings = get_settings()

        resolved_agent_model = (
            agent_model or f"openai/{settings.model_defaults.openai_chat}"
        )
        # Parse model
        if "/" in resolved_agent_model:
            provider, model = resolved_agent_model.split("/", 1)
        else:
            provider = "openai"
            model = resolved_agent_model

        self.provider = provider
        self.model = model
        self.agent_model = resolved_agent_model

        # Initialize LLM
        try:
            self.llm = _get_chat_model(
                provider, model, temperature=settings.agent.temperature
            )
            self.llm_available = True
        except Exception as e:
            print(f"[MCP Agent] LLM init failed: {e}, using rule-based analysis")
            self.llm = None
            self.llm_available = False

        # Initialize MCP server (in-process)
        self.mcp = InProcessMCPServer()

        self.steps: list[AgentStep] = []
        self.step_counter = 0

    def _add_step(self, name: str, tool: str, status: str, **kwargs) -> AgentStep:
        """Add execution step."""
        self.step_counter += 1
        step = AgentStep(
            id=f"step_{self.step_counter}",
            name=name,
            tool=tool,
            status=status,
            **kwargs,
        )
        self.steps.append(step)
        return step

    def _analyze_task_rules(self, task: str) -> dict:
        """Rule-based task analysis (fallback if LLM unavailable)."""
        task_lower = task.lower()

        # Task type detection
        if any(
            w in task_lower
            for w in ["classify", "categorize", "label", "intent", "sentiment"]
        ):
            task_type = "classification"
            input_field = "text"
            output_field = "category"
        elif any(w in task_lower for w in ["extract", "ner", "entity"]):
            task_type = "extraction"
            input_field = "text"
            output_field = "entities"
        elif any(w in task_lower for w in ["summarize", "summary"]):
            task_type = "summarization"
            input_field = "document"
            output_field = "summary"
        elif any(w in task_lower for w in ["question", "answer", "qa"]):
            task_type = "question_answering"
            input_field = "question"
            output_field = "answer"
        else:
            task_type = "general"
            input_field = "input"
            output_field = "output"

        # Complexity
        word_count = len(task.split())
        if word_count > 50 or "complex" in task_lower or "multi" in task_lower:
            complexity = "high"
        elif word_count > 20:
            complexity = "medium"
        else:
            complexity = "low"

        needs_cot = complexity in ["medium", "high"] or task_type in [
            "question_answering",
            "extraction",
        ]

        # Select metric based on task type
        if task_type == "classification":
            metric_type = "exact_match"
        elif task_type in ["extraction", "summarization", "generation"]:
            metric_type = "contains"
        elif task_type == "question_answering":
            metric_type = "contains"  # QA often has multiple valid phrasings
        else:
            metric_type = "contains"

        return {
            "task_type": task_type,
            "complexity": complexity,
            "needs_cot": needs_cot,
            "input_field": input_field,
            "output_field": output_field,
            "metric_type": metric_type,
            "reasoning": "Rule-based analysis",
        }

    async def _analyze_task_llm(self, task: str) -> dict:
        """LLM-based task analysis."""
        if not self.llm_available:
            return self._analyze_task_rules(task)

        try:
            prompt = render_prompt(
                "analysis_prompts", "mcp_agent_analyze_task", task=task
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            print(f"[MCP Agent] LLM analysis failed: {e}")

        return self._analyze_task_rules(task)

    async def run_async(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str = "BALANCED",
    ) -> AsyncGenerator[dict, None]:
        """Run DSPy optimization workflow via MCP tools.
        Yields steps as they complete.
        """
        self.steps = []
        self.step_counter = 0

        # Reset MCP session
        self.mcp.reset()

        print("[MCP Agent] Starting optimization workflow")
        print(f"[MCP Agent] Task: {business_task[:100]}...")
        print(f"[MCP Agent] Target LM: {target_lm}")
        print(f"[MCP Agent] Dataset size: {len(dataset)}")

        try:
            # ========== Step 1: Analyze Task ==========
            step = self._add_step(
                "Analyze Business Task",
                "analyze_task",
                "running",
                thought="Using LLM to analyze task requirements...",
            )
            yield {"type": "step", "data": step.__dict__}

            analysis = await self._analyze_task_llm(business_task)

            step.status = "success"
            step.observation = f"task_type={analysis['task_type']}, complexity={analysis['complexity']}, needs_cot={analysis['needs_cot']}"
            yield {"type": "step", "data": step.__dict__}

            print(f"[MCP Agent] Analysis: {analysis}")
            print(
                f"[MCP Agent] Selected metric: {analysis.get('metric_type', 'exact_match')}"
            )

            # ========== Step 2: Create Signature via MCP ==========
            step = self._add_step(
                "Create DSPy Signature",
                "create_signature",
                "running",
                thought=f"Creating signature with {analysis['input_field']} -> {analysis['output_field']}",
            )
            yield {"type": "step", "data": step.__dict__}

            sig_result = await self.mcp.call_tool(
                "create_signature",
                {
                    "input_fields": [analysis["input_field"]],
                    "output_fields": [analysis["output_field"]],
                    "task_description": business_task,
                },
            )

            step.status = "success" if "error" not in sig_result else "error"
            step.observation = sig_result.get("signature_code", str(sig_result))[:200]
            yield {"type": "step", "data": step.__dict__}

            if "error" in sig_result:
                yield {"type": "error", "error": sig_result["error"]}
                return

            # ========== Step 3: Create Module via MCP ==========
            module_type = "ChainOfThought" if analysis["needs_cot"] else "Predict"

            step = self._add_step(
                f"Create {module_type} Module",
                "create_module",
                "running",
                thought=f"Creating {module_type} module based on complexity analysis",
            )
            yield {"type": "step", "data": step.__dict__}

            mod_result = await self.mcp.call_tool(
                "create_module",
                {"module_type": module_type, "use_cot": analysis["needs_cot"]},
            )

            step.status = "success" if "error" not in mod_result else "error"
            step.observation = (
                f"Created {mod_result.get('module_type', 'unknown')} module"
            )
            yield {"type": "step", "data": step.__dict__}

            # ========== Step 4: Configure LM via MCP ==========
            step = self._add_step(
                "Configure Language Model",
                "configure_lm",
                "running",
                thought=f"Configuring {target_lm} as target model",
            )
            yield {"type": "step", "data": step.__dict__}

            lm_result = await self.mcp.call_tool(
                "configure_lm", {"model_name": target_lm}
            )

            step.status = "success" if "error" not in lm_result else "error"
            step.observation = f"Configured {lm_result.get('model', target_lm)}"
            yield {"type": "step", "data": step.__dict__}

            if "error" in lm_result:
                yield {"type": "error", "error": lm_result["error"]}
                return

            # ========== Step 5: Prepare Dataset via MCP ==========
            step = self._add_step(
                "Prepare Dataset",
                "prepare_dataset",
                "running",
                thought=f"Splitting {len(dataset)} examples into train/dev/test",
            )
            yield {"type": "step", "data": step.__dict__}

            data_result = await self.mcp.call_tool(
                "prepare_dataset",
                {"dataset": dataset, "train_ratio": 0.7, "dev_ratio": 0.2},
            )

            step.status = "success" if "error" not in data_result else "error"
            step.observation = f"train={data_result.get('train_size', 0)}, dev={data_result.get('dev_size', 0)}, test={data_result.get('test_size', 0)}"
            yield {"type": "step", "data": step.__dict__}

            # ========== Step 6: Select Optimizer via MCP ==========
            # Choose optimizer and parameters based on dataset size, complexity, and quality profile
            train_size = data_result.get("train_size", 5)
            complexity = analysis["complexity"]

            # Dynamic optimizer parameters based on dataset size
            # max_bootstrapped_demos: should not exceed train_size, typically 2-8
            # max_labeled_demos: should not exceed train_size, typically 2-8
            if train_size <= 5:
                max_bootstrapped = min(2, train_size)
                max_labeled = min(2, train_size)
            elif train_size <= 10:
                max_bootstrapped = min(3, train_size)
                max_labeled = min(3, train_size)
            elif train_size <= 20:
                max_bootstrapped = min(4, train_size)
                max_labeled = min(4, train_size)
            else:
                max_bootstrapped = min(6, train_size)
                max_labeled = min(6, train_size)

            # Adjust based on complexity
            if complexity == "high":
                max_bootstrapped = min(max_bootstrapped + 2, train_size, 8)
                max_labeled = min(max_labeled + 2, train_size, 8)

            # Choose optimizer type and num_candidates based on quality profile
            if quality_profile == "FAST_CHEAP":
                optimizer_type = "BootstrapFewShot"
                num_candidates = 3
            elif quality_profile == "HIGH_QUALITY" or complexity == "high":
                optimizer_type = "BootstrapFewShotWithRandomSearch"
                num_candidates = min(15, max(5, train_size))
            else:  # BALANCED
                optimizer_type = "BootstrapFewShotWithRandomSearch"
                num_candidates = min(10, max(5, train_size))

            step = self._add_step(
                f"Select {optimizer_type}",
                "select_optimizer",
                "running",
                thought=f"Choosing {optimizer_type} based on {quality_profile} profile, {complexity} complexity, {train_size} train examples",
            )
            yield {"type": "step", "data": step.__dict__}

            opt_result = await self.mcp.call_tool(
                "select_optimizer",
                {
                    "optimizer_type": optimizer_type,
                    "max_bootstrapped_demos": max_bootstrapped,
                    "max_labeled_demos": max_labeled,
                    "num_candidate_programs": num_candidates,
                },
            )

            print(
                f"[MCP Agent] Optimizer params: max_bootstrapped={max_bootstrapped}, max_labeled={max_labeled}, num_candidates={num_candidates}"
            )

            step.status = "success"
            step.observation = f"Selected {optimizer_type}: {max_bootstrapped} bootstrapped, {max_labeled} labeled, {num_candidates} candidates"
            yield {"type": "step", "data": step.__dict__}

            # ========== Step 7: Run Optimization via MCP ==========
            # Use metric_type determined by agent analysis
            metric_type = analysis.get("metric_type", "exact_match")

            step = self._add_step(
                "Run DSPy Optimization",
                "run_optimization",
                "running",
                thought=f"Running DSPy compilation with {optimizer_type} and {metric_type} metric...",
            )
            yield {"type": "step", "data": step.__dict__}
            compile_result = await self.mcp.call_tool(
                "run_optimization", {"metric_type": metric_type}
            )

            step.status = "success" if "error" not in compile_result else "error"
            if "error" in compile_result:
                step.observation = compile_result["error"]
                step.error = compile_result["error"]
            else:
                dev_acc = compile_result.get("dev_accuracy", 0)
                step.observation = f"Optimization complete. Dev accuracy: {dev_acc:.1%}"
            yield {"type": "step", "data": step.__dict__}

            if "error" in compile_result:
                yield {"type": "error", "error": compile_result["error"]}
                return

            # ========== Step 8: Evaluate Program via MCP ==========
            step = self._add_step(
                "Evaluate on Test Set",
                "evaluate_program",
                "running",
                thought="Evaluating compiled program on held-out test set",
            )
            yield {"type": "step", "data": step.__dict__}

            eval_result = await self.mcp.call_tool(
                "evaluate_program",
                {"num_samples": min(50, data_result.get("test_size", 10))},
            )

            step.status = "success" if "error" not in eval_result else "error"
            if "error" not in eval_result:
                test_acc = eval_result.get("test_accuracy", 0)
                step.observation = f"Test accuracy: {test_acc:.1%} ({eval_result.get('correct', 0)}/{eval_result.get('total', 0)})"
            yield {"type": "step", "data": step.__dict__}

            # ========== Step 9: Export Program via MCP ==========
            step = self._add_step(
                "Export Optimized Program",
                "export_program",
                "running",
                thought="Exporting optimized program and saving artifacts",
            )
            yield {"type": "step", "data": step.__dict__}

            export_result = await self.mcp.call_tool(
                "export_program",
                {
                    "artifact_name": f"dspy_{analysis['task_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                },
            )

            step.status = "success" if "error" not in export_result else "error"
            step.observation = (
                f"Exported to {export_result.get('artifact_id', 'unknown')}"
            )
            yield {"type": "step", "data": step.__dict__}

            # ========== Final Result ==========
            final_accuracy = eval_result.get(
                "test_accuracy", compile_result.get("dev_accuracy", 0)
            )

            final_result = {
                "artifact_version_id": export_result.get("artifact_id", ""),
                "compiled_program_id": export_result.get("artifact_id", ""),
                "signature_id": f"sig_{analysis['task_type']}",
                "eval_results": {
                    "metric_name": "exact_match",
                    "metric_value": final_accuracy,
                    "dev_accuracy": compile_result.get("dev_accuracy", 0),
                    "test_accuracy": eval_result.get("test_accuracy", 0),
                    "real_dspy": True,
                },
                "task_analysis": {
                    "task_type": analysis.get("task_type", "general"),
                    "domain": analysis.get("domain", "general"),
                    "complexity_level": analysis.get("complexity", "medium"),
                    "input_roles": [analysis.get("input_field", "input")],
                    "output_roles": [analysis.get("output_field", "output")],
                    "needs_retrieval": False,
                    "needs_chain_of_thought": analysis.get("needs_cot", False),
                    "safety_level": "standard",
                },
                "program_code": export_result.get("program_code", ""),
                "optimizer_type": optimizer_type,
                "module_type": module_type,
                "react_iterations": self.step_counter,
                "steps_completed": self.step_counter,
            }

            print(
                f"[MCP Agent] Optimization complete! Test accuracy: {final_accuracy:.1%}"
            )
            yield {"type": "complete", "data": final_result}

        except Exception as e:
            print(f"[MCP Agent] Error: {e}")
            import traceback

            traceback.print_exc()
            yield {"type": "error", "error": str(e)}
