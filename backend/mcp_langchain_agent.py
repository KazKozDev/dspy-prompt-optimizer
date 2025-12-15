"""MCP-based LangChain Agent for DSPy Optimization.

This agent uses MCP tools to configure and run DSPy optimization.
It makes intelligent decisions about how to set up DSPy based on the task.
"""

import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prompts.templates import get_prompt
from utils.settings import get_settings

# Lazy imports for LLM providers
ChatOpenAI = None
ChatAnthropic = None
ChatOllama = None


def _get_chat_model(
    provider: str, model_name: str, temperature: float = 0.2
) -> BaseChatModel:
    """Get chat model for agent."""
    global ChatOpenAI, ChatAnthropic, ChatOllama

    settings = get_settings()

    if provider == "openai":
        if ChatOpenAI is None:
            from langchain_openai import ChatOpenAI as _ChatOpenAI

            ChatOpenAI = _ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        if ChatAnthropic is None:
            from langchain_anthropic import ChatAnthropic as _ChatAnthropic

            ChatAnthropic = _ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "ollama":
        if ChatOllama is None:
            from langchain_ollama import ChatOllama as _ChatOllama

            ChatOllama = _ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=settings.endpoints.ollama_base_url,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


@dataclass
class AgentStep:
    """Represents a step in agent execution."""

    id: str
    name: str
    tool: str
    status: str  # pending, running, success, error
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: int | None = None


class DSPyMCPTools:
    """Direct implementation of DSPy tools (no MCP server needed).
    This provides the same interface as MCP but runs in-process.
    """

    def __init__(self):
        self.dspy = None
        self.session = {
            "task_analysis": None,
            "signature_class": None,
            "signature_code": None,
            "module_type": "Predict",
            "module_instance": None,
            "dataset": [],
            "train_set": [],
            "dev_set": [],
            "test_set": [],
            "optimizer_type": "BootstrapFewShot",
            "optimizer_config": {},
            "compiled_program": None,
            "eval_results": {},
            "target_lm": None,
            "lm_instance": None,
            "program_code": None,
        }

    def _load_dspy(self):
        if self.dspy is None:
            import dspy

            self.dspy = dspy
            if hasattr(dspy, "configure_cache"):
                dspy.configure_cache(False)
        return self.dspy

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Call a DSPy tool."""
        try:
            if name == "analyze_task":
                return await self._analyze_task(arguments.get("task_description", ""))
            elif name == "create_signature":
                return await self._create_signature(
                    arguments.get("input_fields", []),
                    arguments.get("output_fields", []),
                    arguments.get("task_description", ""),
                )
            elif name == "create_module":
                return await self._create_module(
                    arguments.get("module_type", "Predict"),
                    arguments.get("use_cot", False),
                )
            elif name == "configure_lm":
                return await self._configure_lm(arguments.get("model_name", ""))
            elif name == "prepare_dataset":
                return await self._prepare_dataset(
                    arguments.get("dataset", []),
                    arguments.get("train_ratio", 0.7),
                    arguments.get("dev_ratio", 0.2),
                )
            elif name == "select_optimizer":
                return await self._select_optimizer(
                    arguments.get("optimizer_type", "BootstrapFewShot"),
                    arguments.get("max_bootstrapped_demos", 4),
                    arguments.get("max_labeled_demos", 4),
                    arguments.get("num_candidate_programs", 10),
                )
            elif name == "run_optimization":
                return await self._run_optimization(
                    arguments.get("metric_type", "exact_match")
                )
            elif name == "evaluate_program":
                return await self._evaluate_program(arguments.get("num_samples", 50))
            elif name == "export_program":
                return await self._export_program(arguments.get("artifact_name", ""))
            elif name == "get_session_status":
                return self._get_session_status()
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}

    async def _analyze_task(self, task_description: str) -> dict:
        """Analyze business task."""
        task_lower = task_description.lower()

        # Determine task type
        if any(
            w in task_lower
            for w in ["classify", "categorize", "label", "intent", "sentiment"]
        ):
            task_type = "classification"
        elif any(w in task_lower for w in ["extract", "ner", "entity"]):
            task_type = "extraction"
        elif any(w in task_lower for w in ["summarize", "summary"]):
            task_type = "summarization"
        elif any(w in task_lower for w in ["question", "answer", "qa"]):
            task_type = "question_answering"
        else:
            task_type = "general"

        # Determine domain
        if any(w in task_lower for w in ["customer", "support", "service"]):
            domain = "customer_support"
        elif any(w in task_lower for w in ["legal", "law"]):
            domain = "legal"
        elif any(w in task_lower for w in ["medical", "health"]):
            domain = "medical"
        else:
            domain = "general"

        # Complexity
        word_count = len(task_description.split())
        complexity = (
            "high" if word_count > 50 else "medium" if word_count > 20 else "low"
        )

        needs_cot = complexity in ["medium", "high"]

        if task_type == "classification":
            input_fields = ["text"]
            output_fields = ["category"]
        elif task_type == "question_answering":
            input_fields = ["question", "context"]
            output_fields = ["answer"]
        else:
            input_fields = ["input"]
            output_fields = ["output"]

        analysis = {
            "task_type": task_type,
            "domain": domain,
            "complexity": complexity,
            "needs_chain_of_thought": needs_cot,
            "recommended_input_fields": input_fields,
            "recommended_output_fields": output_fields,
            "recommended_optimizer": (
                "BootstrapFewShotWithRandomSearch"
                if complexity != "low"
                else "BootstrapFewShot"
            ),
            "recommended_module": "ChainOfThought" if needs_cot else "Predict",
        }

        self.session["task_analysis"] = analysis
        return {"status": "success", "analysis": analysis}

    async def _create_signature(
        self,
        input_fields: list[str],
        output_fields: list[str],
        task_description: str = "",
    ) -> dict:
        """Create DSPy Signature."""
        dspy = self._load_dspy()

        sig_fields = {}
        for field in input_fields:
            sig_fields[field] = dspy.InputField(desc=f"Input: {field}")
        for field in output_fields:
            sig_fields[field] = dspy.OutputField(desc=f"Output: {field}")

        doc = (
            task_description
            or f"Process {', '.join(input_fields)} to produce {', '.join(output_fields)}"
        )

        SignatureClass = type(
            "DynamicSignature", (dspy.Signature,), {"__doc__": doc, **sig_fields}
        )

        self.session["signature_class"] = SignatureClass

        code = f'''class TaskSignature(dspy.Signature):
    """{doc}"""
'''
        for field in input_fields:
            code += f'    {field} = dspy.InputField(desc="Input: {field}")\n'
        for field in output_fields:
            code += f'    {field} = dspy.OutputField(desc="Output: {field}")\n'

        self.session["signature_code"] = code

        return {
            "status": "success",
            "signature_code": code,
            "input_fields": input_fields,
            "output_fields": output_fields,
        }

    async def _create_module(self, module_type: str, use_cot: bool = False) -> dict:
        """Create DSPy module."""
        dspy = self._load_dspy()

        if self.session["signature_class"] is None:
            return {"error": "No signature defined. Call create_signature first."}

        self.session["module_type"] = module_type

        if module_type == "ChainOfThought" or use_cot:
            self.session["module_instance"] = dspy.ChainOfThought(
                self.session["signature_class"]
            )
            actual_type = "ChainOfThought"
        elif module_type == "ReAct":
            self.session["module_instance"] = dspy.ReAct(
                self.session["signature_class"]
            )
            actual_type = "ReAct"
        else:
            self.session["module_instance"] = dspy.Predict(
                self.session["signature_class"]
            )
            actual_type = "Predict"

        return {"status": "success", "module_type": actual_type}

    async def _configure_lm(self, model_name: str) -> dict:
        """Configure Language Model."""
        dspy = self._load_dspy()

        if "/" in model_name:
            provider, model = model_name.split("/", 1)
        else:
            provider = "openai"
            model = model_name

        try:
            if provider == "ollama":
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                lm = dspy.LM(f"ollama_chat/{model}", api_base=base_url)
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                lm = dspy.LM(f"anthropic/{model}", api_key=api_key)
            elif provider == "google" or provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                lm = dspy.LM(f"google/{model}", api_key=api_key)
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                lm = dspy.LM(f"openai/{model}", api_key=api_key)

            dspy.configure(lm=lm)
            self.session["lm_instance"] = lm
            self.session["target_lm"] = model_name

            return {"status": "success", "model": model_name, "provider": provider}
        except Exception as e:
            return {"error": f"Failed to configure LM: {str(e)}"}

    async def _prepare_dataset(
        self, dataset: list[dict], train_ratio: float = 0.7, dev_ratio: float = 0.2
    ) -> dict:
        """Prepare dataset splits."""
        dspy = self._load_dspy()

        if not dataset:
            return {"error": "Empty dataset"}

        self.session["dataset"] = dataset

        if self.session["task_analysis"]:
            input_fields = self.session["task_analysis"].get(
                "recommended_input_fields", ["input"]
            )
            output_fields = self.session["task_analysis"].get(
                "recommended_output_fields", ["output"]
            )
        else:
            input_fields = ["input"]
            output_fields = ["output"]

        examples = []
        for item in dataset:
            ex_dict = {}
            if "input" in item:
                ex_dict[input_fields[0]] = item["input"]
            if "output" in item:
                ex_dict[output_fields[0]] = item["output"]
            example = dspy.Example(**ex_dict).with_inputs(*input_fields)
            examples.append(example)

        n = len(examples)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))

        self.session["train_set"] = examples[:train_end]
        self.session["dev_set"] = examples[train_end:dev_end]
        self.session["test_set"] = examples[dev_end:]

        return {
            "status": "success",
            "total_examples": n,
            "train_size": len(self.session["train_set"]),
            "dev_size": len(self.session["dev_set"]),
            "test_size": len(self.session["test_set"]),
        }

    async def _select_optimizer(
        self,
        optimizer_type: str,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        num_candidate_programs: int = 10,
    ) -> dict:
        """Select optimizer."""
        self.session["optimizer_type"] = optimizer_type
        self.session["optimizer_config"] = {
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
            "num_candidate_programs": num_candidate_programs,
        }
        return {
            "status": "success",
            "optimizer_type": optimizer_type,
            "config": self.session["optimizer_config"],
        }

    async def _run_optimization(self, metric_type: str = "exact_match") -> dict:
        """Run DSPy optimization."""
        dspy = self._load_dspy()

        if self.session["module_instance"] is None:
            return {"error": "No module created"}
        if not self.session["train_set"]:
            return {"error": "No training data"}
        if self.session["lm_instance"] is None:
            return {"error": "No LM configured"}

        if self.session["task_analysis"]:
            output_field = self.session["task_analysis"].get(
                "recommended_output_fields", ["output"]
            )[0]
        else:
            output_field = "output"

        def exact_match_metric(example, pred, trace=None):
            expected = getattr(example, output_field, "").strip().lower()
            predicted = getattr(pred, output_field, "").strip().lower()
            return 1.0 if expected == predicted else 0.0

        from dspy import teleprompt as tp

        optimizer_type = self.session["optimizer_type"]
        config = self.session["optimizer_config"]

        try:
            if optimizer_type == "BootstrapFewShot":
                optimizer = tp.BootstrapFewShot(
                    metric=exact_match_metric,
                    max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
                    max_labeled_demos=config.get("max_labeled_demos", 4),
                )
            elif optimizer_type == "BootstrapFewShotWithRandomSearch":
                optimizer = tp.BootstrapFewShotWithRandomSearch(
                    metric=exact_match_metric,
                    max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
                    max_labeled_demos=config.get("max_labeled_demos", 4),
                    num_candidate_programs=config.get("num_candidate_programs", 10),
                )
            else:
                optimizer = tp.BootstrapFewShot(metric=exact_match_metric)

            compiled = optimizer.compile(
                self.session["module_instance"], trainset=self.session["train_set"]
            )

            self.session["compiled_program"] = compiled

            # Quick dev evaluation
            if self.session["dev_set"]:
                correct = 0
                total = min(len(self.session["dev_set"]), 20)
                for ex in self.session["dev_set"][:total]:
                    try:
                        input_field = list(ex.inputs().keys())[0]
                        pred = compiled(**{input_field: getattr(ex, input_field)})
                        if exact_match_metric(ex, pred) > 0.5:
                            correct += 1
                    except:
                        pass
                dev_accuracy = correct / total if total > 0 else 0
            else:
                dev_accuracy = 0

            self.session["eval_results"] = {
                "optimizer_type": optimizer_type,
                "dev_accuracy": dev_accuracy,
                "train_size": len(self.session["train_set"]),
            }

            return {
                "status": "success",
                "optimizer_type": optimizer_type,
                "dev_accuracy": dev_accuracy,
                "message": f"Optimization complete. Dev accuracy: {dev_accuracy:.1%}",
            }

        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}

    async def _evaluate_program(self, num_samples: int = 50) -> dict:
        """Evaluate on test set."""
        if self.session["compiled_program"] is None:
            return {"error": "No compiled program"}

        if not self.session["test_set"]:
            return {"error": "No test data"}

        if self.session["task_analysis"]:
            output_field = self.session["task_analysis"].get(
                "recommended_output_fields", ["output"]
            )[0]
        else:
            output_field = "output"

        correct = 0
        total = min(len(self.session["test_set"]), num_samples)

        for ex in self.session["test_set"][:total]:
            try:
                input_field = list(ex.inputs().keys())[0]
                pred = self.session["compiled_program"](
                    **{input_field: getattr(ex, input_field)}
                )
                expected = getattr(ex, output_field, "").strip().lower()
                predicted = getattr(pred, output_field, "").strip().lower()
                if expected == predicted:
                    correct += 1
            except:
                pass

        accuracy = correct / total if total > 0 else 0
        self.session["eval_results"]["test_accuracy"] = accuracy

        return {
            "status": "success",
            "test_accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    async def _export_program(self, artifact_name: str = "") -> dict:
        """Export program."""
        if self.session["compiled_program"] is None:
            return {"error": "No compiled program"}

        import uuid

        artifact_id = (
            artifact_name
            or f"dspy_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        code = f'''"""
DSPy Optimized Program
Generated: {datetime.now().isoformat()}
Optimizer: {self.session["optimizer_type"]}
"""

import dspy

# Signature
{self.session["signature_code"] or "# Signature not available"}

# Module
class OptimizedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.{self.session["module_type"]}(TaskSignature)
    
    def forward(self, **kwargs):
        return self.predictor(**kwargs)
'''

        self.session["program_code"] = code

        from pathlib import Path

        artifact_dir = Path("data/artifacts") / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with open(artifact_dir / "program.py", "w") as f:
            f.write(code)

        metadata = {
            "artifact_id": artifact_id,
            "created_at": datetime.now().isoformat(),
            "task_analysis": self.session["task_analysis"],
            "optimizer_type": self.session["optimizer_type"],
            "eval_results": self.session["eval_results"],
            "target_lm": self.session["target_lm"],
        }

        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {"status": "success", "artifact_id": artifact_id, "program_code": code}

    def _get_session_status(self) -> dict:
        """Get session status."""
        return {
            "task_analysis": self.session["task_analysis"],
            "signature_defined": self.session["signature_class"] is not None,
            "module_type": (
                self.session["module_type"] if self.session["module_instance"] else None
            ),
            "lm_configured": self.session["target_lm"],
            "dataset_size": len(self.session["dataset"]),
            "train_size": len(self.session["train_set"]),
            "optimizer_type": self.session["optimizer_type"],
            "compiled": self.session["compiled_program"] is not None,
            "eval_results": self.session["eval_results"],
        }


class MCPLangChainAgent:
    """LangChain Agent that uses DSPy tools via MCP-like interface.
    Uses a structured workflow with LLM for analysis decisions.
    """

    def __init__(self, agent_model: str | None = None):
        settings = get_settings()

        resolved_agent_model = (
            agent_model or f"openai/{settings.model_defaults.openai_chat}"
        )
        if "/" in resolved_agent_model:
            provider, model = resolved_agent_model.split("/", 1)
        else:
            provider = "openai"
            model = resolved_agent_model

        self.system_prompt = get_prompt("system_prompts", "default")
        self.llm = _get_chat_model(
            provider, model, temperature=settings.agent.temperature
        )
        self.tools = DSPyMCPTools()
        self.steps: list[AgentStep] = []
        self.step_counter = 0
        self.agent_model = resolved_agent_model

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

    def _parse_tool_call(self, response: str) -> dict | None:
        """Parse tool call from LLM response."""
        try:
            # Try to find JSON in response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                data = json.loads(json_str)
                if "tool" in data:
                    return data
        except:
            pass
        return None

    async def run_async(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str = "BALANCED",
    ) -> AsyncGenerator[dict, None]:
        """Run agent asynchronously, yielding steps as they complete."""
        self.steps = []
        self.step_counter = 0

        # Reset tools session
        self.tools.session = {
            "task_analysis": None,
            "signature_class": None,
            "signature_code": None,
            "module_type": "Predict",
            "module_instance": None,
            "dataset": [],
            "train_set": [],
            "dev_set": [],
            "test_set": [],
            "optimizer_type": "BootstrapFewShot",
            "optimizer_config": {},
            "compiled_program": None,
            "eval_results": {},
            "target_lm": None,
            "lm_instance": None,
            "program_code": None,
        }

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"""Configure and optimize DSPy for this task:

Business Task: {business_task}
Target Model: {target_lm}
Dataset Size: {len(dataset)} examples
Quality Profile: {quality_profile}

Start by analyzing the task, then proceed through the optimization workflow."""
            ),
        ]

        max_iterations = 15
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get LLM response
            step = self._add_step(
                f"Agent Thinking (Step {iteration})",
                "llm",
                "running",
                thought="Analyzing situation and deciding next action...",
            )
            yield {"type": "step", "data": step.__dict__}

            try:
                response = self.llm.invoke(messages)
                response_text = response.content

                step.status = "success"
                step.observation = (
                    response_text[:200] + "..."
                    if len(response_text) > 200
                    else response_text
                )
                yield {"type": "step", "data": step.__dict__}

                messages.append(AIMessage(content=response_text))

                # Check for tool call
                tool_call = self._parse_tool_call(response_text)

                if tool_call:
                    tool_name = tool_call.get("tool")
                    tool_args = tool_call.get("arguments", {})

                    # Inject dataset and target_lm for relevant tools
                    if tool_name == "prepare_dataset" and "dataset" not in tool_args:
                        tool_args["dataset"] = dataset
                    if tool_name == "configure_lm" and "model_name" not in tool_args:
                        tool_args["model_name"] = target_lm

                    # Execute tool
                    tool_step = self._add_step(
                        f"Execute: {tool_name}",
                        tool_name,
                        "running",
                        action=f"{tool_name}({json.dumps(tool_args)[:100]}...)",
                    )
                    yield {"type": "step", "data": tool_step.__dict__}

                    result = await self.tools.call_tool(tool_name, tool_args)

                    tool_step.status = "success" if "error" not in result else "error"
                    tool_step.observation = json.dumps(result)[:500]
                    yield {"type": "step", "data": tool_step.__dict__}

                    # Add result to messages
                    messages.append(
                        HumanMessage(content=f"Tool result: {json.dumps(result)}")
                    )

                    # Check if we're done (export_program was called)
                    if tool_name == "export_program" and "error" not in result:
                        # Final result
                        final_result = {
                            "artifact_version_id": result.get("artifact_id", ""),
                            "compiled_program_id": result.get("artifact_id", ""),
                            "signature_id": "sig_" + result.get("artifact_id", "")[:8],
                            "eval_results": {
                                "metric_name": "exact_match",
                                "metric_value": self.tools.session["eval_results"].get(
                                    "test_accuracy",
                                    self.tools.session["eval_results"].get(
                                        "dev_accuracy", 0
                                    ),
                                ),
                                "real_dspy": True,
                            },
                            "task_analysis": self.tools.session["task_analysis"] or {},
                            "program_code": result.get("program_code", ""),
                            "react_iterations": iteration,
                            "optimizer_type": self.tools.session["optimizer_type"],
                        }
                        yield {"type": "complete", "data": final_result}
                        return

                else:
                    # No tool call - check if agent is done or confused
                    if (
                        "complete" in response_text.lower()
                        or "finished" in response_text.lower()
                    ):
                        # Try to export if not done yet
                        if self.tools.session["compiled_program"] is not None:
                            result = await self.tools.call_tool("export_program", {})
                            final_result = {
                                "artifact_version_id": result.get("artifact_id", ""),
                                "compiled_program_id": result.get("artifact_id", ""),
                                "signature_id": "sig_default",
                                "eval_results": {
                                    "metric_name": "exact_match",
                                    "metric_value": self.tools.session[
                                        "eval_results"
                                    ].get(
                                        "test_accuracy",
                                        self.tools.session["eval_results"].get(
                                            "dev_accuracy", 0
                                        ),
                                    ),
                                    "real_dspy": True,
                                },
                                "task_analysis": self.tools.session["task_analysis"]
                                or {},
                                "program_code": result.get("program_code", ""),
                                "react_iterations": iteration,
                                "optimizer_type": self.tools.session["optimizer_type"],
                            }
                            yield {"type": "complete", "data": final_result}
                            return

                    # Prompt agent to continue
                    messages.append(
                        HumanMessage(
                            content='Please continue with the next step. Call a tool using JSON format: {"tool": "tool_name", "arguments": {...}}'
                        )
                    )

            except Exception as e:
                step.status = "error"
                step.error = str(e)
                yield {"type": "step", "data": step.__dict__}
                yield {"type": "error", "error": str(e)}
                return

        # Max iterations reached
        yield {
            "type": "error",
            "error": "Max iterations reached without completing optimization",
        }
