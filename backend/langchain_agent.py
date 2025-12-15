"""DSPy LangChain Agent Orchestrator.

LLM-Агент с ReAct логикой для автоматической оркестрации DSPy пайплайнов.
Поддерживает OpenAI, Anthropic и Ollama как "мозг" агента.
"""

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

# LangChain imports
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from prompts.templates import render_prompt
from utils.settings import get_settings

# Lazy imports for optional providers
ChatOpenAI = None
ChatAnthropic = None
ChatOllama = None


def _get_chat_model(
    provider: str,
    model_name: str,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Get appropriate chat model based on provider."""
    global ChatOpenAI, ChatAnthropic, ChatOllama

    settings = get_settings()

    if provider == "openai":
        if ChatOpenAI is None:
            from langchain_openai import ChatOpenAI as _ChatOpenAI

            ChatOpenAI = _ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        if ChatAnthropic is None:
            from langchain_anthropic import ChatAnthropic as _ChatAnthropic

            ChatAnthropic = _ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "ollama":
        if ChatOllama is None:
            from langchain_ollama import ChatOllama as _ChatOllama

            ChatOllama = _ChatOllama
        base_url = settings.endpoints.ollama_base_url
        return ChatOllama(model=model_name, temperature=temperature, base_url=base_url)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# DSPy imports - lazy loaded
dspy = None
BootstrapFewShot = None
BootstrapFewShotWithRandomSearch = None
MIPROv2 = None
COPRO = None


def _load_dspy():
    """Lazy load DSPy to avoid import conflicts."""
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

        # Disable caching to avoid SQLite lock issues
        if hasattr(_dspy, "configure_cache"):
            _dspy.configure_cache(False)

    return dspy, BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2, COPRO


# ==================== Tool Input Schemas ====================


class AnalyzeBusinessGoalInput(BaseModel):
    task_description: str = Field(description="Business task description to analyze")


class RegisterTargetLMInput(BaseModel):
    target_lm_name: str = Field(
        description="Name of target LLM (e.g., openai/gpt-5, ollama/llama3.1:8b)"
    )


class DefineSignatureInput(BaseModel):
    input_roles: list[str] = Field(description="List of input field names")
    output_roles: list[str] = Field(description="List of output field names")
    task_type: str = Field(
        description="Type of task (classification, extraction, etc.)"
    )
    domain: str = Field(description="Business domain (legal, finance, etc.)")


class AssemblePipelineInput(BaseModel):
    task_type: str = Field(description="Type of task")
    needs_retrieval: bool = Field(
        default=False, description="Whether task needs retrieval"
    )
    needs_chain_of_thought: bool = Field(
        default=False, description="Whether task needs CoT"
    )
    complexity_level: str = Field(
        default="medium", description="Complexity: low/medium/high"
    )


class PrepareDataInput(BaseModel):
    train_ratio: float = Field(default=0.7, description="Training data ratio")
    dev_ratio: float = Field(default=0.2, description="Dev data ratio")


class SelectOptimizerInput(BaseModel):
    task_type: str = Field(description="Type of task")
    complexity_level: str = Field(description="Complexity level")
    data_size: int = Field(description="Size of dataset")
    profile: str = Field(default="BALANCED", description="Quality profile")


class RunCompilationInput(BaseModel):
    program_id: str = Field(description="ID of program to compile")
    optimizer_type: str = Field(description="Type of optimizer to use")


# ==================== Agent State ====================


@dataclass
class AgentState:
    """State maintained by the agent during orchestration."""

    business_task: str = ""
    target_lm: str = ""
    task_analysis: dict | None = None
    signature_id: str | None = None
    signature_code: str | None = None
    program_id: str | None = None
    program_code: str | None = None
    program_spec: dict | None = None
    data_splits: dict | None = None
    optimizer_type: str | None = None
    optimizer_params: dict | None = None
    compilation_result: dict | None = None
    artifact_id: str | None = None
    steps: list[dict] = field(default_factory=list)
    dataset: list[dict] = field(default_factory=list)


# ==================== DSPy LangChain Agent ====================


class DSPyLangChainAgent:
    """LangChain-based DSPy Agent Orchestrator.

    Uses LLM as the agent brain with ReAct reasoning pattern.
    Supports OpenAI, Anthropic, and Ollama as agent providers.
    """

    def __init__(
        self,
        agent_model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        artifacts_dir: str = "data/artifacts",
        step_callback: Callable[[dict], None] | None = None,
    ):
        """Initialize the LangChain agent.

        Args:
            agent_model: Model for agent brain (format: provider/model)
            temperature: Sampling temperature
            max_iterations: Max ReAct iterations
            artifacts_dir: Directory to store artifacts
            step_callback: Optional callback called when each step completes
        """
        settings = get_settings()

        resolved_agent_model = (
            agent_model or f"openai/{settings.model_defaults.openai_chat}"
        )
        resolved_temperature = (
            settings.agent.temperature if temperature is None else temperature
        )
        resolved_max_iterations = (
            settings.agent.max_iterations if max_iterations is None else max_iterations
        )

        # Parse provider from model string
        if "/" in resolved_agent_model:
            self.provider, self.model_name = resolved_agent_model.split("/", 1)
        else:
            self.provider = "openai"
            self.model_name = resolved_agent_model

        self.temperature = resolved_temperature
        self.max_iterations = resolved_max_iterations
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.step_callback = step_callback

        # Initialize state
        self.state = AgentState()

        # Initialize LLM
        self.llm = _get_chat_model(
            self.provider, self.model_name, temperature=self.temperature
        )

        # Create tools
        self.tools = self._create_tools()

        # Create LangGraph ReAct agent
        try:
            from langgraph.prebuilt import create_react_agent

            self.agent = create_react_agent(self.llm, self.tools)
            self.use_langgraph = True
        except ImportError:
            # Fallback to manual tool execution
            self.agent = None
            self.use_langgraph = False
            print("LangGraph not available, using manual tool execution")

    def _add_step(self, name: str, tool: str, status: str, **kwargs):
        """Add a step to the execution history and notify via callback."""
        step = {
            "id": f"step_{len(self.state.steps) + 1}",
            "name": name,
            "tool": tool,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.state.steps.append(step)

        if self.step_callback:
            try:
                self.step_callback(step)
            except Exception as e:
                print(f"Step callback error: {e}")

        return step

    def _create_tools(self) -> list[StructuredTool]:
        """Create LangChain tools for the agent."""

        # Tool 1: Analyze Business Goal
        def analyze_business_goal(task_description: str) -> str:
            """Analyze business task and extract structured requirements."""
            self._add_step(
                "Analyze Business Goal",
                "analyze_business_goal",
                "running",
                thought="Analyzing the business task to extract requirements...",
            )

            analysis_prompt = render_prompt(
                "analysis_prompts",
                "langchain_agent_analyze_business_goal",
                task=task_description,
            )

            try:
                response = self.llm.invoke(analysis_prompt)
                content = response.content

                # Parse JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                analysis = json.loads(content.strip())
                self.state.task_analysis = analysis
                self.state.business_task = task_description

                self._add_step(
                    "Analyze Business Goal",
                    "analyze_business_goal",
                    "success",
                    action=f'analyze_business_goal("{task_description[:50]}...")',
                    observation=f'task_type="{analysis.get("task_type")}", complexity="{analysis.get("complexity_level")}"',
                )

                return json.dumps(analysis, indent=2)
            except Exception as e:
                self._add_step(
                    "Analyze Business Goal",
                    "analyze_business_goal",
                    "error",
                    error=str(e),
                )
                return f"Error: {str(e)}"

        # Tool 2: Register Target LM
        def register_target_lm(target_lm_name: str) -> str:
            """Register the target LLM for inference."""
            self._add_step(
                "Register Target LM",
                "register_target_lm",
                "running",
                thought=f"Registering {target_lm_name} as target model...",
            )

            self.state.target_lm = target_lm_name

            # Determine provider
            if "/" in target_lm_name:
                provider = target_lm_name.split("/")[0]
            else:
                provider = "openai"

            result = {
                "registered": True,
                "target_lm": target_lm_name,
                "provider": provider,
            }

            self._add_step(
                "Register Target LM",
                "register_target_lm",
                "success",
                action=f'register_target_lm("{target_lm_name}")',
                observation=f"registered=true, provider={provider}",
            )

            return json.dumps(result)

        # Tool 3: Define Signature
        def define_contract_signature(
            input_roles: list[str], output_roles: list[str], task_type: str, domain: str
        ) -> str:
            """Create DSPy Signature based on requirements."""
            self._add_step(
                "Define Contract Signature",
                "define_contract_signature",
                "running",
                thought="Creating DSPy Signature with input/output fields...",
            )

            signature_id = f"sig_{uuid.uuid4().hex[:8]}"
            class_name = f"{task_type.title().replace('_', '')}Signature"

            input_fields = "\n    ".join(
                [
                    f'{role}: str = dspy.InputField(desc="{role} for the task")'
                    for role in input_roles
                ]
            )

            output_fields = "\n    ".join(
                [
                    f'{role}: str = dspy.OutputField(desc="{role} from the model")'
                    for role in output_roles
                ]
            )

            signature_code = f'''class {class_name}(dspy.Signature):
    """{task_type} task in {domain} domain."""
    {input_fields}
    {output_fields}
'''

            self.state.signature_id = signature_id
            self.state.signature_code = signature_code

            result = {
                "signature_id": signature_id,
                "class_name": class_name,
                "input_roles": input_roles,
                "output_roles": output_roles,
            }

            self._add_step(
                "Define Contract Signature",
                "define_contract_signature",
                "success",
                action=f"define_contract_signature(inputs={input_roles}, outputs={output_roles})",
                observation=f'signature_id="{signature_id}" created',
            )

            return json.dumps(result)

        # Tool 4: Assemble Pipeline
        def assemble_program_pipeline(
            task_type: str,
            needs_retrieval: bool,
            needs_chain_of_thought: bool,
            complexity_level: str,
        ) -> str:
            """Assemble DSPy program pipeline."""
            self._add_step(
                "Assemble Program Pipeline",
                "assemble_program_pipeline",
                "running",
                thought=f"Building pipeline for {task_type} task...",
            )

            modules = []

            if needs_retrieval:
                modules.append(
                    {"name": "Retriever", "type": "dspy.Retrieve", "params": {"k": 5}}
                )

            if needs_chain_of_thought:
                modules.append({"name": "MainPredictor", "type": "dspy.ChainOfThought"})
            else:
                modules.append({"name": "MainPredictor", "type": "dspy.Predict"})

            program_spec = {
                "modules": modules,
                "task_type": task_type,
                "complexity": complexity_level,
            }

            self.state.program_spec = program_spec
            self.state.program_id = f"prog_{uuid.uuid4().hex[:8]}"

            module_names = [m["type"].split(".")[-1] for m in modules]

            self._add_step(
                "Assemble Program Pipeline",
                "assemble_program_pipeline",
                "success",
                action=f'assemble_program_pipeline(task_type="{task_type}", cot={needs_chain_of_thought})',
                observation=f'modules: [{", ".join(module_names)}]',
            )

            return json.dumps(program_spec)

        # Tool 5: Prepare Data
        def prepare_eval_splits(train_ratio: float, dev_ratio: float) -> str:
            """Prepare train/dev/test data splits."""
            self._add_step(
                "Prepare Eval Splits",
                "prepare_eval_splits",
                "running",
                thought="Splitting dataset for training...",
            )

            n = len(self.state.dataset)
            train_size = int(n * train_ratio)
            dev_size = int(n * dev_ratio)
            test_size = n - train_size - dev_size

            splits = {"train": train_size, "dev": dev_size, "test": test_size}

            self.state.data_splits = splits

            self._add_step(
                "Prepare Eval Splits",
                "prepare_eval_splits",
                "success",
                action=f"prepare_eval_splits(train={train_ratio}, dev={dev_ratio})",
                observation=f"train={train_size}, dev={dev_size}, test={test_size}",
            )

            return json.dumps(splits)

        # Tool 6: Select Optimizer
        def select_compiler_strategy(
            task_type: str, complexity_level: str, data_size: int, profile: str
        ) -> str:
            """Select DSPy optimizer strategy."""
            self._add_step(
                "Select Compiler Strategy",
                "select_compiler_strategy",
                "running",
                thought=f"Selecting optimizer for {data_size} examples with {profile} profile...",
            )

            profile = profile or "BALANCED"
            if profile == "FAST_CHEAP" or profile == "Fast":
                optimizer_type = "BootstrapFewShot"
                params = {
                    "max_bootstrapped_demos": 2,
                    "max_labeled_demos": 4,
                    "max_rounds": 1,
                }
            elif profile == "HIGH_QUALITY" or profile == "Quality":
                if data_size >= 30:
                    optimizer_type = "MIPROv2"
                    params = {
                        "max_bootstrapped_demos": 4,
                        "max_labeled_demos": 4,
                        "auto": "heavy",
                        "num_candidates": 16,
                    }
                else:
                    optimizer_type = "BootstrapFewShotWithRandomSearch"
                    params = {
                        "max_bootstrapped_demos": 3,
                        "max_labeled_demos": 8,
                        "max_rounds": 1,
                        "num_candidate_programs": 8,
                    }
            else:
                # BALANCED
                optimizer_type = "BootstrapFewShotWithRandomSearch"
                params = {
                    "max_bootstrapped_demos": 3,
                    "max_labeled_demos": 8,
                    "max_rounds": 1,
                    "num_candidate_programs": 12,
                }

            self.state.optimizer_type = optimizer_type
            self.state.optimizer_params = params

            result = {"optimizer_type": optimizer_type, "params": params}

            self._add_step(
                "Select Compiler Strategy",
                "select_compiler_strategy",
                "success",
                action=f'select_compiler_strategy(profile="{profile}", size={data_size})',
                observation=f'optimizer="{optimizer_type}"',
            )

            return json.dumps(result)

        # Tool 7: Run Compilation
        def run_compilation(program_id: str, optimizer_type: str) -> str:
            """Run REAL DSPy compilation/optimization."""
            self._add_step(
                "Run Compilation",
                "run_compilation",
                "running",
                thought="Starting REAL DSPy optimization process...",
            )

            task_analysis = self.state.task_analysis or {}
            input_roles = task_analysis.get("input_roles", ["input"])
            output_roles = task_analysis.get("output_roles", ["output"])
            task_type = task_analysis.get("task_type", "classification")
            needs_cot = task_analysis.get("needs_chain_of_thought", False)

            try:
                (
                    dspy_mod,
                    BootstrapFewShot,
                    BootstrapFewShotWithRandomSearch,
                    MIPROv2,
                    COPRO,
                ) = _load_dspy()

                # Configure DSPy with target LM
                target_lm = (
                    self.state.target_lm
                    or f"openai/{get_settings().model_defaults.openai_chat}"
                )
                if "/" in target_lm:
                    provider, model_name = target_lm.split("/", 1)
                else:
                    provider = "openai"
                    model_name = target_lm

                if provider == "ollama":
                    ollama_base_url = get_settings().endpoints.ollama_base_url
                    lm = dspy_mod.LM(
                        f"ollama_chat/{model_name}", api_base=ollama_base_url
                    )
                elif provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    lm = dspy_mod.LM(f"anthropic/{model_name}", api_key=api_key)
                else:
                    api_key = os.getenv("OPENAI_API_KEY")
                    lm = dspy_mod.LM(f"openai/{model_name}", api_key=api_key)

                dspy_mod.configure(lm=lm)

                # Create dynamic Signature
                sig_fields = {}
                for role in input_roles:
                    sig_fields[role] = dspy_mod.InputField(desc=f"{role} for the task")
                for role in output_roles:
                    sig_fields[role] = dspy_mod.OutputField(
                        desc=f"{role} from the model"
                    )

                DynamicSignature = type(
                    f"{task_type.title()}Signature",
                    (dspy_mod.Signature,),
                    {
                        "__doc__": f"{task_type} task in {task_analysis.get('domain', 'general')} domain.",
                        **sig_fields,
                    },
                )

                # Create predictor
                if needs_cot:
                    predictor = dspy_mod.ChainOfThought(DynamicSignature)
                else:
                    predictor = dspy_mod.Predict(DynamicSignature)

                # Prepare training data
                full_dataset = self.state.dataset or []
                n = len(full_dataset)
                train_size = (
                    self.state.data_splits.get("train", int(n * 0.7))
                    if self.state.data_splits
                    else int(n * 0.7)
                )
                dev_size = (
                    self.state.data_splits.get("dev", int(n * 0.2))
                    if self.state.data_splits
                    else int(n * 0.2)
                )

                train_data = full_dataset[:train_size]
                dev_data = (
                    full_dataset[train_size : train_size + dev_size]
                    if dev_size > 0
                    else []
                )

                trainset = []
                for item in train_data:
                    example_dict = {}
                    if len(input_roles) == 1:
                        example_dict[input_roles[0]] = item.get("input", "")
                    if len(output_roles) == 1:
                        example_dict[output_roles[0]] = item.get("output", "")
                    trainset.append(
                        dspy_mod.Example(**example_dict).with_inputs(*input_roles)
                    )

                # Create metric
                def metric(example, pred, trace=None):
                    output_field = output_roles[0] if output_roles else "output"
                    gold = str(getattr(example, output_field, "")).strip().lower()
                    predicted = str(getattr(pred, output_field, "")).strip().lower()

                    if gold == predicted:
                        return 1.0
                    if gold in predicted or predicted in gold:
                        return 0.5
                    return 0.0

                # Select and run optimizer
                optimizer_params = self.state.optimizer_params or {}

                if optimizer_type == "BootstrapFewShot" and BootstrapFewShot:
                    optimizer = BootstrapFewShot(
                        metric=metric,
                        max_bootstrapped_demos=optimizer_params.get(
                            "max_bootstrapped_demos", 4
                        ),
                        max_labeled_demos=optimizer_params.get("max_labeled_demos", 16),
                        max_rounds=optimizer_params.get("max_rounds", 1),
                    )
                elif optimizer_type == "MIPROv2" and MIPROv2:
                    optimizer = MIPROv2(
                        metric=metric,
                        max_bootstrapped_demos=optimizer_params.get(
                            "max_bootstrapped_demos", 4
                        ),
                        max_labeled_demos=optimizer_params.get("max_labeled_demos", 4),
                        auto=optimizer_params.get("auto", "light"),
                    )
                else:
                    # Default to BootstrapFewShotWithRandomSearch
                    optimizer = BootstrapFewShotWithRandomSearch(
                        metric=metric,
                        max_bootstrapped_demos=optimizer_params.get(
                            "max_bootstrapped_demos", 4
                        ),
                        max_labeled_demos=optimizer_params.get("max_labeled_demos", 16),
                        max_rounds=optimizer_params.get("max_rounds", 1),
                        num_candidate_programs=optimizer_params.get(
                            "num_candidate_programs", 12
                        ),
                    )

                # Run compilation
                compiled_predictor = optimizer.compile(predictor, trainset=trainset)

                # Evaluate
                eval_examples = []
                for item in dev_data or train_data[:10]:
                    example_dict = {}
                    if len(input_roles) == 1:
                        example_dict[input_roles[0]] = item.get("input", "")
                    if len(output_roles) == 1:
                        example_dict[output_roles[0]] = item.get("output", "")
                    eval_examples.append(
                        dspy_mod.Example(**example_dict).with_inputs(*input_roles)
                    )

                correct = 0.0
                eval_count = min(10, len(eval_examples))
                for example in eval_examples[:eval_count]:
                    try:
                        input_kwargs = {
                            role: getattr(example, role) for role in input_roles
                        }
                        pred = compiled_predictor(**input_kwargs)
                        correct += metric(example, pred)
                    except:
                        pass

                final_metric = (correct / eval_count) if eval_count > 0 else 0.0

                self.state.compilation_result = {
                    "status": "success",
                    "metric_value": round(final_metric, 3),
                    "metric_name": "accuracy",
                    "iterations": len(trainset),
                    "real_dspy": True,
                }

                # Generate program code
                self.state.program_code = f"""import dspy

{self.state.signature_code or "# Signature not defined"}

class {task_type.title()}Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.{"ChainOfThought" if needs_cot else "Predict"}({task_type.title()}Signature)
    
    def forward(self, {", ".join(input_roles)}):
        return self.predictor({", ".join([f"{r}={r}" for r in input_roles])})

# Optimized with {optimizer_type}
# Metric: {final_metric:.3f}
# Target LM: {self.state.target_lm}
"""

                self._add_step(
                    "Run Compilation",
                    "run_compilation",
                    "success",
                    action=f'run_compilation(optimizer="{optimizer_type}")',
                    observation=f"metric={final_metric:.3f}, real_dspy=True",
                )

                return json.dumps(self.state.compilation_result)

            except Exception as e:
                self.state.compilation_result = {
                    "status": "error",
                    "error": str(e),
                    "real_dspy": False,
                }
                self._add_step(
                    "Run Compilation",
                    "run_compilation",
                    "error",
                    observation=f"compilation_failed: {e}",
                )
                return json.dumps(self.state.compilation_result)

        # Tool 8: Log Artifacts
        def log_artifacts() -> str:
            """Save artifacts and return final result."""
            self._add_step(
                "Log Artifacts",
                "log_artifacts",
                "running",
                thought="Saving optimized program and metadata...",
            )

            artifact_id = (
                f"v_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            )
            self.state.artifact_id = artifact_id

            artifact_dir = self.artifacts_dir / artifact_id
            artifact_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "artifact_version_id": artifact_id,
                "created_at": datetime.now().isoformat(),
                "target_lm": self.state.target_lm,
                "task_analysis": self.state.task_analysis,
                "signature_id": self.state.signature_id,
                "program_id": self.state.program_id,
                "eval_results": self.state.compilation_result,
                "react_iterations": len(self.state.steps),
            }

            with open(artifact_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            if self.state.program_code:
                with open(artifact_dir / "program.py", "w") as f:
                    f.write(self.state.program_code)

            self._add_step(
                "Log Artifacts",
                "log_artifacts",
                "success",
                action="log_artifacts()",
                observation=f'artifact_id="{artifact_id}"',
            )

            return json.dumps({"artifact_id": artifact_id, "path": str(artifact_dir)})

        # Create structured tools
        tools = [
            StructuredTool.from_function(
                func=analyze_business_goal,
                name="analyze_business_goal",
                description="Analyze business task and extract structured requirements",
                args_schema=AnalyzeBusinessGoalInput,
            ),
            StructuredTool.from_function(
                func=register_target_lm,
                name="register_target_lm",
                description="Register the target LLM for production inference",
                args_schema=RegisterTargetLMInput,
            ),
            StructuredTool.from_function(
                func=lambda input_roles, output_roles, task_type, domain: define_contract_signature(
                    input_roles, output_roles, task_type, domain
                ),
                name="define_contract_signature",
                description="Create DSPy Signature with input/output field definitions",
                args_schema=DefineSignatureInput,
            ),
            StructuredTool.from_function(
                func=lambda task_type, needs_retrieval, needs_chain_of_thought, complexity_level: assemble_program_pipeline(
                    task_type, needs_retrieval, needs_chain_of_thought, complexity_level
                ),
                name="assemble_program_pipeline",
                description="Assemble DSPy program pipeline with appropriate modules",
                args_schema=AssemblePipelineInput,
            ),
            StructuredTool.from_function(
                func=lambda train_ratio, dev_ratio: prepare_eval_splits(
                    train_ratio, dev_ratio
                ),
                name="prepare_eval_splits",
                description="Prepare train/dev/test data splits",
                args_schema=PrepareDataInput,
            ),
            StructuredTool.from_function(
                func=lambda task_type, complexity_level, data_size, profile: select_compiler_strategy(
                    task_type, complexity_level, data_size, profile
                ),
                name="select_compiler_strategy",
                description="Select DSPy optimizer strategy",
                args_schema=SelectOptimizerInput,
            ),
            StructuredTool.from_function(
                func=lambda program_id, optimizer_type: run_compilation(
                    program_id, optimizer_type
                ),
                name="run_compilation",
                description="Run DSPy compilation/optimization",
                args_schema=RunCompilationInput,
            ),
            StructuredTool.from_function(
                func=log_artifacts,
                name="log_artifacts",
                description="Save optimized artifacts and return final result",
            ),
        ]

        return tools

    def run(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str = "BALANCED",
    ) -> dict[str, Any]:
        """Run the full orchestration pipeline."""
        # Reset state
        self.state = AgentState()
        self.state.dataset = dataset
        self.state.business_task = business_task
        self.state.target_lm = target_lm

        print(f"Starting LangChain agent orchestration for: {business_task[:100]}...")

        if self.use_langgraph and self.agent:
            return self._run_with_langgraph(
                business_task, target_lm, dataset, quality_profile
            )
        else:
            return self._run_manual(business_task, target_lm, dataset, quality_profile)

    def _run_with_langgraph(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str,
    ) -> dict[str, Any]:
        """Run using LangGraph ReAct agent."""
        agent_input = f"""Build an optimized DSPy program for this task:

Business Task: {business_task}
Target LLM: {target_lm}
Dataset Size: {len(dataset)} examples
Quality Profile: {quality_profile}

Execute these steps in order:
1. analyze_business_goal - Analyze the business task
2. register_target_lm - Register {target_lm} as target
3. define_contract_signature - Create DSPy signature
4. assemble_program_pipeline - Build the pipeline
5. prepare_eval_splits - Split data (train=0.7, dev=0.2)
6. select_compiler_strategy - Select optimizer for {quality_profile} profile
7. run_compilation - Run DSPy optimization
8. log_artifacts - Save results

Return the final artifact ID when complete."""

        try:
            messages = [{"role": "user", "content": agent_input}]
            result = self.agent.invoke({"messages": messages})

            return self._build_response(True)

        except Exception as e:
            print(f"LangGraph agent failed: {e}, falling back to manual execution")
            return self._run_manual(business_task, target_lm, dataset, quality_profile)

    def _run_manual(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str,
    ) -> dict[str, Any]:
        """Run tools manually in sequence."""
        try:
            # Execute tools in order
            tools_by_name = {t.name: t for t in self.tools}

            # 1. Analyze business goal
            tools_by_name["analyze_business_goal"].invoke(
                {"task_description": business_task}
            )

            # 2. Register target LM
            tools_by_name["register_target_lm"].invoke({"target_lm_name": target_lm})

            # 3. Define signature
            analysis = self.state.task_analysis or {}
            tools_by_name["define_contract_signature"].invoke(
                {
                    "input_roles": analysis.get("input_roles", ["input"]),
                    "output_roles": analysis.get("output_roles", ["output"]),
                    "task_type": analysis.get("task_type", "classification"),
                    "domain": analysis.get("domain", "general"),
                }
            )

            # 4. Assemble pipeline
            tools_by_name["assemble_program_pipeline"].invoke(
                {
                    "task_type": analysis.get("task_type", "classification"),
                    "needs_retrieval": analysis.get("needs_retrieval", False),
                    "needs_chain_of_thought": analysis.get(
                        "needs_chain_of_thought", False
                    ),
                    "complexity_level": analysis.get("complexity_level", "medium"),
                }
            )

            # 5. Prepare data splits
            tools_by_name["prepare_eval_splits"].invoke(
                {"train_ratio": 0.7, "dev_ratio": 0.2}
            )

            # 6. Select optimizer
            tools_by_name["select_compiler_strategy"].invoke(
                {
                    "task_type": analysis.get("task_type", "classification"),
                    "complexity_level": analysis.get("complexity_level", "medium"),
                    "data_size": len(dataset),
                    "profile": quality_profile,
                }
            )

            # 7. Run compilation
            tools_by_name["run_compilation"].invoke(
                {
                    "program_id": self.state.program_id or "prog_default",
                    "optimizer_type": self.state.optimizer_type
                    or "BootstrapFewShotWithRandomSearch",
                }
            )

            # 8. Log artifacts
            tools_by_name["log_artifacts"].invoke({})

            return self._build_response(True)

        except Exception as e:
            print(f"Manual execution failed: {e}")
            return self._build_response(False, str(e))

    def _build_response(self, success: bool, error: str = None) -> dict[str, Any]:
        """Build the final response."""
        return {
            "success": success,
            "artifact_version_id": self.state.artifact_id,
            "compiled_program_id": self.state.program_id,
            "signature_id": self.state.signature_id,
            "eval_results": self.state.compilation_result or {},
            "task_analysis": self.state.task_analysis or {},
            "program_code": self.state.program_code or "",
            "optimizer_type": self.state.optimizer_type,
            "data_splits": self.state.data_splits,
            "react_iterations": len(self.state.steps),
            "steps": self.state.steps,
            "error": error,
        }

    async def run_async(
        self,
        business_task: str,
        target_lm: str,
        dataset: list[dict[str, str]],
        quality_profile: str = "BALANCED",
    ):
        """Async generator that yields steps as they complete."""
        # Reset state
        self.state = AgentState()
        self.state.dataset = dataset
        self.state.business_task = business_task
        self.state.target_lm = target_lm

        tools_by_name = {t.name: t for t in self.tools}

        try:
            # 1. Analyze business goal
            yield {
                "type": "step",
                "data": {"name": "Analyze Business Goal", "status": "running"},
            }
            tools_by_name["analyze_business_goal"].invoke(
                {"task_description": business_task}
            )
            yield {"type": "step", "data": self.state.steps[-1]}

            # 2. Register target LM
            yield {
                "type": "step",
                "data": {"name": "Register Target LM", "status": "running"},
            }
            tools_by_name["register_target_lm"].invoke({"target_lm_name": target_lm})
            yield {"type": "step", "data": self.state.steps[-1]}

            # 3. Define signature
            analysis = self.state.task_analysis or {}
            yield {
                "type": "step",
                "data": {"name": "Define Signature", "status": "running"},
            }
            tools_by_name["define_contract_signature"].invoke(
                {
                    "input_roles": analysis.get("input_roles", ["input"]),
                    "output_roles": analysis.get("output_roles", ["output"]),
                    "task_type": analysis.get("task_type", "classification"),
                    "domain": analysis.get("domain", "general"),
                }
            )
            yield {"type": "step", "data": self.state.steps[-1]}

            # 4. Assemble pipeline
            yield {
                "type": "step",
                "data": {"name": "Assemble Pipeline", "status": "running"},
            }
            tools_by_name["assemble_program_pipeline"].invoke(
                {
                    "task_type": analysis.get("task_type", "classification"),
                    "needs_retrieval": analysis.get("needs_retrieval", False),
                    "needs_chain_of_thought": analysis.get(
                        "needs_chain_of_thought", False
                    ),
                    "complexity_level": analysis.get("complexity_level", "medium"),
                }
            )
            yield {"type": "step", "data": self.state.steps[-1]}

            # 5. Prepare data splits
            yield {
                "type": "step",
                "data": {"name": "Prepare Data", "status": "running"},
            }
            tools_by_name["prepare_eval_splits"].invoke(
                {"train_ratio": 0.7, "dev_ratio": 0.2}
            )
            yield {"type": "step", "data": self.state.steps[-1]}

            # 6. Select optimizer
            yield {
                "type": "step",
                "data": {"name": "Select Optimizer", "status": "running"},
            }
            tools_by_name["select_compiler_strategy"].invoke(
                {
                    "task_type": analysis.get("task_type", "classification"),
                    "complexity_level": analysis.get("complexity_level", "medium"),
                    "data_size": len(dataset),
                    "profile": quality_profile,
                }
            )
            yield {"type": "step", "data": self.state.steps[-1]}

            # 7. Run compilation
            yield {
                "type": "step",
                "data": {"name": "Run Compilation", "status": "running"},
            }
            tools_by_name["run_compilation"].invoke(
                {
                    "program_id": self.state.program_id or "prog_default",
                    "optimizer_type": self.state.optimizer_type
                    or "BootstrapFewShotWithRandomSearch",
                }
            )
            yield {"type": "step", "data": self.state.steps[-1]}

            # 8. Log artifacts
            yield {
                "type": "step",
                "data": {"name": "Log Artifacts", "status": "running"},
            }
            tools_by_name["log_artifacts"].invoke({})
            yield {"type": "step", "data": self.state.steps[-1]}

            # Final result
            yield {"type": "complete", "data": self._build_response(True)}

        except Exception as e:
            yield {"type": "error", "data": {"error": str(e)}}
