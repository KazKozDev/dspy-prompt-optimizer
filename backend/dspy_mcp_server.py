"""
DSPy MCP Server - Model Context Protocol server for DSPy configuration.

Provides tools for LangChain agent to configure and run DSPy optimization.
"""

import json
import os
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# MCP Server imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# DSPy imports (lazy loaded)
dspy = None


def _load_dspy():
    """Lazy load DSPy."""
    global dspy
    if dspy is None:
        import dspy as _dspy
        dspy = _dspy
        # Disable caching to avoid SQLite lock issues
        if hasattr(dspy, "configure_cache"):
            dspy.configure_cache(False)
    return dspy


# ==================== State Management ====================

class DSPySessionState:
    """Holds state for a DSPy optimization session."""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.task_analysis: Optional[Dict] = None
        self.signature_code: Optional[str] = None
        self.signature_class: Optional[type] = None
        self.module_type: str = "Predict"
        self.module_instance: Any = None
        self.dataset: List[Dict] = []
        self.train_set: List = []
        self.dev_set: List = []
        self.test_set: List = []
        self.optimizer_type: str = "BootstrapFewShot"
        self.optimizer_config: Dict = {}
        self.compiled_program: Any = None
        self.eval_results: Dict = {}
        self.target_lm: Optional[str] = None
        self.lm_instance: Any = None
        self.program_code: Optional[str] = None
        self.artifacts_dir: str = "data/artifacts"


# Global session state
_session = DSPySessionState()


def reset_session():
    """Reset session state."""
    global _session
    _session = DSPySessionState()
    return _session


# ==================== MCP Server ====================

server = Server("dspy-optimizer")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available DSPy tools."""
    return [
        Tool(
            name="analyze_task",
            description="Analyze a business task to determine task type, domain, complexity, and required DSPy configuration. Returns structured analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "The business task description to analyze"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="create_signature",
            description="Create a DSPy Signature with specified input and output fields. The signature defines the contract between input and output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of input field names (e.g., ['question', 'context'])"
                    },
                    "output_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of output field names (e.g., ['answer', 'confidence'])"
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Description of what the signature should do"
                    }
                },
                "required": ["input_fields", "output_fields"]
            }
        ),
        Tool(
            name="create_module",
            description="Create a DSPy module (Predict, ChainOfThought, or ReAct) using the current signature.",
            inputSchema={
                "type": "object",
                "properties": {
                    "module_type": {
                        "type": "string",
                        "enum": ["Predict", "ChainOfThought", "ReAct"],
                        "description": "Type of DSPy module to create"
                    },
                    "use_cot": {
                        "type": "boolean",
                        "description": "Whether to use Chain of Thought reasoning",
                        "default": False
                    }
                },
                "required": ["module_type"]
            }
        ),
        Tool(
            name="configure_lm",
            description="Configure the target Language Model for DSPy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name with provider prefix (e.g., 'ollama/gemma3:4b', 'openai/gpt-4o')"
                    }
                },
                "required": ["model_name"]
            }
        ),
        Tool(
            name="prepare_dataset",
            description="Prepare dataset for DSPy optimization by splitting into train/dev/test sets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string"},
                                "output": {"type": "string"}
                            }
                        },
                        "description": "Dataset as list of {input, output} objects"
                    },
                    "train_ratio": {
                        "type": "number",
                        "description": "Ratio for training set (0.0-1.0)",
                        "default": 0.7
                    },
                    "dev_ratio": {
                        "type": "number",
                        "description": "Ratio for dev/validation set (0.0-1.0)",
                        "default": 0.2
                    }
                },
                "required": ["dataset"]
            }
        ),
        Tool(
            name="select_optimizer",
            description="Select and configure the DSPy optimizer based on task requirements.",
            inputSchema={
                "type": "object",
                "properties": {
                    "optimizer_type": {
                        "type": "string",
                        "enum": ["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2", "COPRO"],
                        "description": "Type of DSPy optimizer to use"
                    },
                    "max_bootstrapped_demos": {
                        "type": "integer",
                        "description": "Maximum number of bootstrapped demonstrations",
                        "default": 4
                    },
                    "max_labeled_demos": {
                        "type": "integer",
                        "description": "Maximum number of labeled demonstrations",
                        "default": 4
                    },
                    "num_candidate_programs": {
                        "type": "integer",
                        "description": "Number of candidate programs to evaluate (for random search)",
                        "default": 10
                    }
                },
                "required": ["optimizer_type"]
            }
        ),
        Tool(
            name="run_optimization",
            description="Run DSPy optimization/compilation with the configured settings. This is the main optimization step.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_type": {
                        "type": "string",
                        "enum": ["exact_match", "contains", "f1", "custom"],
                        "description": "Type of evaluation metric",
                        "default": "exact_match"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="evaluate_program",
            description="Evaluate the compiled program on the test set.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of test samples to evaluate",
                        "default": 50
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="export_program",
            description="Export the optimized program as Python code and save artifacts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_name": {
                        "type": "string",
                        "description": "Name for the exported artifact"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_session_status",
            description="Get current session status including configured components and results.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global _session
    
    try:
        if name == "analyze_task":
            result = await _analyze_task(arguments.get("task_description", ""))
        elif name == "create_signature":
            result = await _create_signature(
                arguments.get("input_fields", []),
                arguments.get("output_fields", []),
                arguments.get("task_description", "")
            )
        elif name == "create_module":
            result = await _create_module(
                arguments.get("module_type", "Predict"),
                arguments.get("use_cot", False)
            )
        elif name == "configure_lm":
            result = await _configure_lm(arguments.get("model_name", ""))
        elif name == "prepare_dataset":
            result = await _prepare_dataset(
                arguments.get("dataset", []),
                arguments.get("train_ratio", 0.7),
                arguments.get("dev_ratio", 0.2)
            )
        elif name == "select_optimizer":
            result = await _select_optimizer(
                arguments.get("optimizer_type", "BootstrapFewShot"),
                arguments.get("max_bootstrapped_demos", 4),
                arguments.get("max_labeled_demos", 4),
                arguments.get("num_candidate_programs", 10)
            )
        elif name == "run_optimization":
            result = await _run_optimization(
                arguments.get("metric_type", "exact_match")
            )
        elif name == "evaluate_program":
            result = await _evaluate_program(
                arguments.get("num_samples", 50)
            )
        elif name == "export_program":
            result = await _export_program(
                arguments.get("artifact_name", "")
            )
        elif name == "get_session_status":
            result = await _get_session_status()
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ==================== Tool Implementations ====================

async def _analyze_task(task_description: str) -> Dict:
    """Analyze business task and determine DSPy configuration."""
    task_lower = task_description.lower()
    
    # Determine task type
    if any(w in task_lower for w in ["classify", "categorize", "label", "intent", "sentiment"]):
        task_type = "classification"
    elif any(w in task_lower for w in ["extract", "ner", "entity", "parse"]):
        task_type = "extraction"
    elif any(w in task_lower for w in ["summarize", "summary", "tldr"]):
        task_type = "summarization"
    elif any(w in task_lower for w in ["question", "answer", "qa", "respond"]):
        task_type = "question_answering"
    elif any(w in task_lower for w in ["translate", "convert"]):
        task_type = "translation"
    elif any(w in task_lower for w in ["generate", "write", "create"]):
        task_type = "generation"
    else:
        task_type = "general"
    
    # Determine domain
    if any(w in task_lower for w in ["customer", "support", "service", "ticket"]):
        domain = "customer_support"
    elif any(w in task_lower for w in ["legal", "law", "contract"]):
        domain = "legal"
    elif any(w in task_lower for w in ["medical", "health", "clinical"]):
        domain = "medical"
    elif any(w in task_lower for w in ["finance", "banking", "money"]):
        domain = "finance"
    elif any(w in task_lower for w in ["code", "programming", "software"]):
        domain = "technical"
    else:
        domain = "general"
    
    # Determine complexity
    word_count = len(task_description.split())
    if word_count > 50 or any(w in task_lower for w in ["complex", "multi-step", "reasoning"]):
        complexity = "high"
    elif word_count > 20:
        complexity = "medium"
    else:
        complexity = "low"
    
    # Recommend configuration
    needs_cot = complexity in ["medium", "high"] or task_type in ["question_answering", "extraction"]
    
    if task_type == "classification":
        input_fields = ["text"]
        output_fields = ["category"]
    elif task_type == "extraction":
        input_fields = ["text"]
        output_fields = ["entities"]
    elif task_type == "question_answering":
        input_fields = ["question", "context"]
        output_fields = ["answer"]
    elif task_type == "summarization":
        input_fields = ["document"]
        output_fields = ["summary"]
    else:
        input_fields = ["input"]
        output_fields = ["output"]
    
    # Recommend optimizer
    if complexity == "high":
        recommended_optimizer = "MIPROv2"
    elif complexity == "medium":
        recommended_optimizer = "BootstrapFewShotWithRandomSearch"
    else:
        recommended_optimizer = "BootstrapFewShot"
    
    analysis = {
        "task_type": task_type,
        "domain": domain,
        "complexity": complexity,
        "needs_chain_of_thought": needs_cot,
        "recommended_input_fields": input_fields,
        "recommended_output_fields": output_fields,
        "recommended_optimizer": recommended_optimizer,
        "recommended_module": "ChainOfThought" if needs_cot else "Predict"
    }
    
    _session.task_analysis = analysis
    return {"status": "success", "analysis": analysis}


async def _create_signature(
    input_fields: List[str],
    output_fields: List[str],
    task_description: str = ""
) -> Dict:
    """Create DSPy Signature."""
    dspy = _load_dspy()
    
    # Build signature fields
    sig_fields = {}
    for field in input_fields:
        sig_fields[field] = dspy.InputField(desc=f"Input: {field}")
    for field in output_fields:
        sig_fields[field] = dspy.OutputField(desc=f"Output: {field}")
    
    # Create signature class
    doc = task_description or f"Process {', '.join(input_fields)} to produce {', '.join(output_fields)}"
    
    SignatureClass = type(
        "DynamicSignature",
        (dspy.Signature,),
        {"__doc__": doc, **sig_fields}
    )
    
    _session.signature_class = SignatureClass
    
    # Generate code representation
    code = f'''class TaskSignature(dspy.Signature):
    """{doc}"""
'''
    for field in input_fields:
        code += f'    {field} = dspy.InputField(desc="Input: {field}")\n'
    for field in output_fields:
        code += f'    {field} = dspy.OutputField(desc="Output: {field}")\n'
    
    _session.signature_code = code
    
    return {
        "status": "success",
        "signature_code": code,
        "input_fields": input_fields,
        "output_fields": output_fields
    }


async def _create_module(module_type: str, use_cot: bool = False) -> Dict:
    """Create DSPy module."""
    dspy = _load_dspy()
    
    if _session.signature_class is None:
        return {"error": "No signature defined. Call create_signature first."}
    
    _session.module_type = module_type
    
    if module_type == "ChainOfThought" or use_cot:
        _session.module_instance = dspy.ChainOfThought(_session.signature_class)
        actual_type = "ChainOfThought"
    elif module_type == "ReAct":
        _session.module_instance = dspy.ReAct(_session.signature_class)
        actual_type = "ReAct"
    else:
        _session.module_instance = dspy.Predict(_session.signature_class)
        actual_type = "Predict"
    
    return {
        "status": "success",
        "module_type": actual_type,
        "message": f"Created {actual_type} module with signature"
    }


async def _configure_lm(model_name: str) -> Dict:
    """Configure Language Model."""
    dspy = _load_dspy()
    
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
        _session.lm_instance = lm
        _session.target_lm = model_name
        
        return {
            "status": "success",
            "model": model_name,
            "provider": provider
        }
    except Exception as e:
        return {"error": f"Failed to configure LM: {str(e)}"}


async def _prepare_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.2
) -> Dict:
    """Prepare dataset splits."""
    dspy = _load_dspy()
    
    if not dataset:
        return {"error": "Empty dataset provided"}
    
    _session.dataset = dataset
    
    # Determine field names from task analysis or dataset
    if _session.task_analysis:
        input_fields = _session.task_analysis.get("recommended_input_fields", ["input"])
        output_fields = _session.task_analysis.get("recommended_output_fields", ["output"])
    else:
        input_fields = ["input"]
        output_fields = ["output"]
    
    # Convert to DSPy Examples
    examples = []
    for item in dataset:
        ex_dict = {}
        # Map input
        if "input" in item:
            ex_dict[input_fields[0]] = item["input"]
        # Map output
        if "output" in item:
            ex_dict[output_fields[0]] = item["output"]
        
        example = dspy.Example(**ex_dict).with_inputs(*input_fields)
        examples.append(example)
    
    # Split
    n = len(examples)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    
    _session.train_set = examples[:train_end]
    _session.dev_set = examples[train_end:dev_end]
    _session.test_set = examples[dev_end:]
    
    return {
        "status": "success",
        "total_examples": n,
        "train_size": len(_session.train_set),
        "dev_size": len(_session.dev_set),
        "test_size": len(_session.test_set)
    }


async def _select_optimizer(
    optimizer_type: str,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_candidate_programs: int = 10
) -> Dict:
    """Select and configure optimizer."""
    _session.optimizer_type = optimizer_type
    _session.optimizer_config = {
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
        "num_candidate_programs": num_candidate_programs
    }
    
    return {
        "status": "success",
        "optimizer_type": optimizer_type,
        "config": _session.optimizer_config
    }


async def _run_optimization(metric_type: str = "exact_match") -> Dict:
    """Run DSPy optimization."""
    dspy = _load_dspy()
    
    # Validate prerequisites
    if _session.module_instance is None:
        return {"error": "No module created. Call create_module first."}
    if not _session.train_set:
        return {"error": "No training data. Call prepare_dataset first."}
    if _session.lm_instance is None:
        return {"error": "No LM configured. Call configure_lm first."}
    
    # Define metric
    if _session.task_analysis:
        output_field = _session.task_analysis.get("recommended_output_fields", ["output"])[0]
    else:
        output_field = "output"
    
    def exact_match_metric(example, pred, trace=None):
        expected = getattr(example, output_field, "").strip().lower()
        predicted = getattr(pred, output_field, "").strip().lower()
        return 1.0 if expected == predicted else 0.0
    
    def contains_metric(example, pred, trace=None):
        expected = getattr(example, output_field, "").strip().lower()
        predicted = getattr(pred, output_field, "").strip().lower()
        return 1.0 if expected in predicted or predicted in expected else 0.0
    
    metric = exact_match_metric if metric_type == "exact_match" else contains_metric
    
    # Get optimizer
    from dspy import teleprompt as tp
    
    optimizer_type = _session.optimizer_type
    config = _session.optimizer_config
    
    try:
        if optimizer_type == "BootstrapFewShot":
            optimizer = tp.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
                max_labeled_demos=config.get("max_labeled_demos", 4)
            )
        elif optimizer_type == "BootstrapFewShotWithRandomSearch":
            optimizer = tp.BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=config.get("max_bootstrapped_demos", 4),
                max_labeled_demos=config.get("max_labeled_demos", 4),
                num_candidate_programs=config.get("num_candidate_programs", 10)
            )
        elif optimizer_type == "MIPROv2":
            optimizer = tp.MIPROv2(
                metric=metric,
                num_candidates=config.get("num_candidate_programs", 10)
            )
        elif optimizer_type == "COPRO":
            optimizer = tp.COPRO(
                metric=metric
            )
        else:
            optimizer = tp.BootstrapFewShot(metric=metric)
        
        # Run compilation
        compiled = optimizer.compile(
            _session.module_instance,
            trainset=_session.train_set
        )
        
        _session.compiled_program = compiled
        
        # Quick evaluation on dev set
        if _session.dev_set:
            correct = 0
            total = min(len(_session.dev_set), 20)
            for ex in _session.dev_set[:total]:
                try:
                    input_field = list(ex.inputs().keys())[0]
                    pred = compiled(**{input_field: getattr(ex, input_field)})
                    if metric(ex, pred) > 0.5:
                        correct += 1
                except:
                    pass
            dev_accuracy = correct / total if total > 0 else 0
        else:
            dev_accuracy = 0
        
        _session.eval_results = {
            "optimizer_type": optimizer_type,
            "dev_accuracy": dev_accuracy,
            "train_size": len(_session.train_set)
        }
        
        return {
            "status": "success",
            "optimizer_type": optimizer_type,
            "dev_accuracy": dev_accuracy,
            "message": f"Optimization complete. Dev accuracy: {dev_accuracy:.1%}"
        }
    
    except Exception as e:
        return {"error": f"Optimization failed: {str(e)}"}


async def _evaluate_program(num_samples: int = 50) -> Dict:
    """Evaluate compiled program on test set."""
    if _session.compiled_program is None:
        return {"error": "No compiled program. Run optimization first."}
    
    if not _session.test_set:
        return {"error": "No test data available."}
    
    if _session.task_analysis:
        output_field = _session.task_analysis.get("recommended_output_fields", ["output"])[0]
    else:
        output_field = "output"
    
    correct = 0
    total = min(len(_session.test_set), num_samples)
    results = []
    
    for ex in _session.test_set[:total]:
        try:
            input_field = list(ex.inputs().keys())[0]
            input_val = getattr(ex, input_field)
            pred = _session.compiled_program(**{input_field: input_val})
            
            expected = getattr(ex, output_field, "").strip().lower()
            predicted = getattr(pred, output_field, "").strip().lower()
            
            is_correct = expected == predicted
            if is_correct:
                correct += 1
            
            results.append({
                "input": input_val[:100],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct
            })
        except Exception as e:
            results.append({"error": str(e)})
    
    accuracy = correct / total if total > 0 else 0
    
    _session.eval_results["test_accuracy"] = accuracy
    _session.eval_results["test_samples"] = total
    
    return {
        "status": "success",
        "test_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sample_results": results[:5]
    }


async def _export_program(artifact_name: str = "") -> Dict:
    """Export optimized program."""
    if _session.compiled_program is None:
        return {"error": "No compiled program to export."}
    
    # Generate artifact ID
    artifact_id = artifact_name or f"dspy_{_session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate program code
    code = f'''"""
DSPy Optimized Program
Generated: {datetime.now().isoformat()}
Optimizer: {_session.optimizer_type}
"""

import dspy

# Signature
{_session.signature_code or "# Signature not available"}

# Module
class OptimizedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.{_session.module_type}(TaskSignature)
    
    def forward(self, **kwargs):
        return self.predictor(**kwargs)

# Usage
# module = OptimizedModule()
# result = module(input="your input here")
'''
    
    _session.program_code = code
    
    # Save artifact
    artifact_dir = Path(_session.artifacts_dir) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save code
    with open(artifact_dir / "program.py", "w") as f:
        f.write(code)
    
    # Save metadata
    metadata = {
        "artifact_id": artifact_id,
        "created_at": datetime.now().isoformat(),
        "task_analysis": _session.task_analysis,
        "optimizer_type": _session.optimizer_type,
        "eval_results": _session.eval_results,
        "target_lm": _session.target_lm
    }
    
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "status": "success",
        "artifact_id": artifact_id,
        "artifact_path": str(artifact_dir),
        "program_code": code
    }


async def _get_session_status() -> Dict:
    """Get current session status."""
    return {
        "session_id": _session.session_id,
        "task_analysis": _session.task_analysis,
        "signature_defined": _session.signature_class is not None,
        "module_type": _session.module_type if _session.module_instance else None,
        "lm_configured": _session.target_lm,
        "dataset_size": len(_session.dataset),
        "train_size": len(_session.train_set),
        "dev_size": len(_session.dev_set),
        "test_size": len(_session.test_set),
        "optimizer_type": _session.optimizer_type,
        "compiled": _session.compiled_program is not None,
        "eval_results": _session.eval_results
    }


# ==================== Main ====================

async def main():
    """Run MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
