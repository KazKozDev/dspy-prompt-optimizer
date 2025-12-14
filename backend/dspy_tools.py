"""
DSPy Tools - Direct implementation of MCP-style tools for DSPy configuration.

These tools are used by the LangChain agent to configure and run DSPy optimization.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# DSPy lazy loading
_dspy = None


def _load_dspy():
    """Lazy load DSPy with cache disabled."""
    global _dspy
    if _dspy is None:
        import dspy
        _dspy = dspy
        if hasattr(dspy, "configure_cache"):
            dspy.configure_cache(False)
    return _dspy


class DSPyToolsSession:
    """Session state for DSPy tools."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
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


# Global session
_session = DSPyToolsSession()


def reset_session():
    """Reset the global session."""
    global _session
    _session.reset()


async def analyze_task(task_description: str) -> Dict:
    """Analyze business task and determine DSPy configuration."""
    task_lower = task_description.lower()
    
    # Task type
    if any(w in task_lower for w in ["classify", "categorize", "label", "intent", "sentiment"]):
        task_type = "classification"
    elif any(w in task_lower for w in ["extract", "ner", "entity"]):
        task_type = "extraction"
    elif any(w in task_lower for w in ["summarize", "summary"]):
        task_type = "summarization"
    elif any(w in task_lower for w in ["question", "answer", "qa"]):
        task_type = "question_answering"
    else:
        task_type = "general"
    
    # Domain
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
    complexity = "high" if word_count > 50 else "medium" if word_count > 20 else "low"
    needs_cot = complexity in ["medium", "high"]
    
    # Field suggestions
    if task_type == "classification":
        input_fields = ["text"]
        output_fields = ["category"]
    elif task_type == "question_answering":
        input_fields = ["question"]
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
        "recommended_optimizer": "BootstrapFewShotWithRandomSearch" if complexity != "low" else "BootstrapFewShot",
        "recommended_module": "ChainOfThought" if needs_cot else "Predict"
    }
    
    _session.task_analysis = analysis
    return {"status": "success", "analysis": analysis}


async def create_signature(
    input_fields: List[str],
    output_fields: List[str],
    task_description: str = ""
) -> Dict:
    """Create DSPy Signature."""
    dspy = _load_dspy()
    
    sig_fields = {}
    for field in input_fields:
        sig_fields[field] = dspy.InputField(desc=f"Input: {field}")
    for field in output_fields:
        sig_fields[field] = dspy.OutputField(desc=f"Output: {field}")
    
    doc = task_description or f"Process {', '.join(input_fields)} to produce {', '.join(output_fields)}"
    
    SignatureClass = type(
        "DynamicSignature",
        (dspy.Signature,),
        {"__doc__": doc, **sig_fields}
    )
    
    _session.signature_class = SignatureClass
    
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


async def create_module(module_type: str, use_cot: bool = False) -> Dict:
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
    
    return {"status": "success", "module_type": actual_type}


async def configure_lm(model_name: str) -> Dict:
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
        
        return {"status": "success", "model": model_name, "provider": provider}
    except Exception as e:
        return {"error": f"Failed to configure LM: {str(e)}"}


async def prepare_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.2
) -> Dict:
    """Prepare dataset splits."""
    import random
    dspy = _load_dspy()
    
    if not dataset:
        return {"error": "Empty dataset"}
    
    # Detect input/output field names from dataset
    # Support common naming conventions: input/output, text/label, question/answer, etc.
    INPUT_FIELD_ALIASES = ["input", "text", "question", "query", "content", "email", "message", "sentence", "document"]
    OUTPUT_FIELD_ALIASES = ["output", "label", "answer", "category", "class", "response", "target", "result"]
    
    # Auto-detect fields from first example
    first_item = dataset[0]
    dataset_input_field = None
    dataset_output_field = None
    
    for alias in INPUT_FIELD_ALIASES:
        if alias in first_item:
            dataset_input_field = alias
            break
    
    for alias in OUTPUT_FIELD_ALIASES:
        if alias in first_item:
            dataset_output_field = alias
            break
    
    # Fallback: use first two keys if no known aliases found
    if not dataset_input_field or not dataset_output_field:
        keys = list(first_item.keys())
        if len(keys) >= 2:
            dataset_input_field = dataset_input_field or keys[0]
            dataset_output_field = dataset_output_field or keys[1]
        else:
            return {"error": f"Dataset must have at least 2 fields. Found: {keys}"}
    
    print(f"[DSPy Tools] Detected dataset fields: {dataset_input_field} -> {dataset_output_field}")
    
    # Validate and filter dataset
    valid_dataset = []
    invalid_count = 0
    for i, item in enumerate(dataset):
        inp = item.get(dataset_input_field, "")
        out = item.get(dataset_output_field, "")
        
        # Skip empty or invalid examples
        if not inp or not out or not str(inp).strip() or not str(out).strip():
            invalid_count += 1
            continue
        
        valid_dataset.append(item)
    
    if invalid_count > 0:
        print(f"[DSPy Tools] Skipped {invalid_count} invalid/empty examples")
    
    if len(valid_dataset) < 3:
        return {"error": f"Not enough valid examples. Need at least 3, got {len(valid_dataset)}"}
    
    dataset = valid_dataset
    
    # Shuffle dataset to ensure balanced splits across categories
    dataset = dataset.copy()  # Don't modify original
    # Use hash of first+last items for reproducibility within same dataset
    # but different shuffles for different datasets
    seed_str = str(dataset[0]) + str(dataset[-1]) + str(len(dataset))
    seed_val = hash(seed_str) % (2**32)
    random.seed(seed_val)
    random.shuffle(dataset)
    print(f"[DSPy Tools] Shuffled dataset (n={len(dataset)})")
    
    _session.dataset = dataset
    
    # Get field names from signature if available
    if _session.signature_class:
        # Extract actual field names from signature
        sig_fields = _session.signature_class.model_fields
        sig_input_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'input']
        sig_output_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'output']
        
        # Fallback if detection fails
        if not sig_input_fields:
            sig_input_fields = [list(sig_fields.keys())[0]] if sig_fields else ["input"]
        if not sig_output_fields:
            sig_output_fields = [list(sig_fields.keys())[-1]] if len(sig_fields) > 1 else ["output"]
    else:
        sig_input_fields = ["input"]
        sig_output_fields = ["output"]
    
    print(f"[DSPy Tools] Mapping: dataset[{dataset_input_field}] -> signature[{sig_input_fields[0]}], dataset[{dataset_output_field}] -> signature[{sig_output_fields[0]}]")
    
    examples = []
    for item in dataset:
        ex_dict = {}
        # Map dataset field to signature field
        ex_dict[sig_input_fields[0]] = str(item[dataset_input_field]).strip()
        ex_dict[sig_output_fields[0]] = str(item[dataset_output_field]).strip()
        example = dspy.Example(**ex_dict).with_inputs(*sig_input_fields)
        examples.append(example)
    
    n = len(examples)
    
    # Adaptive split ratios for small datasets
    # For small datasets, we need more test samples for reliable evaluation
    if n <= 10:
        # Very small: 50% train, 20% dev, 30% test (minimum 3 test samples)
        actual_train_ratio = 0.5
        actual_dev_ratio = 0.2
        print(f"[DSPy Tools] Small dataset ({n} examples), using 50/20/30 split")
    elif n <= 20:
        # Small: 60% train, 20% dev, 20% test
        actual_train_ratio = 0.6
        actual_dev_ratio = 0.2
        print(f"[DSPy Tools] Medium-small dataset ({n} examples), using 60/20/20 split")
    else:
        # Normal: use provided ratios
        actual_train_ratio = train_ratio
        actual_dev_ratio = dev_ratio
    
    train_end = int(n * actual_train_ratio)
    dev_end = int(n * (actual_train_ratio + actual_dev_ratio))
    
    # Ensure at least 1 example in each split if possible
    if n >= 3:
        train_end = max(1, train_end)
        dev_end = max(train_end + 1, dev_end)
        dev_end = min(dev_end, n - 1)  # Leave at least 1 for test
    
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


async def select_optimizer(
    optimizer_type: str,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_candidate_programs: int = 10
) -> Dict:
    """Select optimizer."""
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


async def run_optimization(metric_type: str = "exact_match") -> Dict:
    """Run DSPy optimization."""
    dspy = _load_dspy()
    
    if _session.module_instance is None:
        return {"error": "No module created"}
    if not _session.train_set:
        return {"error": "No training data"}
    if _session.lm_instance is None:
        return {"error": "No LM configured"}
    
    # Get output field from signature
    if _session.signature_class:
        sig_fields = _session.signature_class.model_fields
        output_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'output']
        output_field = output_fields[0] if output_fields else list(sig_fields.keys())[-1]
    else:
        output_field = "output"
    
    print(f"[DSPy Tools] Using output field for metric: {output_field}")
    
    def exact_match_metric(example, pred, trace=None):
        expected = getattr(example, output_field, "").strip().lower()
        predicted = getattr(pred, output_field, "").strip().lower()
        return 1.0 if expected == predicted else 0.0
    
    def contains_metric(example, pred, trace=None):
        expected = getattr(example, output_field, "").strip().lower()
        predicted = getattr(pred, output_field, "").strip().lower()
        return 1.0 if expected in predicted or predicted in expected else 0.0
    
    metric = exact_match_metric if metric_type == "exact_match" else contains_metric
    
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
        else:
            optimizer = tp.BootstrapFewShot(metric=metric)
        
        compiled = optimizer.compile(
            _session.module_instance,
            trainset=_session.train_set
        )
        
        _session.compiled_program = compiled
        
        # Quick dev evaluation
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


async def evaluate_program(num_samples: int = 50) -> Dict:
    """Evaluate on test set."""
    if _session.compiled_program is None:
        return {"error": "No compiled program"}
    
    if not _session.test_set:
        return {"error": "No test data", "test_accuracy": 0, "correct": 0, "total": 0}
    
    # Get field names from signature
    input_field = "input"
    output_field = "output"
    if _session.signature_class:
        sig_fields = _session.signature_class.model_fields
        input_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'input']
        output_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'output']
        if input_fields:
            input_field = input_fields[0]
        if output_fields:
            output_field = output_fields[0]
    
    print(f"[Evaluate] Using fields: input={input_field}, output={output_field}")
    print(f"[Evaluate] Test set size: {len(_session.test_set)}")
    
    correct = 0
    total = min(len(_session.test_set), num_samples)
    
    for i, ex in enumerate(_session.test_set[:total]):
        try:
            # Get input value from example
            ex_input_field = list(ex.inputs().keys())[0]
            input_val = getattr(ex, ex_input_field, "")
            
            # Run prediction with signature's input field name
            pred = _session.compiled_program(**{input_field: input_val})
            
            # Get expected from example's output field
            ex_output_field = [k for k in ex.keys() if k not in ex.inputs().keys()][0] if ex.keys() else output_field
            expected = getattr(ex, ex_output_field, "").strip().lower()
            
            # Get predicted from prediction's output field
            predicted = getattr(pred, output_field, "").strip().lower()
            
            print(f"[Evaluate] #{i}: input='{input_val[:30]}...' expected='{expected}' predicted='{predicted}'")
            
            # Flexible matching: exact match OR expected is contained in predicted
            # This handles cases where model outputs "meeting" vs "Category: meeting"
            if expected == predicted or expected in predicted or predicted in expected:
                correct += 1
        except Exception as e:
            print(f"[Evaluate] Error on example {i}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    _session.eval_results["test_accuracy"] = accuracy
    
    print(f"[Evaluate] Result: {correct}/{total} = {accuracy:.1%}")
    
    return {
        "status": "success",
        "test_accuracy": accuracy,
        "correct": correct,
        "total": total
    }


async def export_program(artifact_name: str = "") -> Dict:
    """Export program with few-shot demos."""
    if _session.compiled_program is None:
        return {"error": "No compiled program"}
    
    artifact_id = artifact_name or f"dspy_{_session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Extract few-shot demos from compiled program
    demos_code = ""
    demos_list = []
    
    try:
        # Get demos from the compiled predictor
        if hasattr(_session.compiled_program, 'predictor'):
            predictor = _session.compiled_program.predictor
        else:
            predictor = _session.compiled_program
        
        if hasattr(predictor, 'demos') and predictor.demos:
            demos_list = []
            for demo in predictor.demos:
                demo_dict = {}
                for key in demo.keys():
                    demo_dict[key] = getattr(demo, key, "")
                demos_list.append(demo_dict)
            
            if demos_list:
                demos_code = f"\n# Few-shot demonstrations (optimized by DSPy)\nDEMOS = {json.dumps(demos_list, indent=4, ensure_ascii=False)}\n"
    except Exception as e:
        print(f"[Export] Could not extract demos: {e}")
    
    # Get field names
    input_field = "input"
    output_field = "output"
    if _session.signature_class:
        sig_fields = _session.signature_class.model_fields
        input_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'input']
        output_fields = [k for k, v in sig_fields.items() if hasattr(v, 'json_schema_extra') and v.json_schema_extra and v.json_schema_extra.get('__dspy_field_type') == 'output']
        if input_fields:
            input_field = input_fields[0]
        if output_fields:
            output_field = output_fields[0]
    
    # Generate LM configuration code based on target_lm
    target_lm = _session.target_lm or "ollama/gemma3:4b"
    if "/" in target_lm:
        provider, model = target_lm.split("/", 1)
    else:
        provider = "openai"
        model = target_lm
    
    if provider == "ollama":
        lm_config_code = f'lm = dspy.LM("ollama_chat/{model}", api_base="http://localhost:11434")'
    elif provider == "anthropic":
        lm_config_code = f'lm = dspy.LM("anthropic/{model}", api_key=os.getenv("ANTHROPIC_API_KEY"))'
    elif provider == "google" or provider == "gemini":
        lm_config_code = f'lm = dspy.LM("google/{model}", api_key=os.getenv("GOOGLE_API_KEY"))'
    else:
        lm_config_code = f'lm = dspy.LM("openai/{model}", api_key=os.getenv("OPENAI_API_KEY"))'
    
    # Generate code with demos
    code = f'''"""
DSPy Optimized Program
Generated: {datetime.now().isoformat()}
Optimizer: {_session.optimizer_type}
Target LM: {target_lm}
Dev Accuracy: {_session.eval_results.get('dev_accuracy', 0):.1%}
Test Accuracy: {_session.eval_results.get('test_accuracy', 0):.1%}
"""

import os
import dspy

# Signature
{_session.signature_code or "# Signature not available"}
{demos_code}

class OptimizedModule(dspy.Module):
    """
    Optimized DSPy module with few-shot demonstrations.
    
    Usage:
        # Configure LM
        {lm_config_code}
        dspy.configure(lm=lm)
        
        # Create and use module
        module = OptimizedModule()
        result = module({input_field}="your input here")
        print(result.{output_field})
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.{_session.module_type}(TaskSignature)
        
        # Load optimized few-shot demos
        if 'DEMOS' in globals() and DEMOS:
            self.predictor.demos = [
                dspy.Example(**demo).with_inputs("{input_field}")
                for demo in DEMOS
            ]
    
    def forward(self, {input_field}: str) -> dspy.Prediction:
        return self.predictor({input_field}={input_field})


# Quick test
if __name__ == "__main__":
    # Configure your LM (same as used during optimization)
    {lm_config_code}
    dspy.configure(lm=lm)
    
    # Test the module
    module = OptimizedModule()
    
    test_inputs = [
        "I want to cancel my subscription",
        "Where is my package?",
        "Please refund my money",
    ]
    
    print("Testing optimized module:")
    for inp in test_inputs:
        result = module({input_field}=inp)
        print(f"  {{inp[:40]}}... -> {{result.{output_field}}}")
'''
    
    _session.program_code = code
    
    artifact_dir = Path("data/artifacts") / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    with open(artifact_dir / "program.py", "w") as f:
        f.write(code)
    
    # Also save demos separately as JSON
    if demos_list:
        with open(artifact_dir / "demos.json", "w") as f:
            json.dump(demos_list, f, indent=2, ensure_ascii=False)
    
    metadata = {
        "artifact_id": artifact_id,
        "created_at": datetime.now().isoformat(),
        "task_analysis": _session.task_analysis,
        "optimizer_type": _session.optimizer_type,
        "eval_results": _session.eval_results,
        "target_lm": _session.target_lm,
        "num_demos": len(demos_list)
    }
    
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "status": "success",
        "artifact_id": artifact_id,
        "artifact_path": str(artifact_dir),
        "program_code": code,
        "num_demos": len(demos_list)
    }


async def get_session_status() -> Dict:
    """Get session status."""
    return {
        "session_id": _session.session_id,
        "task_analysis": _session.task_analysis,
        "signature_defined": _session.signature_class is not None,
        "module_type": _session.module_type if _session.module_instance else None,
        "lm_configured": _session.target_lm,
        "dataset_size": len(_session.dataset),
        "train_size": len(_session.train_set),
        "optimizer_type": _session.optimizer_type,
        "compiled": _session.compiled_program is not None,
        "eval_results": _session.eval_results
    }


# Tool registry for MCP-like interface
TOOLS = {
    "analyze_task": analyze_task,
    "create_signature": create_signature,
    "create_module": create_module,
    "configure_lm": configure_lm,
    "prepare_dataset": prepare_dataset,
    "select_optimizer": select_optimizer,
    "run_optimization": run_optimization,
    "evaluate_program": evaluate_program,
    "export_program": export_program,
    "get_session_status": get_session_status,
}


TOOLS_SCHEMA = [
    {"name": "analyze_task", "description": "Analyze business task"},
    {"name": "create_signature", "description": "Create DSPy Signature"},
    {"name": "create_module", "description": "Create DSPy module"},
    {"name": "configure_lm", "description": "Configure Language Model"},
    {"name": "prepare_dataset", "description": "Prepare dataset splits"},
    {"name": "select_optimizer", "description": "Select optimizer"},
    {"name": "run_optimization", "description": "Run DSPy optimization"},
    {"name": "evaluate_program", "description": "Evaluate program"},
    {"name": "export_program", "description": "Export program"},
    {"name": "get_session_status", "description": "Get session status"},
]
