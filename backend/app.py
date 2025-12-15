"""DSPy Prompt Optimizer - FastAPI Backend
Standalone application for automated prompt optimization using DSPy framework.
"""

import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(dotenv_path=str(REPO_ROOT / ".env"))

from utils.logger import safe_setup_default_logging, setup_logging
from utils.settings import get_settings

_SETTINGS = get_settings()
try:
    setup_logging(_SETTINGS.logging.config_path)
except Exception:
    safe_setup_default_logging()

DEFAULT_OPTIMIZER_LM = f"openai/{_SETTINGS.model_defaults.openai_optimizer}"
ARTIFACTS_DIR = REPO_ROOT / "data" / "artifacts"

# Import DSPy orchestrator
from dspy_engine import DSPyEngine
from hf_dataset_provider import (
    import_hf_dataset,
    inspect_hf_dataset,
    search_hf_datasets,
)
from llm_providers import LLMProviderFactory, OllamaProvider
from mcp_agent import DSPyMCPAgent

# New Hybrid Engine with Meta-Agent
try:
    from hybrid_engine import HybridDSPyEngine

    HYBRID_ENGINE_AVAILABLE = True
except ImportError:
    HYBRID_ENGINE_AVAILABLE = False

# ==================== Pydantic Models ====================


class OrchestratorRequest(BaseModel):
    """Request for DSPy orchestration."""

    business_task: str = Field(..., description="Description of the task to optimize")
    target_lm: str = Field(
        ..., description="Target LLM for inference (e.g., openai/gpt-5)"
    )
    optimizer_lm: str = Field(
        default=DEFAULT_OPTIMIZER_LM, description="LLM for optimization process"
    )
    dataset: list[dict[str, str]] = Field(
        ..., description="Training examples [{input, output}]"
    )
    quality_profile: str = Field(
        default="BALANCED", description="FAST_CHEAP, BALANCED, HIGH_QUALITY"
    )
    optimizer_strategy: str = Field(
        default="auto", description="Optimizer: auto, BootstrapFewShot, MIPROv2, COPRO"
    )
    use_agent: bool = Field(
        default=True, description="Use LangChain agent for intelligent orchestration"
    )
    # New Hybrid Mode fields
    mode: str = Field(
        default="auto",
        description="Mode: auto (agent decides) or manual (user overrides)",
    )
    use_hybrid: bool = Field(
        default=False, description="Use new Hybrid Engine with Meta-Agent"
    )
    manual_overrides: dict[str, Any] | None = Field(
        default=None, description="Manual configuration overrides"
    )


class TestArtifactRequest(BaseModel):
    """Request to test an optimized artifact."""

    artifact_id: str
    input_text: str
    target_lm: str
    program_code: str | None = None


class ModelsResponse(BaseModel):
    """Response with available models."""

    models: list[str]
    provider: str


# ==================== FastAPI App ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan handler."""
    # Startup
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown
    pass


app = FastAPI(
    title="DSPy Prompt Optimizer",
    description="Automated prompt optimization using DSPy framework",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API Endpoints ====================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/models/{provider}")
async def get_models(provider: str) -> ModelsResponse:
    """Get available models for a provider (dynamically for Ollama)."""
    try:
        llm_provider = LLMProviderFactory.get_provider(provider)
        models = await llm_provider.list_models()
        return ModelsResponse(models=models, provider=provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # Fallback to empty list on error
        return ModelsResponse(models=[], provider=provider)


@app.get("/api/ollama/status")
async def ollama_status():
    """Check if Ollama is running and available."""
    ollama = OllamaProvider()
    is_available = await ollama.is_available()
    models = await ollama.list_models() if is_available else []
    return {
        "available": is_available,
        "models_count": len(models),
        "base_url": ollama.base_url,
    }


@app.post("/api/dspy/orchestrate")
async def orchestrate_dspy(request: OrchestratorRequest):
    """Run DSPy orchestration with streaming updates.
    Returns Server-Sent Events with step updates.

    Supports three modes:
    1. use_hybrid=True: New Hybrid Engine with Meta-Agent (AUTO/MANUAL modes)
    2. use_agent=True: Legacy MCP-based agent
    3. use_agent=False: Simple DSPyEngine
    """
    print(
        f"[ORCHESTRATE] use_hybrid={request.use_hybrid}, use_agent={request.use_agent}, mode={request.mode}"
    )

    async def event_generator():
        try:
            if request.use_hybrid and HYBRID_ENGINE_AVAILABLE:
                # NEW: Use Hybrid Engine with Meta-Agent
                print(f"[ORCHESTRATE] Using HybridDSPyEngine in {request.mode} mode...")
                engine = HybridDSPyEngine(
                    optimizer_model=request.optimizer_lm,
                    artifacts_dir=str(ARTIFACTS_DIR),
                )

                async for event in engine.run_async(
                    business_task=request.business_task,
                    target_lm=request.target_lm,
                    dataset=request.dataset,
                    quality_profile=request.quality_profile,
                    mode=request.mode,
                    manual_overrides=request.manual_overrides,
                ):
                    yield f"data: {json.dumps(event, default=str)}\n\n"

            elif request.use_agent:
                # Legacy: Use MCP-based agent for intelligent orchestration
                print("[ORCHESTRATE] Using DSPyMCPAgent...")
                agent = DSPyMCPAgent(agent_model=request.optimizer_lm)

                async for event in agent.run_async(
                    business_task=request.business_task,
                    target_lm=request.target_lm,
                    dataset=request.dataset,
                    quality_profile=request.quality_profile,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                # Legacy: Use simple DSPyEngine (no agent)
                engine = DSPyEngine(
                    optimizer_model=request.optimizer_lm,
                    artifacts_dir=str(ARTIFACTS_DIR),
                )

                async for event in engine.run_async(
                    business_task=request.business_task,
                    target_lm=request.target_lm,
                    dataset=request.dataset,
                    quality_profile=request.quality_profile,
                    optimizer_strategy=request.optimizer_strategy,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            import traceback

            traceback.print_exc()
            error_event = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/dspy/test")
async def test_artifact(request: TestArtifactRequest):
    """Test an optimized artifact with new input."""
    try:
        engine = DSPyEngine(artifacts_dir="data/artifacts")
        engine.artifacts_dir = ARTIFACTS_DIR
        result = await engine.test_artifact(
            artifact_id=request.artifact_id,
            input_text=request.input_text,
            target_lm=request.target_lm,
            program_code=request.program_code,
        )
        return {"output": result, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/artifacts")
async def list_artifacts():
    """List all saved artifacts."""
    artifacts_dir = ARTIFACTS_DIR
    artifacts = []

    if artifacts_dir.exists():
        for item in artifacts_dir.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        artifacts.append(metadata)

    return {
        "artifacts": sorted(
            artifacts, key=lambda x: x.get("created_at", ""), reverse=True
        )
    }


@app.get("/api/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get a specific artifact."""
    artifact_dir = ARTIFACTS_DIR / artifact_id

    if not artifact_dir.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    metadata_file = artifact_dir / "metadata.json"
    program_file = artifact_dir / "program.py"

    result = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            result["metadata"] = json.load(f)
    if program_file.exists():
        with open(program_file) as f:
            result["program_code"] = f.read()

    return result


# ==================== HuggingFace Dataset Catalog ====================


@app.get("/api/datasets/catalog/hf/search")
async def search_hf_catalog(q: str, limit: int = 20):
    """Search Hugging Face datasets catalog."""
    try:
        results = search_hf_datasets(q, limit=limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/datasets/catalog/hf/inspect")
async def inspect_hf_catalog_dataset(payload: dict[str, Any]):
    """Inspect a HuggingFace dataset to get columns and suggested mapping."""
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    config_name = payload.get("config_name")
    split = payload.get("split", "train")

    try:
        info = inspect_hf_dataset(dataset_id, config_name=config_name, split=split)
        return info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/datasets/catalog/hf/import")
async def import_hf_catalog_dataset(payload: dict[str, Any]):
    """Import a dataset from Hugging Face.

    Returns dataset in DSPy format: [{input, output}, ...]
    """
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    config_name = payload.get("config_name")
    split = payload.get("split", "train")
    input_key = payload.get("input_key")
    output_key = payload.get("output_key")
    max_items = payload.get("max_items", 500)

    try:
        imported = import_hf_dataset(
            dataset_id,
            config_name=config_name,
            split=split,
            input_key=input_key,
            output_key=output_key,
            max_items=max_items,
        )
        return imported
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Hybrid Engine API Endpoints ====================


@app.get("/api/hybrid/templates")
async def list_pipeline_templates():
    """List available pipeline templates for manual configuration."""
    try:
        from pipelines.templates import list_templates

        templates = list_templates()
        return {"templates": templates}
    except ImportError:
        return {"templates": [], "error": "Pipeline templates not available"}


@app.get("/api/hybrid/tools")
async def list_available_tools():
    """List available tools for ReAct agents."""
    try:
        from tools.builtin import register_all_builtin_tools
        from tools.registry import get_registry

        register_all_builtin_tools()
        registry = get_registry()
        tools = registry.list_tools()
        return {"tools": tools}
    except ImportError:
        return {"tools": [], "error": "Tools not available"}


@app.get("/api/hybrid/metrics")
async def list_available_metrics():
    """List available evaluation metrics."""
    metrics = [
        {
            "type": "exact_match",
            "name": "Exact Match",
            "description": "Exact string match between prediction and expected output",
            "best_for": ["classification", "routing", "labeling"],
        },
        {
            "type": "token_f1",
            "name": "Token F1",
            "description": "Token-level F1 score for partial matches",
            "best_for": ["extraction", "QA", "summarization"],
        },
        {
            "type": "semantic_similarity",
            "name": "Semantic Similarity",
            "description": "Embedding-based semantic similarity",
            "best_for": ["paraphrase", "generation", "summarization"],
            "requires": "sentence-transformers",
        },
        {
            "type": "llm_judge",
            "name": "LLM-as-Judge",
            "description": "Use LLM to evaluate response quality",
            "best_for": ["generation", "reasoning", "complex tasks"],
            "subtypes": ["correctness", "faithfulness", "coherence", "custom"],
        },
    ]
    return {"metrics": metrics}


@app.get("/api/hybrid/optimizers")
async def list_available_optimizers():
    """List available DSPy optimizers."""
    optimizers = [
        {
            "type": "BootstrapFewShot",
            "name": "Bootstrap Few-Shot",
            "description": "Fast, works with 10-50 examples",
            "min_examples": 10,
            "speed": "fast",
        },
        {
            "type": "BootstrapFewShotWithRandomSearch",
            "name": "Bootstrap + Random Search",
            "description": "Better quality, needs 20+ examples",
            "min_examples": 20,
            "speed": "medium",
        },
        {
            "type": "MIPROv2",
            "name": "MIPRO v2",
            "description": "Best quality, needs 50+ examples",
            "min_examples": 50,
            "speed": "slow",
        },
        {
            "type": "COPRO",
            "name": "COPRO",
            "description": "Instruction optimization",
            "min_examples": 30,
            "speed": "medium",
        },
    ]
    return {"optimizers": optimizers}


@app.post("/api/hybrid/analyze")
async def analyze_task(payload: dict[str, Any]):
    """Analyze a task without running optimization.

    Returns agent's analysis and recommended configuration.
    """
    business_task = payload.get("business_task", "")
    dataset = payload.get("dataset", [])

    if not business_task:
        raise HTTPException(status_code=400, detail="business_task is required")

    try:
        from agent import MetaAgent

        agent = MetaAgent()
        config = agent.configure(
            business_task=business_task,
            target_model="",
            dataset=dataset,
            quality_profile="BALANCED",
            mode="auto",
        )

        summary = agent.get_configuration_summary(config)
        warnings = agent.validate_config(config)

        return {
            "analysis": summary,
            "warnings": warnings,
            "reasoning": config.agent_reasoning,
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Meta-Agent not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hybrid/status")
async def hybrid_engine_status():
    """Check if Hybrid Engine is available."""
    return {
        "available": HYBRID_ENGINE_AVAILABLE,
        "features": {
            "meta_agent": HYBRID_ENGINE_AVAILABLE,
            "llm_judge": HYBRID_ENGINE_AVAILABLE,
            "multi_stage_pipelines": HYBRID_ENGINE_AVAILABLE,
            "react_tools": HYBRID_ENGINE_AVAILABLE,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
