"""
DSPy Prompt Optimizer - FastAPI Backend
Standalone application for automated prompt optimization using DSPy framework.
"""

import json
import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Import DSPy orchestrator
from dspy_engine import DSPyEngine, AgentState
from llm_providers import LLMProviderFactory, OllamaProvider
from langchain_agent import DSPyLangChainAgent
from mcp_agent import DSPyMCPAgent
from hf_dataset_provider import search_hf_datasets, import_hf_dataset, inspect_hf_dataset

# ==================== Pydantic Models ====================

class OrchestratorRequest(BaseModel):
    """Request for DSPy orchestration."""
    business_task: str = Field(..., description="Description of the task to optimize")
    target_lm: str = Field(..., description="Target LLM for inference (e.g., openai/gpt-4o)")
    optimizer_lm: str = Field(default="openai/gpt-4o-mini", description="LLM for optimization process")
    dataset: List[Dict[str, str]] = Field(..., description="Training examples [{input, output}]")
    quality_profile: str = Field(default="BALANCED", description="FAST_CHEAP, BALANCED, HIGH_QUALITY")
    optimizer_strategy: str = Field(default="auto", description="Optimizer: auto, BootstrapFewShot, MIPROv2, COPRO")
    use_agent: bool = Field(default=True, description="Use LangChain agent for intelligent orchestration")


class TestArtifactRequest(BaseModel):
    """Request to test an optimized artifact."""
    artifact_id: str
    input_text: str
    target_lm: str
    program_code: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response with available models."""
    models: List[str]
    provider: str


# ==================== FastAPI App ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan handler."""
    # Startup
    artifacts_dir = Path("data/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown
    pass


app = FastAPI(
    title="DSPy Prompt Optimizer",
    description="Automated prompt optimization using DSPy framework",
    version="1.0.0",
    lifespan=lifespan
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
    except Exception as e:
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
        "base_url": ollama.base_url
    }


@app.post("/api/dspy/orchestrate")
async def orchestrate_dspy(request: OrchestratorRequest):
    """
    Run DSPy orchestration with streaming updates.
    Returns Server-Sent Events with step updates.
    Uses LangChain agent for intelligent orchestration when use_agent=True.
    """
    print(f"[ORCHESTRATE] use_agent={request.use_agent}, optimizer_lm={request.optimizer_lm}")
    async def event_generator():
        try:
            if request.use_agent:
                # Use MCP-based agent for intelligent orchestration
                print("[ORCHESTRATE] Using DSPyMCPAgent...")
                agent = DSPyMCPAgent(
                    agent_model=request.optimizer_lm
                )
                
                async for event in agent.run_async(
                    business_task=request.business_task,
                    target_lm=request.target_lm,
                    dataset=request.dataset,
                    quality_profile=request.quality_profile
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                # Use simple DSPyEngine (no agent)
                engine = DSPyEngine(
                    optimizer_model=request.optimizer_lm,
                    artifacts_dir="data/artifacts"
                )
                
                async for event in engine.run_async(
                    business_task=request.business_task,
                    target_lm=request.target_lm,
                    dataset=request.dataset,
                    quality_profile=request.quality_profile,
                    optimizer_strategy=request.optimizer_strategy
                ):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_event = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/dspy/test")
async def test_artifact(request: TestArtifactRequest):
    """Test an optimized artifact with new input."""
    try:
        engine = DSPyEngine(artifacts_dir="data/artifacts")
        result = await engine.test_artifact(
            artifact_id=request.artifact_id,
            input_text=request.input_text,
            target_lm=request.target_lm,
            program_code=request.program_code
        )
        return {"output": result, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/artifacts")
async def list_artifacts():
    """List all saved artifacts."""
    artifacts_dir = Path("data/artifacts")
    artifacts = []
    
    if artifacts_dir.exists():
        for item in artifacts_dir.iterdir():
            if item.is_dir():
                metadata_file = item / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        artifacts.append(metadata)
    
    return {"artifacts": sorted(artifacts, key=lambda x: x.get("created_at", ""), reverse=True)}


@app.get("/api/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get a specific artifact."""
    artifact_dir = Path(f"data/artifacts/{artifact_id}")
    
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
async def inspect_hf_catalog_dataset(payload: Dict[str, Any]):
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
async def import_hf_catalog_dataset(payload: Dict[str, Any]):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
