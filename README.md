![baner](https://github.com/user-attachments/assets/6a761950-e8f2-4761-82f2-adc1767dda8c)

# DSPy Prompt Optimizer

**AI agent that writes DSPy programs for you.**

Task description + examples → Agent builds DSPy pipeline → Optimized prompts + Python code.

<img width="1161" height="947" alt="Screenshot 2025-12-14 at 23 03 57" src="https://github.com/user-attachments/assets/971d5c98-cc4f-4ec6-b675-93b759fc8de4" />

---

## Features

- **Hybrid Engine** — Meta-Agent auto-configures pipeline, metrics, optimizer
- **Multiple Pipelines** — Predict, Chain-of-Thought, ReAct, RAG
- **ReAct Tools** — Calculator, Web Search, Python REPL, Wikipedia
- **RAG/Retrieval** — FAISS and ChromaDB vector search
- **LLM-as-Judge** — Evaluation via GPT-5/Claude with custom criteria
- **Teacher-Student Distillation** — Generate training data from large models
- **HuggingFace Import** — Load datasets directly from HF Hub

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Hybrid DSPy Engine                     │
├─────────────────────────────────────────────────────────┤
│  Meta-Agent (AUTO mode)        Manual Overrides          │
│  ├─ TaskAnalyzer               ├─ Pipeline: Predict/CoT/ │
│  ├─ PipelineSelector              ReAct/RAG              │
│  ├─ MetricSelector             ├─ Metric: Exact/F1/      │
│  ├─ OptimizerSelector             LLM Judge              │
│  └─ ToolSelector               ├─ Tools: calc/search/... │
│                                └─ Distillation: ON/OFF   │
├─────────────────────────────────────────────────────────┤
│  DSPy Compilation                                        │
│  ├─ BootstrapFewShot / MIPROv2 / COPRO                  │
│  ├─ Metric evaluation                                    │
│  └─ Optimized program export                             │
└─────────────────────────────────────────────────────────┘
```

**AUTO mode**: Agent analyzes task and configures everything automatically.  
**MANUAL mode**: You choose pipeline, metric, tools, and advanced options.

---

## Quick Start

### macOS (Double-Click)

Double-click `DSPy Optimizer.command` — installs dependencies and starts both servers.

### Manual

```bash
# Backend
cd backend && pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python app.py

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:3000 → Configure API keys → Describe task → Add examples → Run.

---

## Modes & Options

| Mode | Pipeline | What it does |
|------|----------|--------------|
| Auto | Agent decides | Analyzes task, picks best pipeline/metric/optimizer |
| Predict | `dspy.Predict` | Simple input→output |
| CoT | `dspy.ChainOfThought` | Step-by-step reasoning |
| ReAct | `dspy.ReAct` | Agent with tools (calc, search, python, wiki) |
| RAG | Retrieve + Generate | Vector search + generation |

| Metric | Use case |
|--------|----------|
| Exact Match | Classification, short answers |
| Token F1 | Extraction, partial matches |
| LLM Judge | Generation quality, complex outputs |

| Advanced | Description |
|----------|-------------|
| Distillation | Generate training data from GPT-5/Claude |
| Custom Criteria | Define evaluation rules for LLM Judge |

---

## Supported Providers

- **OpenAI** — GPT-5, GPT-5-mini
- **Anthropic** — Claude 3.5 Sonnet, Claude 3 Haiku
- **Google** — Gemini Pro, Gemini Flash
- **Ollama** — Llama 3, Mistral, Qwen (local)

---

## Project Structure

```
backend/
├── agent/           # Meta-Agent & selectors
├── metrics/         # Exact Match, F1, LLM Judge, Semantic
├── pipelines/       # Pipeline builder & templates
├── tools/           # ReAct tools (calc, search, python, wiki)
├── retrieval/       # FAISS & ChromaDB retrievers
├── distillation/    # Teacher-Student distillation
├── hybrid_engine.py # Main orchestration engine
└── app.py           # FastAPI backend

frontend/
├── src/App.tsx      # React UI
└── src/api.ts       # API client
```

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
