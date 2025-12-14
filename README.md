# DSPy Prompt Optimizer

Automated prompt engineering for production LLM applications.

---

## Overview

DSPy Prompt Optimizer transforms how you build LLM applications. Instead of manual prompt engineering, define your task and provide examples — the system automatically discovers optimal prompts through systematic experimentation.

**Key capabilities:**
- Automated prompt optimization using Stanford's DSPy framework
- Support for OpenAI, Anthropic, Google Gemini, and local Ollama models
- Real-time optimization progress with ReAct reasoning steps
- Export production-ready Python code

---

## Requirements

- Python 3.10+
- Node.js 18+
- API key for at least one LLM provider (or Ollama installed locally)

---

## Installation

**Clone and setup backend:**

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Setup frontend:**

```bash
cd frontend
npm install
```

---

## Configuration

Create `backend/.env` with your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

Keys can also be configured through the application settings.

---

## Usage

**Start the application:**

```bash
# Terminal 1: Backend
cd backend && source venv/bin/activate && python app.py

# Terminal 2: Frontend  
cd frontend && npm run dev
```

Open `http://localhost:5173`

**Optimization workflow:**

1. Select target model (production) and optimizer model
2. Describe your task in natural language
3. Provide training examples as JSON:
   ```json
   [
     {"input": "...", "output": "..."},
     {"input": "...", "output": "..."}
   ]
   ```
4. Run optimization
5. Review metrics, test results, export code

---

## Optimization Strategies

| Strategy | Use Case | Examples Needed |
|----------|----------|-----------------|
| Auto | Let the system decide | Any |
| BootstrapFewShot | Quick optimization | 10–50 |
| MIPROv2 | Maximum quality | 50+ |
| COPRO | Instruction tuning | 30+ |

---

## Quality Profiles

| Profile | Iterations | Best For |
|---------|------------|----------|
| Fast | Minimal | Prototyping |
| Balanced | Moderate | Production |
| Quality | Maximum | Critical tasks |

---

## When to Use

**Ideal for:**
- Classification and extraction tasks
- Multi-step reasoning pipelines
- RAG applications
- Cross-model prompt adaptation

**Not recommended for:**
- Creative writing tasks
- Datasets with fewer than 10 examples
- One-off prompt experiments

---

## Project Structure

```
backend/
  app.py                 # API server
  dspy_engine.py         # Optimization engine
  orchestrator.py        # LangChain agent orchestration

frontend/
  src/App.tsx            # Main application
  src/api.ts             # API client
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/{provider}` | GET | Available models |
| `/api/dspy/orchestrate` | POST | Run optimization (SSE) |
| `/api/dspy/test` | POST | Test artifact |
| `/api/ollama/status` | GET | Ollama connection status |

---

## License

MIT

---

Built with [DSPy](https://github.com/stanfordnlp/dspy) · [FastAPI](https://fastapi.tiangolo.com) · [React](https://react.dev)
