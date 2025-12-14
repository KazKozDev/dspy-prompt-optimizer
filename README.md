# DSPy Prompt Optimizer

**AI agent that writes DSPy programs for you.**

Task description + examples → Agent builds DSPy pipeline → Optimized prompts + Python code.

<img width="1161" height="947" alt="Screenshot 2025-12-14 at 23 03 57" src="https://github.com/user-attachments/assets/971d5c98-cc4f-4ec6-b675-93b759fc8de4" />

---

## Agent Architecture

```
┌─────────────────────────────────────┐
│       LangChain ReAct Agent         │
├─────────────────────────────────────┤
│ analyze_task      → task type       │
│ create_signature  → DSPy Signature  │
│ select_module     → Predict/CoT     │
│ select_optimizer  → BootstrapFewShot│
│ run_optimization  → DSPy compile    │
│ export_program    → Python code     │
└─────────────────────────────────────┘
```

Agent decides each step based on task analysis and intermediate results. Adapts strategy if optimization fails or accuracy is low.

---

## Quick Start

```bash
# Backend
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && echo "OPENAI_API_KEY=sk-..." > .env
python app.py

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 → Describe task → Add examples → Run.

---

## Supported Providers

OpenAI, Anthropic, Google Gemini, Ollama (local). Models fetched dynamically from APIs.

---

## License

MIT
