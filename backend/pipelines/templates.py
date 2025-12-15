"""Pipeline Templates - Pre-built pipeline configurations for common use cases."""

from typing import Any

from .builder import PipelineBuilder

PIPELINE_TEMPLATES: dict[str, dict[str, Any]] = {
    "simple_predict": {
        "name": "Simple Predict",
        "description": "Single-step prediction without chain of thought",
        "use_cases": ["classification", "simple extraction", "labeling"],
        "stages": [
            {
                "name": "predict",
                "module_type": "predict",
                "input_fields": ["text"],
                "output_fields": ["result"],
                "description": "Direct prediction",
            }
        ],
    },
    "chain_of_thought": {
        "name": "Chain of Thought",
        "description": "Single-step with reasoning before answer",
        "use_cases": ["reasoning", "complex classification", "analysis"],
        "stages": [
            {
                "name": "reason",
                "module_type": "chain_of_thought",
                "input_fields": ["text"],
                "output_fields": ["reasoning", "result"],
                "description": "Reason step by step then answer",
            }
        ],
    },
    "rag_basic": {
        "name": "Basic RAG",
        "description": "Retrieve relevant context then generate answer",
        "use_cases": ["question answering", "knowledge base", "document QA"],
        "stages": [
            {
                "name": "retrieve",
                "module_type": "retrieve",
                "input_fields": ["query"],
                "output_fields": ["passages"],
                "description": "Retrieve relevant passages",
                "config": {"k": 5},
            },
            {
                "name": "generate",
                "module_type": "chain_of_thought",
                "input_fields": ["query", "passages"],
                "output_fields": ["answer"],
                "description": "Generate answer from context",
            },
        ],
    },
    "rag_with_rerank": {
        "name": "RAG with Reranking",
        "description": "Retrieve, rerank, then generate",
        "use_cases": ["high-quality QA", "precise retrieval"],
        "stages": [
            {
                "name": "retrieve",
                "module_type": "retrieve",
                "input_fields": ["query"],
                "output_fields": ["passages"],
                "description": "Initial retrieval",
                "config": {"k": 10},
            },
            {
                "name": "rerank",
                "module_type": "chain_of_thought",
                "input_fields": ["query", "passages"],
                "output_fields": ["ranked_passages"],
                "description": "Rerank passages by relevance",
            },
            {
                "name": "generate",
                "module_type": "chain_of_thought",
                "input_fields": ["query", "ranked_passages"],
                "output_fields": ["answer"],
                "description": "Generate answer from top passages",
            },
        ],
    },
    "outline_draft_revise": {
        "name": "Outline → Draft → Revise",
        "description": "Three-stage content generation with planning and revision",
        "use_cases": ["article writing", "report generation", "content creation"],
        "stages": [
            {
                "name": "outline",
                "module_type": "chain_of_thought",
                "input_fields": ["topic", "requirements"],
                "output_fields": ["outline"],
                "description": "Create structured outline",
            },
            {
                "name": "draft",
                "module_type": "chain_of_thought",
                "input_fields": ["topic", "outline"],
                "output_fields": ["draft"],
                "description": "Write initial draft from outline",
            },
            {
                "name": "revise",
                "module_type": "chain_of_thought",
                "input_fields": ["draft", "requirements"],
                "output_fields": ["final"],
                "description": "Revise and polish the draft",
            },
        ],
    },
    "extract_summarize": {
        "name": "Extract → Summarize",
        "description": "Extract key information then summarize",
        "use_cases": ["document processing", "information extraction", "summarization"],
        "stages": [
            {
                "name": "extract",
                "module_type": "chain_of_thought",
                "input_fields": ["text"],
                "output_fields": ["entities", "key_points"],
                "description": "Extract entities and key points",
            },
            {
                "name": "summarize",
                "module_type": "chain_of_thought",
                "input_fields": ["text", "key_points"],
                "output_fields": ["summary"],
                "description": "Generate summary from key points",
            },
        ],
    },
    "classify_explain": {
        "name": "Classify → Explain",
        "description": "Classify then generate explanation",
        "use_cases": ["explainable classification", "decision support"],
        "stages": [
            {
                "name": "classify",
                "module_type": "predict",
                "input_fields": ["text"],
                "output_fields": ["category", "confidence"],
                "description": "Classify the input",
            },
            {
                "name": "explain",
                "module_type": "chain_of_thought",
                "input_fields": ["text", "category"],
                "output_fields": ["explanation"],
                "description": "Explain the classification",
            },
        ],
    },
    "react_agent": {
        "name": "ReAct Agent",
        "description": "Reasoning and acting agent with tools",
        "use_cases": ["complex reasoning", "tool use", "multi-step tasks"],
        "stages": [
            {
                "name": "agent",
                "module_type": "react",
                "input_fields": ["task"],
                "output_fields": ["result", "actions_taken"],
                "description": "ReAct agent with tool access",
            }
        ],
    },
    "multi_hop_qa": {
        "name": "Multi-Hop QA",
        "description": "Answer complex questions requiring multiple retrieval steps",
        "use_cases": ["complex QA", "multi-document reasoning"],
        "stages": [
            {
                "name": "decompose",
                "module_type": "chain_of_thought",
                "input_fields": ["question"],
                "output_fields": ["sub_questions"],
                "description": "Break down into sub-questions",
            },
            {
                "name": "retrieve_1",
                "module_type": "retrieve",
                "input_fields": ["query"],
                "output_fields": ["passages_1"],
                "description": "First retrieval hop",
                "config": {"k": 3},
            },
            {
                "name": "intermediate",
                "module_type": "chain_of_thought",
                "input_fields": ["sub_questions", "passages_1"],
                "output_fields": ["intermediate_answer", "next_query"],
                "description": "Answer sub-questions, formulate next query",
            },
            {
                "name": "retrieve_2",
                "module_type": "retrieve",
                "input_fields": ["query"],
                "output_fields": ["passages_2"],
                "description": "Second retrieval hop",
                "config": {"k": 3},
            },
            {
                "name": "synthesize",
                "module_type": "chain_of_thought",
                "input_fields": ["question", "intermediate_answer", "passages_2"],
                "output_fields": ["final_answer"],
                "description": "Synthesize final answer",
            },
        ],
    },
    "code_generation": {
        "name": "Code Generation Pipeline",
        "description": "Plan, implement, and review code",
        "use_cases": ["code generation", "programming tasks"],
        "stages": [
            {
                "name": "plan",
                "module_type": "chain_of_thought",
                "input_fields": ["task", "language"],
                "output_fields": ["approach", "steps"],
                "description": "Plan the implementation approach",
            },
            {
                "name": "implement",
                "module_type": "chain_of_thought",
                "input_fields": ["task", "approach", "steps"],
                "output_fields": ["code"],
                "description": "Write the code",
            },
            {
                "name": "review",
                "module_type": "chain_of_thought",
                "input_fields": ["task", "code"],
                "output_fields": ["issues", "improved_code"],
                "description": "Review and improve the code",
            },
        ],
    },
}


def get_template(template_name: str) -> dict[str, Any] | None:
    """Get a pipeline template by name.

    Args:
        template_name: Name of the template

    Returns:
        Template dict or None if not found
    """
    return PIPELINE_TEMPLATES.get(template_name)


def list_templates() -> list[dict[str, str]]:
    """List all available templates.

    Returns:
        List of template info dicts
    """
    return [
        {
            "name": name,
            "display_name": template["name"],
            "description": template["description"],
            "use_cases": template["use_cases"],
            "num_stages": len(template["stages"]),
        }
        for name, template in PIPELINE_TEMPLATES.items()
    ]


def build_from_template(
    template_name: str, tools: list | None = None
) -> PipelineBuilder | None:
    """Create a PipelineBuilder from a template.

    Args:
        template_name: Name of the template
        tools: Optional tools for ReAct stages

    Returns:
        PipelineBuilder or None if template not found
    """
    template = get_template(template_name)
    if not template:
        return None

    builder = PipelineBuilder()

    for stage in template["stages"]:
        builder.add_stage(
            name=stage["name"],
            module_type=stage["module_type"],
            input_fields=stage["input_fields"],
            output_fields=stage["output_fields"],
            description=stage.get("description", ""),
            **stage.get("config", {}),
        )

    if tools:
        builder.with_tools(tools)

    return builder


def suggest_template(
    task_type: str, needs_retrieval: bool = False, needs_tools: bool = False
) -> str:
    """Suggest a template based on task characteristics.

    Args:
        task_type: Type of task (classification, extraction, qa, generation, etc.)
        needs_retrieval: Whether task needs retrieval
        needs_tools: Whether task needs tool use

    Returns:
        Suggested template name
    """
    if needs_tools:
        return "react_agent"

    if needs_retrieval:
        if task_type in ["qa", "question_answering"]:
            return "rag_basic"
        return "rag_with_rerank"

    if task_type == "classification":
        return "classify_explain"

    if task_type == "extraction":
        return "extract_summarize"

    if task_type == "summarization":
        return "extract_summarize"

    if task_type in ["generation", "writing"]:
        return "outline_draft_revise"

    if task_type == "code":
        return "code_generation"

    if task_type == "reasoning":
        return "chain_of_thought"

    return "chain_of_thought"
