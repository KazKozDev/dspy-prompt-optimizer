"""Task Analyzer - Analyzes business task to extract requirements."""

from ..config import ComplexityLevel, TaskAnalysis, TaskType


class TaskAnalyzer:
    """Analyzes business task description to determine:
    - Task type (classification, extraction, QA, etc.)
    - Domain (legal, finance, medical, etc.)
    - Complexity level
    - Required capabilities (retrieval, tools, multi-stage)
    """

    TASK_KEYWORDS = {
        TaskType.CLASSIFICATION: [
            "classify",
            "categorize",
            "label",
            "tag",
            "sentiment",
            "detect",
            "identify type",
            "determine category",
            "sort into",
        ],
        TaskType.EXTRACTION: [
            "extract",
            "parse",
            "find",
            "identify",
            "locate",
            "pull out",
            "get entities",
            "named entity",
            "ner",
            "information extraction",
        ],
        TaskType.SUMMARIZATION: [
            "summarize",
            "summarise",
            "condense",
            "brief",
            "tldr",
            "abstract",
            "key points",
            "main ideas",
            "synopsis",
        ],
        TaskType.QA: ["answer question", "qa", "q&a", "respond to", "answer based on"],
        TaskType.RAG: [
            "rag",
            "retrieval",
            "search",
            "knowledge base",
            "document",
            "based on context",
            "from documents",
            "look up",
        ],
        TaskType.REASONING: [
            "reason",
            "think",
            "analyze",
            "solve",
            "figure out",
            "deduce",
            "logic",
            "math",
            "calculate",
            "compute",
            "step by step",
        ],
        TaskType.GENERATION: [
            "generate",
            "create",
            "write",
            "compose",
            "draft",
            "produce",
            "article",
            "essay",
            "story",
            "content",
        ],
        TaskType.ROUTING: [
            "route",
            "direct",
            "forward",
            "assign",
            "dispatch",
            "triage",
        ],
        TaskType.CODE: [
            "code",
            "program",
            "script",
            "function",
            "implement",
            "debug",
            "fix bug",
            "refactor",
            "python",
            "javascript",
        ],
    }

    DOMAIN_KEYWORDS = {
        "legal": [
            "legal",
            "law",
            "contract",
            "court",
            "attorney",
            "litigation",
            "compliance",
        ],
        "finance": [
            "finance",
            "financial",
            "bank",
            "investment",
            "stock",
            "trading",
            "accounting",
        ],
        "medical": [
            "medical",
            "health",
            "patient",
            "diagnosis",
            "clinical",
            "doctor",
            "symptom",
        ],
        "support": [
            "support",
            "customer",
            "ticket",
            "help desk",
            "service",
            "complaint",
        ],
        "engineering": [
            "code",
            "software",
            "engineering",
            "developer",
            "technical",
            "api",
            "system",
        ],
        "education": [
            "education",
            "learning",
            "student",
            "course",
            "teach",
            "academic",
        ],
        "marketing": [
            "marketing",
            "campaign",
            "brand",
            "advertising",
            "social media",
            "seo",
        ],
        "hr": [
            "hr",
            "human resources",
            "employee",
            "hiring",
            "recruitment",
            "onboarding",
        ],
    }

    COMPLEXITY_INDICATORS = {
        "high": [
            "complex",
            "multi-step",
            "multiple",
            "comprehensive",
            "detailed",
            "thorough",
            "in-depth",
            "advanced",
            "sophisticated",
        ],
        "low": ["simple", "basic", "quick", "straightforward", "easy", "single"],
    }

    TOOL_INDICATORS = {
        "calculator": [
            "calculate",
            "compute",
            "math",
            "arithmetic",
            "formula",
            "equation",
        ],
        "web_search": [
            "search",
            "look up",
            "find online",
            "current",
            "recent",
            "latest",
            "news",
        ],
        "python_repl": ["code", "execute", "run", "script", "data processing"],
        "wikipedia": [
            "wikipedia",
            "encyclopedia",
            "factual",
            "historical",
            "biography",
        ],
    }

    def analyze(
        self, task_description: str, dataset_sample: list[dict] | None = None
    ) -> TaskAnalysis:
        """Analyze task description and optional dataset sample.

        Args:
            task_description: Business task description
            dataset_sample: Optional sample of dataset items

        Returns:
            TaskAnalysis with inferred requirements
        """
        desc_lower = task_description.lower()

        task_type = self._infer_task_type(desc_lower)
        domain = self._infer_domain(desc_lower)
        complexity = self._infer_complexity(desc_lower, task_description)

        input_fields, output_fields = self._infer_fields(task_type, dataset_sample)

        needs_retrieval = self._needs_retrieval(desc_lower, task_type)
        needs_cot = self._needs_chain_of_thought(task_type, complexity)
        needs_tools = self._needs_tools(desc_lower)
        needs_multi_stage = self._needs_multi_stage(desc_lower, task_type, complexity)

        suggested_tools = self._suggest_tools(desc_lower) if needs_tools else []
        suggested_template = self._suggest_template(
            task_type, needs_retrieval, needs_tools, needs_multi_stage
        )

        reasoning = self._generate_reasoning(
            task_type, domain, complexity, needs_retrieval, needs_cot, needs_tools
        )

        return TaskAnalysis(
            task_type=task_type,
            domain=domain,
            complexity=complexity,
            input_fields=input_fields,
            output_fields=output_fields,
            needs_retrieval=needs_retrieval,
            needs_chain_of_thought=needs_cot,
            needs_tools=needs_tools,
            needs_multi_stage=needs_multi_stage,
            suggested_tools=suggested_tools,
            suggested_pipeline_template=suggested_template,
            confidence=0.8,
            reasoning=reasoning,
        )

    def _infer_task_type(self, desc_lower: str) -> TaskType:
        """Infer task type from description."""
        scores = {}

        for task_type, keywords in self.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > 0:
                scores[task_type] = score

        if scores:
            return max(scores, key=scores.get)

        return TaskType.REASONING

    def _infer_domain(self, desc_lower: str) -> str:
        """Infer domain from description."""
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                return domain
        return "general"

    def _infer_complexity(self, desc_lower: str, original: str) -> ComplexityLevel:
        """Infer complexity level."""
        if any(ind in desc_lower for ind in self.COMPLEXITY_INDICATORS["high"]):
            return ComplexityLevel.HIGH
        if any(ind in desc_lower for ind in self.COMPLEXITY_INDICATORS["low"]):
            return ComplexityLevel.LOW

        if len(original) > 300:
            return ComplexityLevel.HIGH
        if len(original) < 100:
            return ComplexityLevel.LOW

        return ComplexityLevel.MEDIUM

    def _infer_fields(
        self, task_type: TaskType, dataset_sample: list[dict] | None
    ) -> tuple:
        """Infer input and output fields."""
        if dataset_sample and len(dataset_sample) > 0:
            sample = dataset_sample[0]
            input_fields = [
                k
                for k in sample.keys()
                if k not in ["output", "label", "answer", "result"]
            ]
            output_fields = [
                k for k in sample.keys() if k in ["output", "label", "answer", "result"]
            ]

            if not input_fields:
                input_fields = ["text"]
            if not output_fields:
                output_fields = ["result"]

            return input_fields[:3], output_fields[:2]

        field_mapping = {
            TaskType.CLASSIFICATION: (["text"], ["label"]),
            TaskType.EXTRACTION: (["text"], ["entities"]),
            TaskType.SUMMARIZATION: (["text"], ["summary"]),
            TaskType.QA: (["question", "context"], ["answer"]),
            TaskType.RAG: (["query"], ["answer"]),
            TaskType.REASONING: (["problem"], ["solution", "reasoning"]),
            TaskType.GENERATION: (["prompt"], ["content"]),
            TaskType.ROUTING: (["input"], ["route", "reason"]),
            TaskType.CODE: (["task"], ["code"]),
        }

        return field_mapping.get(task_type, (["text"], ["result"]))

    def _needs_retrieval(self, desc_lower: str, task_type: TaskType) -> bool:
        """Determine if task needs retrieval."""
        if task_type == TaskType.RAG:
            return True

        retrieval_keywords = [
            "retriev",
            "search",
            "knowledge base",
            "document",
            "context",
            "rag",
        ]
        return any(kw in desc_lower for kw in retrieval_keywords)

    def _needs_chain_of_thought(
        self, task_type: TaskType, complexity: ComplexityLevel
    ) -> bool:
        """Determine if task needs chain of thought."""
        cot_tasks = {
            TaskType.REASONING,
            TaskType.QA,
            TaskType.CODE,
            TaskType.EXTRACTION,
        }

        if task_type in cot_tasks:
            return True
        if complexity == ComplexityLevel.HIGH:
            return True

        return False

    def _needs_tools(self, desc_lower: str) -> bool:
        """Determine if task needs tools."""
        for tool, indicators in self.TOOL_INDICATORS.items():
            if any(ind in desc_lower for ind in indicators):
                return True
        return False

    def _needs_multi_stage(
        self, desc_lower: str, task_type: TaskType, complexity: ComplexityLevel
    ) -> bool:
        """Determine if task needs multi-stage pipeline."""
        multi_stage_keywords = [
            "then",
            "after that",
            "followed by",
            "step 1",
            "step 2",
            "first",
            "second",
            "finally",
            "multi-step",
            "pipeline",
        ]

        if any(kw in desc_lower for kw in multi_stage_keywords):
            return True

        if task_type == TaskType.GENERATION and complexity == ComplexityLevel.HIGH:
            return True

        return False

    def _suggest_tools(self, desc_lower: str) -> list[str]:
        """Suggest tools based on task description."""
        tools = []

        for tool, indicators in self.TOOL_INDICATORS.items():
            if any(ind in desc_lower for ind in indicators):
                tools.append(tool)

        return tools

    def _suggest_template(
        self,
        task_type: TaskType,
        needs_retrieval: bool,
        needs_tools: bool,
        needs_multi_stage: bool,
    ) -> str:
        """Suggest pipeline template."""
        if needs_tools:
            return "react_agent"

        if needs_retrieval:
            return "rag_basic"

        if needs_multi_stage:
            if task_type == TaskType.GENERATION:
                return "outline_draft_revise"
            if task_type == TaskType.EXTRACTION:
                return "extract_summarize"
            if task_type == TaskType.CODE:
                return "code_generation"

        if task_type == TaskType.CLASSIFICATION:
            return "classify_explain"

        return "chain_of_thought"

    def _generate_reasoning(
        self,
        task_type: TaskType,
        domain: str,
        complexity: ComplexityLevel,
        needs_retrieval: bool,
        needs_cot: bool,
        needs_tools: bool,
    ) -> str:
        """Generate reasoning explanation."""
        parts = [
            f"Detected task type: {task_type.value}",
            f"Domain: {domain}",
            f"Complexity: {complexity.value}",
        ]

        if needs_retrieval:
            parts.append("Task requires retrieval from knowledge base")
        if needs_cot:
            parts.append("Chain of thought reasoning recommended")
        if needs_tools:
            parts.append("External tools may be helpful")

        return ". ".join(parts) + "."
