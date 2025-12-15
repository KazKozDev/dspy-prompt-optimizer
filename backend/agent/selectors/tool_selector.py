"""Tool Selector - Selects appropriate tools for ReAct agents."""

from ..config import TaskAnalysis, TaskType


class ToolSelector:
    """Selects tools for ReAct agents based on task analysis."""

    TASK_TOOL_MAP = {
        TaskType.REASONING: ["calculator", "python_repl"],
        TaskType.QA: ["wikipedia", "web_search"],
        TaskType.RAG: ["web_search"],
        TaskType.CODE: ["python_repl"],
    }

    KEYWORD_TOOL_MAP = {
        "calculator": [
            "calculate",
            "compute",
            "math",
            "arithmetic",
            "formula",
            "equation",
            "number",
            "sum",
            "average",
            "percentage",
        ],
        "web_search": [
            "search",
            "find",
            "look up",
            "current",
            "recent",
            "latest",
            "news",
            "today",
            "2024",
            "2025",
        ],
        "python_repl": [
            "code",
            "execute",
            "run",
            "script",
            "program",
            "data",
            "process",
            "analyze data",
            "pandas",
            "numpy",
        ],
        "wikipedia": [
            "wikipedia",
            "encyclopedia",
            "who is",
            "what is",
            "history",
            "biography",
            "definition",
            "explain",
        ],
    }

    def select(
        self, task_analysis: TaskAnalysis, available_tools: list[str] | None = None
    ) -> list[str]:
        """Select tools based on task analysis.

        Args:
            task_analysis: Result of task analysis
            available_tools: List of available tool names

        Returns:
            List of selected tool names
        """
        if not task_analysis.needs_tools:
            return []

        selected = set()

        if task_analysis.suggested_tools:
            selected.update(task_analysis.suggested_tools)

        task_tools = self.TASK_TOOL_MAP.get(task_analysis.task_type, [])
        selected.update(task_tools)

        selected.update(self._select_from_keywords(task_analysis.reasoning))

        if available_tools:
            selected = selected.intersection(set(available_tools))

        return list(selected)

    def _select_from_keywords(self, text: str) -> list[str]:
        """Select tools based on keyword matching."""
        text_lower = text.lower()
        tools = []

        for tool, keywords in self.KEYWORD_TOOL_MAP.items():
            if any(kw in text_lower for kw in keywords):
                tools.append(tool)

        return tools

    def get_tool_descriptions(self, tool_names: list[str]) -> list[dict]:
        """Get descriptions for selected tools."""
        descriptions = {
            "calculator": {
                "name": "calculator",
                "description": "Performs mathematical calculations",
                "use_when": "Need to compute numbers, formulas, or equations",
            },
            "web_search": {
                "name": "web_search",
                "description": "Searches the web for current information",
                "use_when": "Need recent/current information not in training data",
            },
            "python_repl": {
                "name": "python_repl",
                "description": "Executes Python code",
                "use_when": "Need to run code, process data, or complex calculations",
            },
            "wikipedia": {
                "name": "wikipedia",
                "description": "Searches Wikipedia for factual information",
                "use_when": "Need factual/encyclopedic information about topics",
            },
        }

        return [descriptions[name] for name in tool_names if name in descriptions]
