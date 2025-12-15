"""Built-in tools for ReAct agents."""

from .calculator import CalculatorTool
from .python_repl import PythonREPLTool
from .web_search import WebSearchTool
from .wikipedia import WikipediaTool

__all__ = [
    "CalculatorTool",
    "WebSearchTool",
    "PythonREPLTool",
    "WikipediaTool",
]


def register_all_builtin_tools():
    """Register all built-in tools with the global registry."""
    from ..registry import get_registry

    registry = get_registry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(PythonREPLTool())
    registry.register(WikipediaTool())
