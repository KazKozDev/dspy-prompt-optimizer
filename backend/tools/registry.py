"""Tool Registry - Central registry for ReAct agent tools."""

from typing import Any, Optional

from .base import BaseTool


class ToolRegistry:
    """Central registry for managing ReAct agent tools.

    Provides registration, lookup, and listing of available tools.
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: dict[str, BaseTool] = {}

    def __new__(cls) -> "ToolRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls._instance = None
        cls._tools = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def register_class(self, tool_class: type[BaseTool], **kwargs) -> None:
        """Register a tool by class.

        Args:
            tool_class: Tool class to instantiate and register
            **kwargs: Arguments to pass to tool constructor
        """
        tool = tool_class(**kwargs)
        self.register(tool)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_tools(self, names: list[str]) -> list[BaseTool]:
        """Get multiple tools by name.

        Args:
            names: List of tool names

        Returns:
            List of tool instances (skips unknown names)
        """
        return [self._tools[name] for name in names if name in self._tools]

    def list_tools(self) -> list[dict[str, Any]]:
        """List all registered tools.

        Returns:
            List of tool info dicts with name and description
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema,
            }
            for tool in self._tools.values()
        ]

    def list_names(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def to_dspy_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert tools to DSPy format.

        Args:
            names: Optional list of tool names to include. If None, includes all.

        Returns:
            List of DSPy tool dicts
        """
        if names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[n] for n in names if n in self._tools]

        return [tool.to_dspy_tool() for tool in tools]

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry()
