"""
Base Tool - Abstract base class for ReAct agent tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for ReAct agent tools.
    
    Tools are functions that agents can call to interact with external systems.
    """
    
    name: str = "base_tool"
    description: str = "Base tool class"
    
    @property
    def schema(self) -> Dict[str, Any]:
        """
        JSON schema for tool input parameters.
        
        Override this to define input parameters for the tool.
        """
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input for the tool"
                }
            },
            "required": ["input"]
        }
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with output or error
        """
        pass
    
    async def arun(self, **kwargs) -> ToolResult:
        """
        Async version of run. Default implementation calls sync version.
        
        Override for truly async tools.
        """
        return self.run(**kwargs)
    
    def __call__(self, **kwargs) -> str:
        """
        Call the tool and return output string.
        
        This is the interface expected by DSPy ReAct.
        """
        result = self.run(**kwargs)
        if result.success:
            return result.output
        return f"Error: {result.error}"
    
    def to_dspy_tool(self) -> Dict[str, Any]:
        """
        Convert to DSPy tool format.
        
        Returns dict with name, description, and function.
        """
        return {
            "name": self.name,
            "desc": self.description,
            "func": self.__call__,
            "input_type": "str",
        }
