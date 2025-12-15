"""
Calculator Tool - Performs mathematical calculations.
"""

import math
import re
from typing import Any, Dict

from ..base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """
    Calculator tool for mathematical expressions.
    
    Safely evaluates mathematical expressions using Python's eval
    with a restricted set of allowed functions.
    """
    
    name = "calculator"
    description = "Performs mathematical calculations. Input should be a mathematical expression like '2 + 2' or 'sqrt(16) * 3'."
    
    ALLOWED_NAMES = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "pi": math.pi,
        "e": math.e,
    }
    
    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        }
    
    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize expression for safe evaluation."""
        expression = expression.replace("^", "**")
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        return expression
    
    def run(self, expression: str = "", **kwargs) -> ToolResult:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            ToolResult with the calculated value
        """
        if not expression:
            expression = kwargs.get("input", "")
        
        if not expression:
            return ToolResult(
                success=False,
                output="",
                error="No expression provided"
            )
        
        try:
            sanitized = self._sanitize_expression(expression)
            
            result = eval(sanitized, {"__builtins__": {}}, self.ALLOWED_NAMES)
            
            if isinstance(result, float):
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 10)
            
            return ToolResult(
                success=True,
                output=str(result),
                metadata={"expression": expression, "sanitized": sanitized}
            )
            
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                output="",
                error="Division by zero"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Calculation error: {str(e)}"
            )
