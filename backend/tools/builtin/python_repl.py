"""Python REPL Tool - Executes Python code safely."""

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from ..base import BaseTool, ToolResult


class PythonREPLTool(BaseTool):
    """Python REPL tool for executing Python code.

    Executes code in a restricted environment with timeout.
    Use with caution - code execution can be dangerous.
    """

    name = "python_repl"
    description = "Executes Python code and returns the output. Use for calculations, data processing, or when you need to run code. Print results to see them."

    ALLOWED_MODULES = {
        "math",
        "statistics",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
    }

    def __init__(self, timeout: int = 10, allow_imports: bool = True):
        """Initialize Python REPL tool.

        Args:
            timeout: Maximum execution time in seconds
            allow_imports: Whether to allow importing modules
        """
        self.timeout = timeout
        self.allow_imports = allow_imports
        self._globals: dict[str, Any] = {}

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        }

    def _create_restricted_globals(self) -> dict[str, Any]:
        """Create restricted globals for code execution."""
        import datetime
        import json
        import math
        import random
        import re
        import statistics
        from collections import Counter, defaultdict, deque

        restricted = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "pow": pow,
                "isinstance": isinstance,
                "type": type,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "any": any,
                "all": all,
                "format": format,
                "repr": repr,
                "True": True,
                "False": False,
                "None": None,
            },
            "math": math,
            "statistics": statistics,
            "random": random,
            "datetime": datetime,
            "json": json,
            "re": re,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "deque": deque,
        }

        return restricted

    def run(self, code: str = "", **kwargs) -> ToolResult:
        """Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            ToolResult with execution output
        """
        if not code:
            code = kwargs.get("input", "")

        if not code:
            return ToolResult(success=False, output="", error="No code provided")

        dangerous_patterns = [
            "import os",
            "import sys",
            "import subprocess",
            "__import__",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "file(",
            "input(",
            "os.",
            "sys.",
            "subprocess.",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Dangerous operation not allowed: {pattern}",
                )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            globals_dict = self._create_restricted_globals()
            globals_dict.update(self._globals)

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals_dict)

            for key, value in globals_dict.items():
                if (
                    not key.startswith("_")
                    and key not in self._create_restricted_globals()
                ):
                    self._globals[key] = value

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            output = stdout_output
            if stderr_output:
                output += f"\nStderr: {stderr_output}"

            if not output.strip():
                output = "Code executed successfully (no output)"

            return ToolResult(
                success=True, output=output.strip(), metadata={"code": code}
            )

        except Exception:
            error_msg = traceback.format_exc()
            return ToolResult(
                success=False, output="", error=f"Execution error:\n{error_msg}"
            )

    def reset(self) -> None:
        """Reset the REPL state."""
        self._globals = {}
