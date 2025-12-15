"""Tests for tools module."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import ToolResult, ToolRegistry
from tools.builtin.calculator import CalculatorTool


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(success=True, output="42")
        assert result.success is True
        assert result.output == "42"
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(success=False, output="", error="Division by zero")
        assert result.success is False
        assert result.error == "Division by zero"


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    def test_simple_addition(self):
        """Test simple addition."""
        calc = CalculatorTool()
        result = calc.run(expression="2 + 2")

        assert result.success is True
        assert result.output == "4"

    def test_multiplication(self):
        """Test multiplication."""
        calc = CalculatorTool()
        result = calc.run(expression="6 * 7")

        assert result.success is True
        assert result.output == "42"

    def test_division(self):
        """Test division."""
        calc = CalculatorTool()
        result = calc.run(expression="100 / 4")

        assert result.success is True
        assert result.output == "25"

    def test_power(self):
        """Test power operation."""
        calc = CalculatorTool()
        result = calc.run(expression="2 ** 10")

        assert result.success is True
        assert result.output == "1024"

    def test_sqrt(self):
        """Test square root."""
        calc = CalculatorTool()
        result = calc.run(expression="sqrt(16)")

        assert result.success is True
        assert result.output == "4"

    def test_complex_expression(self):
        """Test complex expression."""
        calc = CalculatorTool()
        result = calc.run(expression="(10 + 5) * 2 - 3")

        assert result.success is True
        assert result.output == "27"

    def test_division_by_zero(self):
        """Test division by zero error."""
        calc = CalculatorTool()
        result = calc.run(expression="1 / 0")

        assert result.success is False
        assert "zero" in result.error.lower()

    def test_invalid_expression(self):
        """Test invalid expression."""
        calc = CalculatorTool()
        result = calc.run(expression="invalid")

        assert result.success is False

    def test_empty_expression(self):
        """Test empty expression."""
        calc = CalculatorTool()
        result = calc.run(expression="")

        assert result.success is False

    def test_callable_interface(self):
        """Test __call__ interface."""
        calc = CalculatorTool()
        output = calc(expression="3 + 3")

        assert output == "6"

    def test_pi_constant(self):
        """Test pi constant."""
        calc = CalculatorTool()
        result = calc.run(expression="pi")

        assert result.success is True
        assert float(result.output) == pytest.approx(3.14159, rel=0.001)


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        ToolRegistry.reset()

    def test_singleton(self):
        """Test singleton pattern."""
        reg1 = ToolRegistry()
        reg2 = ToolRegistry()
        assert reg1 is reg2

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        calc = CalculatorTool()

        registry.register(calc)

        assert "calculator" in registry
        assert len(registry) == 1

    def test_get_tool(self):
        """Test getting a tool."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        retrieved = registry.get("calculator")

        assert retrieved is calc

    def test_get_nonexistent_tool(self):
        """Test getting nonexistent tool."""
        registry = ToolRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_list_tools(self):
        """Test listing tools."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        tools = registry.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert "description" in tools[0]

    def test_list_names(self):
        """Test listing tool names."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        names = registry.list_names()

        assert names == ["calculator"]

    def test_get_multiple_tools(self):
        """Test getting multiple tools."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        tools = registry.get_tools(["calculator", "nonexistent"])

        assert len(tools) == 1
        assert tools[0] is calc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
