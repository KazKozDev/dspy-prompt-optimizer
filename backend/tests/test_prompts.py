from __future__ import annotations

import pytest

from prompts.templates import get_prompt, render_prompt


def test_get_prompt_exists() -> None:
    prompt = get_prompt("system_prompts", "default")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_render_prompt_formats() -> None:
    rendered = render_prompt("analysis_prompts", "mcp_agent_analyze_task", task="hello")
    assert "hello" in rendered


def test_get_prompt_missing_raises() -> None:
    with pytest.raises(KeyError):
        get_prompt("missing_section", "missing")
