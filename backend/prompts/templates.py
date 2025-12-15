from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from utils.settings import get_prompt_config_path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"prompts.yaml must be a mapping: {path}")
    return data


@lru_cache(maxsize=1)
def get_prompts() -> dict[str, Any]:
    """Load prompts from config/prompts.yaml."""
    return _load_yaml(get_prompt_config_path())


def get_prompt(section: str, name: str) -> str:
    """Get a prompt template by section and name."""
    data = get_prompts()
    sec = data.get(section, {})
    if not isinstance(sec, dict):
        raise KeyError(f"Prompt section not found: {section}")
    template = sec.get(name)
    if not isinstance(template, str):
        raise KeyError(f"Prompt not found: {section}.{name}")
    return template


def render_prompt(section: str, name: str, **kwargs: object) -> str:
    """Render a prompt using str.format."""
    return get_prompt(section, name).format(**kwargs)
