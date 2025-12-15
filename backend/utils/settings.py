from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]  # backend/utils/settings.py -> repo root


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file content."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


@dataclass(frozen=True)
class RateLimitSettings:
    """Rate limit settings."""

    requests_per_minute: int
    tokens_per_minute: int


@dataclass(frozen=True)
class CacheSettings:
    """Cache settings."""

    enabled: bool
    ttl_seconds: int
    cache_dir: Path


@dataclass(frozen=True)
class ModelDefaults:
    """Default model names by provider/use-case."""

    openai_chat: str
    openai_optimizer: str
    anthropic_chat: str
    gemini_chat: str
    ollama_chat: str
    semantic_model: str


@dataclass(frozen=True)
class ModelParameters:
    """Default model parameters."""

    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class LoggingSettings:
    """Logging settings."""

    config_path: Path


@dataclass(frozen=True)
class EndpointSettings:
    """External API endpoint settings."""

    ollama_base_url: str


@dataclass(frozen=True)
class AgentSettings:
    """Agent behavior settings."""

    temperature: float
    max_iterations: int


@dataclass(frozen=True)
class JudgeSettings:
    """LLM judge settings."""

    temperature: float


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from YAML + environment variables."""

    model_defaults: ModelDefaults
    model_parameters: ModelParameters
    agent: AgentSettings
    judge: JudgeSettings
    endpoints: EndpointSettings
    rate_limits: RateLimitSettings
    cache: CacheSettings
    logging: LoggingSettings


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from config/model_config.yaml with env overrides."""
    cfg = _load_yaml(project_root() / "config" / "model_config.yaml")

    models: dict[str, Any] = cfg.get("models", {})
    defaults: dict[str, Any] = models.get("defaults", {})
    params: dict[str, Any] = models.get("parameters", {})

    model_defaults = ModelDefaults(
        openai_chat=os.getenv(
            "OPENAI_CHAT_MODEL", defaults.get("openai_chat", "gpt-5-mini")
        ),
        openai_optimizer=os.getenv(
            "OPENAI_OPTIMIZER_MODEL",
            defaults.get("openai_optimizer", "gpt-5-2-instant-20251211"),
        ),
        anthropic_chat=os.getenv(
            "ANTHROPIC_CHAT_MODEL",
            defaults.get("anthropic_chat", "claude-sonnet-4-5-20250929"),
        ),
        gemini_chat=os.getenv(
            "GEMINI_CHAT_MODEL", defaults.get("gemini_chat", "gemini-2-5-flash")
        ),
        ollama_chat=os.getenv(
            "OLLAMA_CHAT_MODEL", defaults.get("ollama_chat", "llama3.2:3b")
        ),
        semantic_model=os.getenv(
            "SEMANTIC_MODEL", defaults.get("semantic_model", "all-MiniLM-L6-v2")
        ),
    )

    model_parameters = ModelParameters(
        temperature=_env_float(
            "LLM_TEMPERATURE", float(params.get("temperature", 0.2))
        ),
        max_tokens=_env_int("LLM_MAX_TOKENS", int(params.get("max_tokens", 2048))),
    )

    agent_cfg: dict[str, Any] = cfg.get("agent", {})
    agent_settings = AgentSettings(
        temperature=_env_float(
            "AGENT_TEMPERATURE",
            float(agent_cfg.get("temperature", model_parameters.temperature)),
        ),
        max_iterations=_env_int(
            "AGENT_MAX_ITERATIONS", int(agent_cfg.get("max_iterations", 20))
        ),
    )

    judge_cfg: dict[str, Any] = cfg.get("judge", {})
    judge_settings = JudgeSettings(
        temperature=_env_float(
            "JUDGE_TEMPERATURE", float(judge_cfg.get("temperature", 0.0))
        ),
    )

    endpoints_cfg: dict[str, Any] = cfg.get("endpoints", {})
    endpoints = EndpointSettings(
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            str(endpoints_cfg.get("ollama_base_url", "http://localhost:11434")),
        ),
    )

    rate: dict[str, Any] = cfg.get("rate_limits", {})
    rate_limits = RateLimitSettings(
        requests_per_minute=_env_int(
            "REQUESTS_PER_MINUTE", int(rate.get("requests_per_minute", 50))
        ),
        tokens_per_minute=_env_int(
            "TOKENS_PER_MINUTE", int(rate.get("tokens_per_minute", 100000))
        ),
    )

    cache_cfg: dict[str, Any] = cfg.get("cache", {})
    cache_dir = Path(str(cache_cfg.get("cache_dir", "data/cache")))
    cache_settings = CacheSettings(
        enabled=_env_bool("CACHE_ENABLED", bool(cache_cfg.get("enabled", True))),
        ttl_seconds=_env_int(
            "CACHE_TTL_SECONDS", int(cache_cfg.get("ttl_seconds", 3600))
        ),
        cache_dir=project_root() / cache_dir,
    )

    logging_cfg: dict[str, Any] = cfg.get("logging", {})
    config_path = Path(
        str(logging_cfg.get("config_path", "config/logging_config.yaml"))
    )
    logging_settings = LoggingSettings(config_path=project_root() / config_path)

    return Settings(
        model_defaults=model_defaults,
        model_parameters=model_parameters,
        agent=agent_settings,
        judge=judge_settings,
        endpoints=endpoints,
        rate_limits=rate_limits,
        cache=cache_settings,
        logging=logging_settings,
    )


def get_prompt_config_path() -> Path:
    """Return path to prompts YAML."""
    return project_root() / "config" / "prompts.yaml"


def get_env_path() -> Path | None:
    """Return path to .env if it exists at repo root."""
    path = project_root() / ".env"
    return path if path.exists() else None
