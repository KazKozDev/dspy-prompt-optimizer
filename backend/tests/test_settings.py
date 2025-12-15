from __future__ import annotations

from utils.settings import get_settings


def test_get_settings_has_expected_sections() -> None:
    settings = get_settings()

    assert settings.model_defaults.openai_chat
    assert settings.model_defaults.openai_optimizer
    assert settings.model_defaults.semantic_model

    assert settings.model_parameters.max_tokens > 0

    assert settings.endpoints.ollama_base_url
    assert settings.agent.max_iterations > 0


def test_env_override_agent_temperature(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.42")

    settings = get_settings()
    assert abs(settings.agent.temperature - 0.42) < 1e-9

    get_settings.cache_clear()
