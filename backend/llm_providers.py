"""LLM Provider Abstraction Layer.
Supports OpenAI, Anthropic, Google Gemini, and Ollama (local).
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from utils.settings import get_settings


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models for this provider."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def get_dspy_model_string(self, model: str) -> str:
        """Get the DSPy-compatible model string."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def list_models(self) -> list[str]:
        """Fetch available models from OpenAI API."""
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("data", []):
                        model_id = model.get("id", "")
                        # GPT-5 series only: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.1
                        if "gpt-5" in model_id:
                            # Exclude non-chat variants
                            if not any(
                                x in model_id
                                for x in [
                                    "realtime",
                                    "audio",
                                    "transcribe",
                                    "tts",
                                    "search",
                                    "embedding",
                                    "moderation",
                                ]
                            ):
                                models.append(model_id)
                    return sorted(models, reverse=True)
                return []
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return []

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Generate using OpenAI API."""
        settings = get_settings()
        resolved_model = model or settings.model_defaults.openai_optimizer
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content

    def get_dspy_model_string(self, model: str) -> str:
        return f"openai/{model}"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def list_models(self) -> list[str]:
        """Fetch available models from Anthropic API."""
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    models = [
                        model.get("id")
                        for model in data.get("data", [])
                        if model.get("id")
                    ]
                    return sorted(models)
                return []
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return []

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Generate using Anthropic API."""
        settings = get_settings()
        resolved_model = model or settings.model_defaults.anthropic_chat
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        response = await client.messages.create(
            model=resolved_model,
            max_tokens=kwargs.get("max_tokens", 2048),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def get_dspy_model_string(self, model: str) -> str:
        return f"anthropic/{model}"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

    async def list_models(self) -> list[str]:
        """Fetch available models from Google Gemini API."""
        if not self.api_key:
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                )
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        name = model.get("name", "")
                        # Extract model ID from "models/gemini-1.5-pro"
                        if name.startswith("models/"):
                            model_id = name.replace("models/", "")
                            # Filter for generative models
                            if "generateContent" in model.get(
                                "supportedGenerationMethods", []
                            ):
                                models.append(model_id)
                    return sorted(models)
                return []
        except Exception as e:
            print(f"Gemini API error: {e}")
            return []

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Generate using Gemini API."""
        settings = get_settings()
        resolved_model = model or settings.model_defaults.gemini_chat
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        gen_model = genai.GenerativeModel(resolved_model)
        response = await gen_model.generate_content_async(prompt)
        return response.text

    def get_dspy_model_string(self, model: str) -> str:
        return f"google/{model}"


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", settings.endpoints.ollama_base_url
        )

    async def list_models(self) -> list[str]:
        """Fetch installed models from Ollama API."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        name = model.get("name", "")
                        if name:
                            models.append(name)
                    return sorted(models)
                return []
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return []

    async def generate(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Generate using Ollama API."""
        settings = get_settings()
        resolved_model = model or settings.model_defaults.ollama_chat
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": resolved_model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs,
                },
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            raise Exception(f"Ollama error: {response.text}")

    def get_dspy_model_string(self, model: str) -> str:
        return f"ollama_chat/{model}"

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: dict[str, BaseLLMProvider] = {}

    @classmethod
    def get_provider(cls, provider_name: str) -> BaseLLMProvider:
        """Get or create a provider instance."""
        if provider_name not in cls._providers:
            if provider_name == "openai":
                cls._providers[provider_name] = OpenAIProvider()
            elif provider_name == "anthropic":
                cls._providers[provider_name] = AnthropicProvider()
            elif provider_name == "gemini":
                cls._providers[provider_name] = GeminiProvider()
            elif provider_name == "ollama":
                cls._providers[provider_name] = OllamaProvider()
            else:
                raise ValueError(f"Unknown provider: {provider_name}")
        return cls._providers[provider_name]

    @classmethod
    async def list_all_models(cls) -> dict[str, list[str]]:
        """List models from all providers."""
        result = {}
        for provider_name in ["openai", "anthropic", "gemini", "ollama"]:
            try:
                provider = cls.get_provider(provider_name)
                models = await provider.list_models()
                result[provider_name] = models
            except Exception as e:
                print(f"Error listing {provider_name} models: {e}")
                result[provider_name] = []
        return result

    @classmethod
    def get_dspy_model_string(cls, provider_name: str, model: str) -> str:
        """Get DSPy-compatible model string."""
        provider = cls.get_provider(provider_name)
        return provider.get_dspy_model_string(model)
