from typing import Any

from src.llm.anthropic_client import AnthropicClient
from src.llm.base import BaseLLMClient
from src.llm.gemini_client import GeminiClient
from src.llm.gpt_client import GPTClient
from src.llm.local_engine import LocalAIEngine
from src.llm.utils import DEFAULT_MODEL_CONFIG, load_model_config


def create_llm_client(
    config_path: str = str(DEFAULT_MODEL_CONFIG),
    config: dict[str, Any] | None = None,
) -> BaseLLMClient:
    """Create the configured provider client."""
    model_config = config if config is not None else load_model_config(config_path)
    provider = model_config.get("default_provider", "gemini")
    clients: dict[str, type[BaseLLMClient]] = {
        "openai": GPTClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
        "local": LocalAIEngine,
    }

    try:
        client_type = clients[provider]
    except (KeyError, TypeError) as error:
        supported_providers = ", ".join(clients)
        raise ValueError(
            f"Unsupported LLM provider {provider!r}; choose one of: {supported_providers}"
        ) from error

    return client_type(config_path=config_path, config=model_config.get(provider, {}))
