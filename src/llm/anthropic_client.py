import os
from typing import Any

from src.llm.base import BaseLLMClient
from src.llm.utils import DEFAULT_MODEL_CONFIG, load_model_config
from src.utils.logger import logger


class AnthropicClient(BaseLLMClient):
    """Anthropic client that uses a deterministic mock without an API key."""

    def __init__(
        self,
        config_path: str = str(DEFAULT_MODEL_CONFIG),
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config if config is not None else load_model_config(config_path).get("anthropic", {})
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = self.config.get("model_name", "claude-3-5-sonnet-20241022")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.info("Sending prompt to %s", self.model_name)

        if not self.api_key:
            logger.info("ANTHROPIC_API_KEY is not set; returning a mock response.")
            return f"[Mock Anthropic response - {self.model_name}]: Response for '{prompt[:30]}...'"

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 1000)),
            temperature=kwargs.get("temperature", self.config.get("temperature", 0.5)),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise RuntimeError("Anthropic returned an empty response.")
        return text
