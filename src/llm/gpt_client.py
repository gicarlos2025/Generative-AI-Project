import os
from typing import Any

from src.llm.base import BaseLLMClient
from src.llm.utils import DEFAULT_MODEL_CONFIG, load_model_config
from src.utils.logger import logger


class GPTClient(BaseLLMClient):
    """OpenAI client that uses a deterministic mock when no API key is set."""

    def __init__(
        self,
        config_path: str = str(DEFAULT_MODEL_CONFIG),
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config if config is not None else load_model_config(config_path).get("openai", {})
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = self.config.get("model_name", "gpt-4o-mini")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.info("Sending prompt to %s", self.model_name)

        if not self.api_key:
            logger.info("OPENAI_API_KEY is not set; returning a mock response.")
            return f"[Mock GPT response - {self.model_name}]: Response for '{prompt[:30]}...'"

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.config.get("temperature", 0.7)),
            max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 500)),
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned an empty response.")
        return content
