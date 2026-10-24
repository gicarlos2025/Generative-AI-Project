import os
from typing import Any

from src.llm.base import BaseLLMClient
from src.llm.utils import DEFAULT_MODEL_CONFIG, load_model_config
from src.utils.logger import logger


class GeminiClient(BaseLLMClient):
    """Google Gemini client with local-mode reporting when no API key is set."""

    def __init__(
        self,
        config_path: str = str(DEFAULT_MODEL_CONFIG),
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = (
            config if config is not None else load_model_config(config_path).get("gemini", {})
        )
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = self.config.get("model_name", "gemini-2.5-flash")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.api_key:
            logger.info("GEMINI_API_KEY is not set; RAG will return retrieved passages.")
            return f"[Mock Gemini response - {self.model_name}]: Response for '{prompt[:30]}...'"

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.config.get("temperature", 0.7)),
                max_output_tokens=kwargs.get(
                    "max_output_tokens",
                    self.config.get("max_output_tokens", 500),
                ),
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text
