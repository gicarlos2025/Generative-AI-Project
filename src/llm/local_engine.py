from typing import Any

from src.llm.base import BaseLLMClient
from src.llm.utils import DEFAULT_MODEL_CONFIG, load_model_config
from src.utils.logger import logger


class LocalAIEngine(BaseLLMClient):
    """Run a Hugging Face text-to-text model on the local machine."""

    def __init__(
        self,
        model_name: str | None = None,
        config_path: str = str(DEFAULT_MODEL_CONFIG),
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = (
            config if config is not None else load_model_config(config_path).get("local", {})
        )
        self.model_name = model_name or self.config.get("model_name", "google/flan-t5-small")
        self.device = self.config.get("device", -1)
        self.pipeline: Any | None = None
        self._initialize_engine()

    @property
    def is_configured(self) -> bool:
        return self.pipeline is not None

    def _initialize_engine(self) -> None:
        """Load the configured model, reporting dependency and model errors explicitly."""
        try:
            logger.info("Loading local AI model %r...", self.model_name)
            from transformers import pipeline

            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=self.device,
            )
            logger.info("Local AI model %r is ready.", self.model_name)
        except ImportError as error:
            raise RuntimeError(
                "The local provider requires torch, transformers, and sentencepiece. "
                "Install them with: pip install torch transformers sentencepiece"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"Could not load local model {self.model_name!r}. "
                "Check model availability, network access for its first download, and memory."
            ) from error

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response locally; the model is downloaded on first use if needed."""
        if self.pipeline is None:
            raise RuntimeError("The local AI model is not initialized.")

        logger.info("Generating locally with %d prompt characters.", len(prompt))
        try:
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
            generation_config: dict[str, Any] = {
                "num_return_sequences": 1,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_config["temperature"] = temperature
            if "max_length" in kwargs:
                generation_config["max_length"] = kwargs["max_length"]
            else:
                generation_config["max_new_tokens"] = kwargs.get(
                    "max_new_tokens",
                    self.config.get("max_new_tokens", 250),
                )

            outputs = self.pipeline(prompt, **generation_config)
        except Exception as error:
            raise RuntimeError("Local model generation failed.") from error

        if not outputs or not isinstance(outputs[0].get("generated_text"), str):
            raise RuntimeError("The local model returned no generated text.")
        generated_text = outputs[0]["generated_text"].strip()
        if not generated_text:
            raise RuntimeError("The local model returned an empty response.")
        return generated_text