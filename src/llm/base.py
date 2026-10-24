from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Common interface implemented by LLM provider clients."""

    @property
    def is_configured(self) -> bool:
        """Whether this client can generate responses with a real model."""
        return True

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response for a prompt."""
