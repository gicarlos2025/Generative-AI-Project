from typing import Any

from src.llm.base import BaseLLMClient
from src.prompt_engineering.templates import PromptTemplate
from src.utils.cache import SimpleCache


class PromptChain:
    """Format a prompt, reuse cached results, and call an LLM client."""

    def __init__(
        self,
        client: BaseLLMClient,
        template: PromptTemplate,
        cache: SimpleCache | None = None,
    ) -> None:
        self.client = client
        self.template = template
        self.cache = cache

    def run(self, **variables: Any) -> str:
        prompt = self.template.format(**variables)
        if self.cache is not None:
            cached_result = self.cache.get(prompt)
            if cached_result is not None:
                return cached_result

        result = self.client.generate(prompt)
        if self.cache is not None:
            self.cache.set(prompt, result)
        return result
