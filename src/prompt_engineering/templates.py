from typing import Any


class PromptTemplate:
    """Format a string template with named prompt variables."""

    def __init__(self, template_str: str) -> None:
        self.template_str = template_str

    def format(self, **kwargs: Any) -> str:
        try:
            return self.template_str.format(**kwargs)
        except KeyError as error:
            raise ValueError(f"Missing variable in prompt template: {error}") from error


SUMMARIZATION_TEMPLATE = PromptTemplate(
    "Summarize the following text clearly and concisely in {max_words} words:\n\n{text}"
)

ANALYSIS_TEMPLATE = PromptTemplate(
    "Analyze the following code or text from the perspective of {perspective}:\n\n{content}"
)
