from src.utils.logger import logger


class GenerativeAIException(Exception):
    """Base exception for application-level errors."""


def handle_llm_error(error: Exception) -> str:
    logger.error("Failed to process LLM request: %s", error)
    return "An error occurred while processing your request. Please try again later."
