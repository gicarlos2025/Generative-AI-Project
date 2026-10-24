import os
import time
import logging
import logging.config
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# ==========================================
# 1. CONFIGURATIONS & LOGGING
# ==========================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "app": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")

# Model configuration schema simulation
MODEL_CONFIG = {
    "default_provider": "openai",
    "openai": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 500
    },
    "cache": {
        "enabled": True,
        "ttl_seconds": 3600
    }
}


# ==========================================
# 2. UTILS & CACHE (src/utils/)
# ==========================================

class SimpleCache:
    """In-memory cache with TTL support."""
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            item = self._cache[key]
            if time.time() < item["expires_at"]:
                return item["value"]
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl if ttl is not None else self.default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }


# ==========================================
# 3. LLM CLIENT LAYER (src/llm/)
# ==========================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class GPTClient(BaseLLMClient):
    """OpenAI GPT client implementation."""
    def __init__(self, config: Optional[dict] = None):
        self.config = config or MODEL_CONFIG["openai"]
        self.api_key = os.getenv("OPENAI_API_KEY", "mock-key")
        self.model_name = self.config.get("model_name", "gpt-4o-mini")

    def generate(self, prompt: str, **kwargs) -> str:
        logger.info(f"Generating output using model: {self.model_name}")

        # Fallback mock mode if no API key is set
        if self.api_key == "mock-key":
            logger.warning("No OPENAI_API_KEY set. Operating in mock mode.")
            return f"[MOCK RESPONSE - {self.model_name}]: Processed prompt: '{prompt[:40]}...'"

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.config.get("temperature", 0.7)),
                max_tokens=kwargs.get("max_tokens", self.config.get("max_tokens", 500))
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Provider Error: {e}")
            raise e


# ==========================================
# 4. PROMPT ENGINEERING (src/prompt_engineering/)
# ==========================================

class PromptTemplate:
    """Prompt template engine."""
    def __init__(self, template_str: str):
        self.template_str = template_str

    def format(self, **kwargs) -> str:
        try:
            return self.template_str.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable in prompt template: {e}")

SUMMARIZATION_TEMPLATE = PromptTemplate(
    "Summarize the following text in under {max_words} words:\n\n{text}"
)


# ==========================================
# 5. ERROR HANDLERS (src/handlers/)
# ==========================================

def handle_llm_error(error: Exception) -> str:
    logger.error(f"[ErrorHandler]: Handled exception -> {error}")
    return "An error occurred while communicating with the AI service. Please try again later."


# ==========================================
# 6. APPLICATION PIPELINE & MAIN EXECUTION
# ==========================================

def run_pipeline(text: str, max_words: int = 15):
    cache = SimpleCache(default_ttl=3600)
    client = GPTClient()

    # Step 1: Format Prompt
    prompt = SUMMARIZATION_TEMPLATE.format(max_words=max_words, text=text)

    # Step 2: Cache Check
    cached_output = cache.get(prompt)
    if cached_output:
        logger.info("Cache hit! Returning cached result.")
        return cached_output

    # Step 3: LLM Call
    try:
        output = client.generate(prompt)
        cache.set(prompt, output)
        return output
    except Exception as e:
        return handle_llm_error(e)


if __name__ == "__main__":
    logger.info("Starting Generative AI MVP Application...")
    sample_text = (
        "Generative Artificial Intelligence refers to models or algorithms that create new content, "
        "including audio, code, images, text, simulations, and videos based on learned patterns."
    )
    result = run_pipeline(sample_text, max_words=20)
    print("\n--- OUTPUT ---")
    print(result)
    print("--------------\n")