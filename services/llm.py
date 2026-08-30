"""
LLM client initialization.

Enhancements over the original version:
  - Validates `provider`/`MODEL_NAME` config at import time with clear,
    specific error messages. The original would have let a missing
    `api_key` or `base_url` propagate as a bare KeyError, or let an
    empty/placeholder API key silently reach `ChatOpenAI` and fail much
    later (and more confusingly) on the first actual request.
  - `ChatOpenAI` construction is wrapped so a bad config value (e.g. a
    malformed `base_url`) raises a clear "failed to initialize LLM
    client" error instead of a raw httpx/openai traceback with no
    context about which setting caused it.
  - Adds `timeout` and `max_retries` to the client. Without these,
    `langchain_openai.ChatOpenAI` defaults still apply, but leaving them
    implicit means a flaky local/self-hosted endpoint (e.g. Ollama,
    OpenRouter under load) can hang a voice turn indefinitely with no
    retry — worth being explicit about for a voice assistant where a
    hung request is much more noticeable than in a chat UI.
  - `logging` instead of `print` in the smoke-test block, and the smoke
    test itself is wrapped so a connectivity/auth failure at startup
    prints a clear diagnostic instead of a raw traceback.
"""

import logging

from langchain_openai import ChatOpenAI

from config import provider, MODEL_NAME


logger = logging.getLogger("hasini.llm")

# Suppress HTTP request logs from httpx/httpcore.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LLMConfigError(RuntimeError):
    """Raised when the LLM provider configuration is missing or invalid."""


def _validate_config() -> None:
    if not MODEL_NAME or not str(MODEL_NAME).strip():
        raise LLMConfigError("MODEL_NAME is not set in config.")

    if not isinstance(provider, dict):
        raise LLMConfigError(
            f"config.provider must be a dict, got {type(provider).__name__}"
        )

    api_key = provider.get("api_key")
    base_url = provider.get("base_url")

    if not api_key or not str(api_key).strip():
        raise LLMConfigError(
            "provider['api_key'] is missing/empty in config. "
            "Set it (or the corresponding env var) before starting."
        )

    if not base_url or not str(base_url).strip():
        raise LLMConfigError(
            "provider['base_url'] is missing/empty in config."
        )


def _build_llm() -> ChatOpenAI:
    _validate_config()

    try:
        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            timeout=60,  # Increased timeout for better handling of overloaded services
            max_retries=3,  # Increased retries for transient failures
        )
    except Exception as exc:
        raise LLMConfigError(
            f"Failed to initialize LLM client (model={MODEL_NAME!r}, "
            f"base_url={provider.get('base_url')!r}): {exc}"
        ) from exc


llm = _build_llm()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        response = llm.invoke("Hello!")
    except Exception:
        logger.error(
            "LLM smoke test failed — check API key, base_url, and network "
            "connectivity to the provider.",
            exc_info=True,
        )
    else:
        logger.info("LLM response: %s", response.content)