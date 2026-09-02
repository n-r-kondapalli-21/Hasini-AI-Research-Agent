
"""
LLM client initialization.
"""

import logging

from langchain_openai import ChatOpenAI

from config import provider, MODEL_NAME


logger = logging.getLogger("hasini.llm")


def _build_llm() -> ChatOpenAI:
    try:
        if not MODEL_NAME:
            raise ValueError("MODEL_NAME is not configured.")

        if not provider.get("api_key"):
            raise ValueError("API key is not configured.")

        if not provider.get("base_url"):
            raise ValueError("Base URL is not configured.")

        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            timeout=60,
            max_retries=2,
        )

    except Exception as exc:
        logger.error("Failed to initialize LLM client: %s", exc)
        raise


llm = _build_llm()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        response = llm.invoke("Hello!")
        logger.info("LLM connection successful.")
        logger.info("Response: %s", response.content)

    except Exception as exc:
        logger.error("LLM test failed: %s", exc)
