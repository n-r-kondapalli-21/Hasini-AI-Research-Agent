import logging

from langchain_openai import ChatOpenAI
from config import provider, MODEL_NAME


# Suppress HTTP request logs from httpx/httpcore.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=provider["api_key"],
    base_url=provider["base_url"],
)


if __name__ == "__main__":

    response = llm.invoke("Hello!")

    print(response.content)