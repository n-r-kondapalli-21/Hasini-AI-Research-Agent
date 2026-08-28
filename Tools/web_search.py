import os
import logging
from langchain_core.tools import tool
from tavily import TavilyClient
from config import Tavily_api_key

logger = logging.getLogger(__name__)

client = TavilyClient(api_key=os.environ["Tavily_api_key"])


@tool
def web_search(query: str) -> str:
    """
    Search the web for current and relevant information.
    """
    try:
        response = client.search(query=query, max_results=5)
        results = response.get("results", [])

        if not results:
            return "No relevant web search results were found."

        return "\n\n".join(
            f"{r['title']}: {r['content']}" for r in results
        )

    except Exception as e:
        logger.warning("Web search failed for query '%s': %s", query, e)
        return (
            "Web search is temporarily unavailable for this query. "
            "Please try again."
        )


web_tools = [web_search]