import os
import logging
from langchain_core.tools import tool
from tavily import TavilyClient
from config import Tavily_api_key

logger = logging.getLogger(__name__)

if Tavily_api_key:
    client = TavilyClient(api_key=Tavily_api_key)
else:
    client = None
    logger.warning("Tavily API key not configured, web search will be unavailable")


@tool
def web_search(query: str) -> str:
    """
    Search the web for current and relevant information.
    """
    if client is None:
        logger.warning("Web search attempted but Tavily client is not configured")
        return "Web search is not available. Please configure the TAVILY_API_KEY in your .env file."
    
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