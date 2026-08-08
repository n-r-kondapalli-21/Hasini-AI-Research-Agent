from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current and relevant information."""
    return search.run(query)


tools = [web_search]