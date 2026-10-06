"""
Knowledge search tool for Hasini AI Research Agent.

Exposes the Hybrid RAG Knowledge Base retriever as a tool callable by the agent LLM.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain_core.tools import tool
from config import RAG_ENABLED

logger = logging.getLogger(__name__)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the user's private indexed Knowledge Base documents for relevant information.

    When to use:
    - Whenever the user explicitly asks to search or check the Knowledge Base, indexed documents, or personal files.
    - Whenever you cannot answer the user query with your own general knowledge and need to check for relevant information in the Knowledge Base.
    - Whenever the query depends on private knowledge, personal notes, documents, or custom data stored in the user's Knowledge Base.

    Do NOT use for:
    - Simple greetings, small talk, or general casual conversation.
    - Basic mathematics or simple calculations.
    - Live external web information, breaking news, or web searches (use web_search for live web info).
    """
    if not RAG_ENABLED:
        logger.warning("Knowledge base search attempted, but RAG_ENABLED is false.")
        return "Knowledge base search is currently disabled in configuration."

    if not query or not query.strip():
        return "Please provide a valid non-empty search query."

    query_clean = query.strip()
    start_time = time.perf_counter()

    try:
        from agent_runtime import get_rag_retriever, initialize_rag_system

        retriever = get_rag_retriever()
        if retriever is None:
            # Attempt lazy initialization if not already loaded
            retriever = initialize_rag_system()

        if retriever is None:
            logger.warning("RAG retriever is unavailable for query '%s'.", query_clean)
            return "Knowledge base system is currently unavailable."

        chunks = retriever.retrieve(query=query_clean)
        duration = time.perf_counter() - start_time

        if not chunks:
            logger.info(
                "Knowledge base search query: '%s' | Results: 0 | Duration: %.3fs",
                query_clean,
                duration,
            )
            return "No relevant documents found."

        logger.info(
            "Knowledge base search query: '%s' | Results: %d | Duration: %.3fs",
            query_clean,
            len(chunks),
            duration,
        )

        formatted_parts = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            source = meta.get("source", "Unknown")
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            text = chunk.get("text", "").strip()

            item_str = (
                f"[{idx}] Source: {filename} (ID: {chunk_id})\n"
                f"Path/URL: {source}\n"
                f"Content:\n{text}"
            )
            formatted_parts.append(item_str)

        return "\n\n" + ("\n" + "-" * 40 + "\n").join(formatted_parts)

    except Exception as exc:
        duration = time.perf_counter() - start_time
        logger.error(
            "Knowledge base search failed for query '%s' after %.3fs: %s",
            query_clean,
            duration,
            exc,
            exc_info=True,
        )
        return f"Error searching knowledge base: {exc}"


knowledge_tools = [search_knowledge_base]
