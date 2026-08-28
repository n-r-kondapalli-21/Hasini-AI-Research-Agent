"""
Shared research-agent runtime.

This module contains the agent factory and streaming interface shared by
the text and voice entrypoints.

Entry points:
    main.py       -> text mode
    voice_main.py -> voice mode
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk

from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT
from services.conversation_memory import ConversationMemory
from services.llm import llm
from system_prompt import get_system_prompt
from tool_management.tool_registry import ToolRegistry
from voice.confirmation import ConfirmationRejected, ConfirmationTimeout


logger = logging.getLogger("hasini.agent_runtime")


async def create_research_agent(confirmation_manager=None,mode: str = "text",):
    """
    Build the research agent using the central Tool Registry.

    Args:
        confirmation_manager:
            Optional voice confirmation manager used by gated tools.
        mode:
            Agent mode, normally "text" or "voice".

    Returns:
        tuple: (agent, registry)
    """
    if mode not in {"text", "voice"}:
        raise ValueError(
            f"Unsupported agent mode: {mode!r}. "
            "Expected 'text' or 'voice'."
        )

    try:
        registry = ToolRegistry(
            confirmation_manager=confirmation_manager,
        )

        await registry.initialize()

        all_tools = registry.get_all_tools()

        logger.info(
            "Tool registry initialized with %d tool(s).",
            len(all_tools),
        )

        prompt = get_system_prompt(mode)

        agent = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt=prompt,
        )

        logger.info(
            "Research agent ready (mode=%s).",
            mode,
        )

        return agent, registry

    except Exception:
        logger.exception(
            "Failed to create research agent (mode=%s).",
            mode,
        )
        raise


async def Agent_stream(
    query: str,
    agent,
    memory: ConversationMemory,
) -> AsyncIterator[str]:
    """
    Stream assistant response content from the research agent.

    Conversation memory is shared by both text and voice interfaces.
    Confirmation rejection/timeout are treated as controlled cancellation
    and are not converted into normal tool results.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = query.strip()

    if not query:
        logger.debug("Ignoring empty agent query.")
        return

    full_response = ""

    try:
        memory.add_user_message(query)

        messages = memory.get_history()

        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": query,
                }
            ]

        logger.info(
            "Processing agent query: %s",
            query,
        )

        async for chunk in agent.astream(
            {"messages": messages},
            stream_mode="messages",
        ):
            if not chunk or len(chunk) != 2:
                logger.debug("Ignoring malformed agent stream chunk.")
                continue

            message, _metadata = chunk

            if not message:
                continue

            msg_type = getattr(message, "type", "")

            if not isinstance(
                message,
                (AIMessage, AIMessageChunk),
            ) and msg_type not in {"ai", "assistant"}:
                continue

            content = getattr(message, "content", "")

            if not content:
                continue

            if isinstance(content, list):
                text_parts = []

                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            text_parts.append(str(text))
                    elif isinstance(item, str):
                        text_parts.append(item)

                content = "".join(text_parts)

            if not isinstance(content, str) or not content:
                continue

            full_response += content
            yield content

    except ConfirmationRejected:
        logger.info(
            "Action cancelled by user for query: %s",
            query,
        )
        return

    except ConfirmationTimeout:
        logger.info(
            "Confirmation timed out for query: %s",
            query,
        )
        return

    except asyncio.CancelledError:
        logger.info(
            "Agent response cancelled for query: %s",
            query,
        )
        raise

    except Exception:
        logger.exception(
            "Agent response failed for query: %s",
            query,
        )
        return

    finally:
        if full_response:
            try:
                memory.add_assistant_message(full_response)
            except Exception:
                logger.exception(
                    "Failed to save assistant response to conversation memory."
                )
