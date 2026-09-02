"""
Hasini text-mode entrypoint.

The shared research-agent runtime lives in agent_runtime.py.
This file is responsible only for the terminal/text interface.
"""

from __future__ import annotations

import asyncio
import logging

from agent_runtime import create_research_agent, Agent_stream
from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT
from services.conversation_memory import ConversationMemory
from tool_commands import handle_tool_command
from text_confirmation import TextConfirmationManager


logger = logging.getLogger("hasini.main")


async def main() -> None:
    """Run the terminal-based text research agent."""

    logger.info("Starting AI Research Agent in text mode...")

    try:
        # Create text-based confirmation manager for permission-gated tools
        confirmation_manager = TextConfirmationManager()
        
        agent, registry = await create_research_agent(
            confirmation_manager=confirmation_manager,
            mode="text"
        )

        memory = ConversationMemory(
            history_limit=MEMORY_HISTORY_LIMIT,
            enabled=MEMORY_ENABLED,
        )

        logger.info(
            "Conversation memory initialized "
            "(enabled=%s, history_limit=%d).",
            MEMORY_ENABLED,
            MEMORY_HISTORY_LIMIT,
        )

        print("Commands:")
        print("  /tools                  List available tool categories")
        print("                          Example: /tools")
        print("  /tools <category>       List tools in a category")
        print("                          Example: /tools openalgo")
        print("  /tool <tool_name>       Inspect a specific tool")
        print("                          Example: /tool openalgo_get_quote")
        print("  exit / quit             Exit")
        print("                          Example: exit")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

            except (EOFError, KeyboardInterrupt):
                logger.info("Terminal input interrupted.")
                print("\nGoodbye!")
                break

            if user_input.lower() in {"exit", "quit"}:
                logger.info("Exit command received.")
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            try:
                if handle_tool_command(user_input, registry):
                    continue
            except Exception:
                logger.exception(
                    "Tool command failed: %s",
                    user_input,
                )
                print("\n❌ Tool command failed.")
                continue

            print("\nHasini: ", end="", flush=True)

            response_received = False

            try:
                async for chunk in Agent_stream(
                    user_input,
                    agent,
                    memory,
                ):
                    response_received = True
                    print(chunk, end="", flush=True)

            except asyncio.CancelledError:
                logger.info("Agent response cancelled.")
                print("\n")
                raise

            except Exception:
                logger.exception(
                    "Unexpected error while processing terminal query."
                )
                print("\n❌ Failed to process the request.")

            print()

            if not response_received:
                logger.debug("No response received for terminal query.")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")

    except Exception:
        logger.exception("Research agent text application failed.")
        raise

    finally:
        logger.info("AI Research Agent text mode stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
