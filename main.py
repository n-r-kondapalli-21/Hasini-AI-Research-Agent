"""
Hasini text-mode entrypoint.

The shared research-agent runtime lives in agent_runtime.py.
This file is responsible only for the terminal/text interface.
"""

from __future__ import annotations

import asyncio
import logging
import time

from agent_runtime import create_research_agent, Agent_stream
from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT
from services.conversation_memory import ConversationMemory
from tool_commands import handle_tool_command
from text_confirmation import TextConfirmationManager


logger = logging.getLogger("hasini.main")


def _configure_logging() -> None:
    """Configure application-wide logging before startup work begins."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Keep noisy third-party loggers from overwhelming the application logs.
    for logger_name in (
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "openai._base_client",
        "hasini.agent_runtime",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger.info("Logging initialized.")


async def main() -> None:
    """Run the terminal-based text research agent."""

    startup_start = time.perf_counter()
    _configure_logging()

    logger.info("Starting AI Research Agent in text mode...")

    try:
        # Create text-based confirmation manager for permission-gated tools
        confirmation_manager = TextConfirmationManager()

        logger.info("Creating research agent...")

        try:
            agent, registry = await create_research_agent(
                confirmation_manager=confirmation_manager,
                mode="text"
            )
        except Exception as exc:
            logger.error("Failed to create research agent: %s", exc)
            raise

        logger.debug("Research agent created successfully for text interface.")

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

        startup_duration = time.perf_counter() - startup_start
        logger.info("Text system initialized in %.2f seconds.", startup_duration)

        _print_banner()

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


def _print_banner() -> None:
    logger.info("Hasini Text Research Agent started.")
    logger.info("Type your message to start a conversation.")
    logger.info("Type 'exit' or 'quit' to end the session.")
    logger.info("Press Ctrl+C to stop.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
