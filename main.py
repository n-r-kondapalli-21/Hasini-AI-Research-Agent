"""
Hasini text-mode entrypoint.

The shared research-agent runtime lives in agent_runtime.py.
This file is responsible only for the terminal/text interface with Rich styling.
"""

from __future__ import annotations

import asyncio
import logging
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_runtime import create_research_agent, Agent_stream
from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT
from services.conversation_memory import ConversationMemory
from tool_commands import handle_tool_command
from text_confirmation import TextConfirmationManager


logger = logging.getLogger("hasini.main")
console = Console()


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


def _print_rich_banner(startup_duration: float) -> None:
    """Display a stylized Rich startup banner with system info and instructions."""
    banner_text = Text()
    banner_text.append("🚀 AI Research Agent initialized in ", style="dim white")
    banner_text.append(f"{startup_duration:.2f}s\n", style="bold green")

    memory_status = "Enabled" if MEMORY_ENABLED else "Disabled"
    memory_style = "bold green" if MEMORY_ENABLED else "bold red"
    banner_text.append("🧠 Conversation Memory: ", style="bold white")
    banner_text.append(f"{memory_status}", style=memory_style)
    banner_text.append(f" (History Limit: {MEMORY_HISTORY_LIMIT})\n\n", style="dim white")

    banner_text.append("Available Commands:\n", style="bold yellow")
    banner_text.append("  • ", style="cyan")
    banner_text.append("/tools", style="bold cyan")
    banner_text.append("                  List available tool categories\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tools <category>", style="bold cyan")
    banner_text.append("       List tools in a specific category\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tool <tool_name>", style="bold cyan")
    banner_text.append("       Inspect detailed tool schema & docs\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("exit / quit", style="bold cyan")
    banner_text.append("             Exit the session\n", style="white")

    panel = Panel(
        banner_text,
        title="🤖 [bold bright_cyan]HASINI AI RESEARCH AGENT[/bold bright_cyan] 🤖",
        subtitle="[dim]Type your message below to begin research[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


async def main() -> None:
    """Run the terminal-based text research agent with Rich CLI UI."""

    startup_start = time.perf_counter()
    _configure_logging()

    logger.info("Starting AI Research Agent in text mode...")

    try:
        # Create text-based confirmation manager for permission-gated tools
        confirmation_manager = TextConfirmationManager(console=console)

        logger.info("Creating research agent...")

        with console.status(
            "[bold bright_cyan]Initializing Hasini Research Agent & MCP Tools...[/bold bright_cyan]",
            spinner="dots",
        ):
            try:
                agent, registry = await create_research_agent(
                    confirmation_manager=confirmation_manager,
                    mode="text"
                )
            except Exception as exc:
                logger.error("Failed to create research agent: %s", exc)
                console.print(f"\n[bold red]❌ Failed to create research agent: {exc}[/bold red]")
                raise

        logger.debug("Research agent created successfully for text interface.")

        memory = ConversationMemory(
            history_limit=MEMORY_HISTORY_LIMIT,
            enabled=MEMORY_ENABLED,
        )

        startup_duration = time.perf_counter() - startup_start
        logger.info("Text system initialized in %.2f seconds.", startup_duration)

        _print_rich_banner(startup_duration)

        while True:
            try:
                console.print("\n[bold cyan]You[/bold cyan] [dim]>[/dim] ", end="")
                user_input = input().strip()

            except (EOFError, KeyboardInterrupt):
                logger.info("Terminal input interrupted.")
                console.print("\n\n[bold yellow]👋 Goodbye![/bold yellow]")
                break

            if user_input.lower() in {"exit", "quit"}:
                logger.info("Exit command received.")
                console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break

            if not user_input:
                continue

            try:
                if handle_tool_command(user_input, registry, console=console):
                    continue
            except Exception:
                logger.exception(
                    "Tool command failed: %s",
                    user_input,
                )
                console.print("\n[bold red]❌ Tool command execution failed.[/bold red]")
                continue

            response_received = False

            try:
                with console.status(
                    "[bold bright_cyan]Hasini is thinking...[/bold bright_cyan]",
                    spinner="dots",
                ) as status:
                    async for chunk in Agent_stream(
                        user_input,
                        agent,
                        memory,
                    ):
                        if not response_received:
                            status.stop()
                            console.print("\n[bold bright_green]Hasini[/bold bright_green] [dim]>[/dim] ", end="")
                            response_received = True
                        console.print(chunk, end="")

            except asyncio.CancelledError:
                logger.info("Agent response cancelled.")
                console.print("\n")
                raise

            except Exception:
                logger.exception(
                    "Unexpected error while processing terminal query."
                )
                console.print("\n[bold red]❌ Failed to process the request.[/bold red]")

            console.print()


            if not response_received:
                logger.warning("No response received for terminal query - possible content filtering issue")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
        console.print("\n[bold yellow]Application interrupted by user.[/bold yellow]")

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

