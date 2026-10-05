"""
Hasini text-mode entrypoint.

The shared research-agent runtime lives in agent_runtime.py.
This file is responsible only for the terminal/text interface with Rich styling.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings

# Suppress all warnings before any imports
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_runtime import create_research_agent, Agent_stream, initialize_rag_system
from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT
from services.conversation_memory import ConversationMemory
from tool_commands import handle_tool_command
from text_confirmation import TextConfirmationManager
from logging_config import configure_logging, get_logger


logger = get_logger("hasini.main")
console = Console()


def _print_rich_banner(startup_duration: float) -> None:
    """Display a stylized Rich startup banner with system info and instructions."""
    banner_text = Text()
    banner_text.append("🚀 AI Research Agent initialized in ", style="dim white")
    banner_text.append(f"{startup_duration:.2f}s\n", style="bold green")

    memory_status = "Enabled" if MEMORY_ENABLED else "Disabled"
    memory_style = "bold green" if MEMORY_ENABLED else "bold red"
    banner_text.append("🧠 Conversation Memory: ", style="bold white")
    banner_text.append(f"{memory_status}", style=memory_style)
    banner_text.append(f" (History Limit: {MEMORY_HISTORY_LIMIT})\n", style="dim white")

    from config import RAG_ENABLED, RAG_SIMILARITY_THRESHOLD, RAG_BM25_ENABLED, RAG_RERANKER_ENABLED
    rag_status = "Enabled" if RAG_ENABLED else "Disabled"
    rag_style = "bold green" if RAG_ENABLED else "bold red"
    banner_text.append("📚 RAG Knowledge Base: ", style="bold white")
    banner_text.append(f"{rag_status}", style=rag_style)

    if RAG_ENABLED:
        features = []
        if RAG_BM25_ENABLED:
            features.append("BM25")
        if RAG_RERANKER_ENABLED:
            features.append("Reranker")
        if features:
            banner_text.append(f" ({', '.join(features)})", style="dim white")

    banner_text.append(f" (Threshold: {RAG_SIMILARITY_THRESHOLD})\n\n", style="dim white")

    banner_text.append("Available Commands:\n", style="bold yellow")
    banner_text.append("  • ", style="cyan")
    banner_text.append("/rag <list|add|remove|stats>", style="bold cyan")
    banner_text.append("  Manage RAG knowledge base\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tools", style="bold cyan")
    banner_text.append("                       List available tool categories\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tools <category>", style="bold cyan")
    banner_text.append("            List tools in a specific category\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tool <tool_name>", style="bold cyan")
    banner_text.append("            Inspect detailed tool schema & docs\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("exit / quit / /exit", style="bold cyan")
    banner_text.append("              Exit the session\n", style="white")

    panel = Panel(
        banner_text,
        title="🤖 [bold bright_cyan]HASINI AI RESEARCH AGENT[/bold bright_cyan] 🤖",
        subtitle="[dim]Type your message below to begin research[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def _print_rich_banner(startup_duration: float) -> None:
    """Display a stylized Rich startup banner with system info and instructions."""
    banner_text = Text()
    banner_text.append("🚀 AI Research Agent initialized in ", style="dim white")
    banner_text.append(f"{startup_duration:.2f}s\n", style="bold green")

    memory_status = "Enabled" if MEMORY_ENABLED else "Disabled"
    memory_style = "bold green" if MEMORY_ENABLED else "bold red"
    banner_text.append("🧠 Conversation Memory: ", style="bold white")
    banner_text.append(f"{memory_status}", style=memory_style)
    banner_text.append(f" (History Limit: {MEMORY_HISTORY_LIMIT})\n", style="dim white")

    from config import RAG_ENABLED, RAG_SIMILARITY_THRESHOLD
    rag_status = "Enabled" if RAG_ENABLED else "Disabled"
    rag_style = "bold green" if RAG_ENABLED else "bold red"
    banner_text.append("📚 RAG Knowledge Base: ", style="bold white")
    banner_text.append(f"{rag_status}", style=rag_style)
    banner_text.append(f" (Threshold: {RAG_SIMILARITY_THRESHOLD})\n\n", style="dim white")

    banner_text.append("Available Commands:\n", style="bold yellow")
    banner_text.append("  • ", style="cyan")
    banner_text.append("/rag <list|add|remove|stats>", style="bold cyan")
    banner_text.append("  Manage RAG knowledge base\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tools", style="bold cyan")
    banner_text.append("                       List available tool categories\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tools <category>", style="bold cyan")
    banner_text.append("            List tools in a specific category\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("/tool <tool_name>", style="bold cyan")
    banner_text.append("            Inspect detailed tool schema & docs\n", style="white")

    banner_text.append("  • ", style="cyan")
    banner_text.append("exit / quit", style="bold cyan")
    banner_text.append("                  Exit the session\n", style="white")

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
    configure_logging(console_level=logging.WARNING, file_level=logging.DEBUG)

    logger.info("Starting AI Research Agent in text mode...")

    try:
        # Initialize RAG system once at startup
        with console.status(
            "[bold bright_cyan]Initializing RAG Knowledge System...[/bold bright_cyan]",
            spinner="dots",
        ):
            initialize_rag_system()

        # Create text-based confirmation manager for permission-gated tools
        confirmation_manager = TextConfirmationManager(console=console)
        # _active_status is updated each iteration so the confirmation manager
        # can stop the spinner before calling input() — otherwise Rich's Live
        # display swallows stdin and the user's y/n keystrokes go nowhere.
        confirmation_manager._active_status = None

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

            # Handle exit commands
            if user_input.lower() in {"exit", "quit", "/exit"}:
                logger.info("Exit command received: %s", user_input)
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
                status = console.status(
                    "[bold bright_cyan]Hasini is thinking...[/bold bright_cyan]",
                    spinner="dots",
                )
                status.start()
                # Give the confirmation manager a reference to the active
                # spinner so it can stop it before calling input().
                # If it doesn't stop first, Rich's Live display holds the
                # terminal in a mode that swallows stdin entirely.
                confirmation_manager._active_status = status
                try:
                    async for chunk in Agent_stream(
                        user_input,
                        agent,
                        memory,
                    ):
                        if not response_received:
                            status.stop()
                            confirmation_manager._active_status = None
                            console.print("\n[bold bright_green]Hasini[/bold bright_green] [dim]>[/dim] ", end="")
                            response_received = True
                        console.print(chunk, end="")
                finally:
                    status.stop()
                    confirmation_manager._active_status = None

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

