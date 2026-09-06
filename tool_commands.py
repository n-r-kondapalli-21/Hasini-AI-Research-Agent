
"""
Terminal tool-command helpers.

Provides commands for inspecting the tools registered by the research
agent. Tool categories are resolved dynamically from ToolRegistry.
"""

from __future__ import annotations

import logging
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger("hasini.tool_commands")
_default_console = Console()


def show_tools(
    tools: Iterable,
    title: str = "Available Tools",
    console: Console | None = None,
) -> None:
    """Display a list of registered tools in a Rich table."""
    if console is None:
        console = _default_console

    try:
        tool_list = list(tools)

        if not tool_list:
            console.print(f"\n[yellow]No tools found for: {title}[/yellow]")
            return

        table = Table(
            title=title,
            title_style="bold bright_green",
            border_style="green",
            header_style="bold white on dark_green",
            expand=True,
        )
        table.add_column("Tool Name", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="dim white")

        for tool in tool_list:
            tool_name = getattr(tool, "name", "<unnamed tool>")
            doc = getattr(tool, "description", None) or getattr(tool, "__doc__", "") or "No description provided."
            # Show first line or brief description
            first_line = doc.strip().split("\n")[0]
            table.add_row(tool_name, first_line)

        console.print()
        console.print(table)

    except Exception:
        logger.exception("Failed to display tools.")
        console.print("\n[bold red]❌ Failed to display tools.[/bold red]")


def show_categories(registry, console: Console | None = None) -> None:
    """Display all currently registered tool categories in a Rich table."""
    if console is None:
        console = _default_console

    try:
        categories = registry.categories()

        if not categories:
            console.print("\n[yellow]No tool categories found.[/yellow]")
            return

        table = Table(
            title="Available Tool Categories",
            title_style="bold bright_cyan",
            border_style="cyan",
            header_style="bold white on blue",
            expand=True,
        )
        table.add_column("Category", style="bold yellow", no_wrap=True)
        table.add_column("Count", style="bold green", justify="right")
        table.add_column("Inspection Command", style="bold magenta")

        for category in categories:
            cat_tools = registry.get_tools(category)
            table.add_row(
                category,
                str(len(cat_tools)),
                f"[bold cyan]/tools {category}[/bold cyan]",
            )

        console.print()
        console.print(table)

    except Exception:
        logger.exception("Failed to display tool categories.")
        console.print("\n[bold red]❌ Failed to display tool categories.[/bold red]")


def handle_tool_command(
    command: str,
    registry,
    console: Console | None = None,
) -> bool:
    """
    Handle a supported terminal tool command with Rich formatting.

    Commands:
        /tools            List available tool categories.
        /tools <category> List tools belonging to a category.
        /tool <tool_name> Inspect a specific tool.

    Returns:
        True  -> command was recognized and handled.
        False -> command is not a tool command.
    """
    if console is None:
        console = _default_console

    if not isinstance(command, str):
        logger.warning(
            "Ignoring invalid tool command type: %s",
            type(command).__name__,
        )
        return False

    command = command.strip()

    if not command:
        return False

    try:
        # --------------------------------------------------
        # /tools
        # --------------------------------------------------
        if command.lower() == "/tools":
            show_categories(registry, console=console)
            return True

        # --------------------------------------------------
        # /tools <category>
        # --------------------------------------------------
        if command.lower().startswith("/tools "):
            category = command[7:].strip()

            if not category:
                show_categories(registry, console=console)
                return True

            tools = registry.get_tools(category)

            if tools:
                show_tools(
                    tools,
                    f"{category.title()} Category Tools",
                    console=console,
                )
            else:
                console.print(
                    f"\n[yellow]No tools found for category '[bold]{category}[/bold]'.[/yellow]"
                )
                show_categories(registry, console=console)

            return True

        # --------------------------------------------------
        # /tool <tool_name>
        # --------------------------------------------------
        if command.lower().startswith("/tool "):
            tool_name = command[6:].strip()

            if not tool_name:
                console.print("\n[bold red]❌ Tool name cannot be empty.[/bold red]")
                return True

            tool = registry.get_tool(tool_name)

            if tool:
                name = getattr(tool, "name", tool_name)
                description = getattr(tool, "description", None) or getattr(tool, "__doc__", "No description provided.")
                
                content_text = Text()
                content_text.append("Name: ", style="bold cyan")
                content_text.append(f"{name}\n\n")
                content_text.append("Description:\n", style="bold yellow")
                content_text.append(f"{description}\n")

                panel = Panel(
                    content_text,
                    title=f"[bold bright_magenta]🛠️ Tool Inspection: {name}[/bold bright_magenta]",
                    border_style="magenta",
                    expand=False,
                )
                console.print()
                console.print(panel)
            else:
                console.print(
                    f"\n[bold red]❌ Tool '[yellow]{tool_name}[/yellow]' not found.[/bold red]"
                )

            return True

    except Exception:
        logger.exception(
            "Failed to handle tool command: %s",
            command,
        )
        console.print("\n[bold red]❌ Failed to process the tool command.[/bold red]")
        return True

    return False

