"""
Terminal tool-command helpers.

Provides commands for inspecting the tools registered by the research
agent. Existing command behavior is preserved.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger("hasini.tool_commands")


def show_tools(
    tools: Iterable,
    title: str = "Available Tools",
) -> None:
    """Display a list of registered tools in the terminal."""
    try:
        tools = list(tools)

        print(f"\n{title}")
        print("-" * 50)

        if not tools:
            print("No tools found.")
        else:
            for tool in tools:
                tool_name = getattr(tool, "name", "<unnamed tool>")
                print(f"• {tool_name}")

        print("-" * 50)

    except Exception:
        logger.exception("Failed to display tools.")
        print("\n❌ Failed to display tools.")


def handle_tool_command(command: str, registry) -> bool:
    """
    Handle a supported terminal tool command.

    Returns:
        True  -> command was recognized and handled.
        False -> command is not a tool command.
    """

    if not isinstance(command, str):
        logger.warning(
            "Ignoring invalid tool command type: %s",
            type(command).__name__,
        )
        return False

    command = command.lower().strip()

    if not command:
        return False

    try:
        # --------------------------------------------------
        # /tools
        # --------------------------------------------------
        if command == "/tools":
            show_tools(registry.get_all_tools())
            return True

        # --------------------------------------------------
        # /tools <category>
        # --------------------------------------------------
        if command.startswith("/tools "):
            category = command[7:].strip()

            if not category:
                show_tools(registry.get_all_tools())
                return True

            tools = registry.get_tools(category)

            if tools:
                show_tools(
                    tools,
                    f"{category.title()} Tools",
                )
            else:
                print(
                    f"\nNo tools found for category "
                    f"'{category}'."
                )

                categories = registry.categories()

                if categories:
                    print("\nAvailable categories:")

                    for item in categories:
                        print(f"• {item}")

            return True

        # --------------------------------------------------
        # /tool <tool_name>
        # --------------------------------------------------
        if command.startswith("/tool "):
            tool_name = command[6:].strip()

            if not tool_name:
                print("\n❌ Tool name cannot be empty.")
                return True

            tool = registry.get_tool(tool_name)

            if tool:
                print("\nTool")
                print("-" * 50)
                print(f"Name: {getattr(tool, 'name', tool_name)}")

                description = getattr(
                    tool,
                    "description",
                    None,
                )

                if description:
                    print(f"\nDescription:\n{description}")

                print("-" * 50)
            else:
                print(
                    f"\nTool '{tool_name}' not found."
                )

            return True

    except Exception:
        logger.exception(
            "Failed to handle tool command: %s",
            command,
        )
        print(
            "\n❌ Failed to process the tool command."
        )
        return True

    return False
