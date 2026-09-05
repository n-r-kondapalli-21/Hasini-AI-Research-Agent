
"""
Application tool registry.

Responsible only for:
- Registering built-in application tools.
- Grouping built-in tools by category.
- Providing access to built-in tools.

MCP tools are managed separately by MCPRegistry.
"""

from __future__ import annotations

import logging

from config import (
    ENABLE_WEB_TOOL,
    ENABLE_WEATHER_TOOL,
    ENABLE_CALCULATOR_TOOL,
)
from Tools.web_search import web_tools
from Tools.weather import weather_tools
from Tools.calculator import calculator_tool


logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for built-in application tools.

    MCP tools are intentionally not managed here.
    """

    def __init__(self):

        self._tools = {}
        self._categories = {}

    async def initialize(self):
        """
        Load and register built-in tools.
        """

        # --------------------------------------------------
        # Web tools
        # --------------------------------------------------

        if ENABLE_WEB_TOOL:
            # Check if Tavily API key is available
            try:
                from Tools.web_search import client as web_client
                if web_client is not None:
                    self.register(
                        category="web",
                        tools=web_tools,
                    )
                    logger.info("Web tool enabled.")
                else:
                    logger.warning("Web tool flag is enabled but Tavily API key is missing. Tool will be disabled.")
            except Exception as e:
                logger.warning("Failed to initialize web tool: %s. Tool will be disabled.", e)
        else:
            logger.info("Web tool disabled.")

        # --------------------------------------------------
        # Weather tools
        # --------------------------------------------------

        if ENABLE_WEATHER_TOOL:
            self.register(
                category="weather",
                tools=weather_tools,
            )
            logger.info("Weather tool enabled.")
        else:
            logger.info("Weather tool disabled.")

        # --------------------------------------------------
        # Calculator tools
        # --------------------------------------------------

        if ENABLE_CALCULATOR_TOOL:
            self.register(
                category="calculator",
                tools=calculator_tool,
            )
            logger.info("Calculator tool enabled.")
        else:
            logger.info("Calculator tool disabled.")

        logger.info(
            "Built-in tool registry initialized with %d tools "
            "across %d categories.",
            len(self._tools),
            len(self._categories),
        )

        return self

    # ------------------------------------------------------
    # Registration
    # ------------------------------------------------------

    def register(self, category, tools):
        """
        Register one tool or multiple built-in tools.
        """

        category = category.lower().strip()

        if category not in self._categories:
            self._categories[category] = []

        # Accept a single tool or a collection of tools.
        if hasattr(tools, "name"):
            tools = [tools]

        for tool in tools:

            if not hasattr(tool, "name"):

                logger.warning(
                    "Skipping invalid tool in category '%s': %r",
                    category,
                    tool,
                )

                continue

            tool_name = tool.name.lower().strip()

            if tool_name not in self._tools:
                self._tools[tool_name] = tool

            if tool not in self._categories[category]:
                self._categories[category].append(tool)

    # ------------------------------------------------------
    # Access all tools
    # ------------------------------------------------------

    def get_all_tools(self):
        """
        Return all registered built-in tools.
        """

        return list(self._tools.values())

    # ------------------------------------------------------
    # Access category
    # ------------------------------------------------------

    def get_tools(self, category):
        """
        Return built-in tools belonging to a category.
        """

        category = category.lower().strip()

        return list(
            self._categories.get(category, [])
        )

    # ------------------------------------------------------
    # Access individual tool
    # ------------------------------------------------------

    def get_tool(self, name):
        """
        Return a built-in tool by name.
        """

        return self._tools.get(
            name.lower().strip()
        )

    # ------------------------------------------------------
    # Available categories
    # ------------------------------------------------------

    def categories(self):
        """
        Return categories containing built-in tools.
        """

        return list(
            self._categories.keys()
        )
