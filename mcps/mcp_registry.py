"""
MCP registry.

Responsible for:
- Initializing MCP servers.
- Discovering MCP tools.
- Applying permission gates to MCP tools.
- Grouping MCP tools by MCP server.
- Providing the combined MCP client.
"""

from __future__ import annotations

import logging

from mcps.mcp_tools import get_mcp_tools

from mcps.mcp_permissions.permission_registry import (
    discover_permissions,
    resolve_tool_server,
)

from mcps.mcp_permissions.permission_gated_tool import (
    create_permission_gated_tool,
)


logger = logging.getLogger(__name__)


class MCPRegistry:
    """
    Registry for MCP servers and their tools.
    """

    def __init__(self, confirmation_manager=None):

        self.confirmation_manager = confirmation_manager

        self._tools = {}
        self._categories = {}

        self.mcp_client = None

    async def initialize(self):
        """
        Discover MCP servers, load their tools, apply permission
        gates, and group tools by MCP server.
        """

        # --------------------------------------------------
        # Discover permissions
        # --------------------------------------------------

        discover_permissions()

        # --------------------------------------------------
        # Load MCP tools
        # --------------------------------------------------

        mcp_client, mcp_tools = await get_mcp_tools()

        self.mcp_client = mcp_client

        for tool in mcp_tools:

            # --------------------------------------------------
            # Resolve server + method
            # --------------------------------------------------

            server_id, method_name = resolve_tool_server(
                tool.name
            )

            # --------------------------------------------------
            # Apply permission gate
            # --------------------------------------------------

            gated_tool = create_permission_gated_tool(
                tool=tool,
                server_id=server_id,
                method_name=method_name,
                confirmation_manager=self.confirmation_manager,
            )

            # --------------------------------------------------
            # Register by MCP server
            # --------------------------------------------------

            if server_id != "unknown":

                self.register(
                    category=server_id,
                    tool=gated_tool,
                )

                logger.debug(
                    "Registered MCP tool '%s' under '%s'.",
                    tool.name,
                    server_id,
                )

            else:

                logger.warning(
                    "MCP tool '%s' could not be mapped to a "
                    "known MCP server.",
                    tool.name,
                )

        logger.info(
            "MCP registry initialized with %d tools across %d servers.",
            len(self._tools),
            len(self._categories),
        )

        return self

    # ------------------------------------------------------
    # Registration
    # ------------------------------------------------------

    def register(self, category, tool):
        """
        Register one MCP tool under a server category.
        """

        category = category.lower().strip()

        if category not in self._categories:
            self._categories[category] = []

        if not hasattr(tool, "name"):

            logger.warning(
                "Skipping invalid MCP tool in category '%s': %r",
                category,
                tool,
            )

            return

        tool_name = tool.name.lower().strip()

        if tool_name not in self._tools:
            self._tools[tool_name] = tool

        if tool not in self._categories[category]:
            self._categories[category].append(tool)

    # ------------------------------------------------------
    # Access all MCP tools
    # ------------------------------------------------------

    def get_all_tools(self):
        """
        Return every registered MCP tool.
        """

        return list(self._tools.values())

    # ------------------------------------------------------
    # Access MCP category
    # ------------------------------------------------------

    def get_tools(self, category):
        """
        Return MCP tools belonging to one server.
        """

        category = category.lower().strip()

        return list(
            self._categories.get(category, [])
        )

    # ------------------------------------------------------
    # MCP categories
    # ------------------------------------------------------

    def categories(self):
        """
        Return registered MCP server IDs.
        """

        return list(
            self._categories.keys()
        )

    # ------------------------------------------------------
    # Individual tool
    # ------------------------------------------------------

    def get_tool(self, name):
        """
        Return an MCP tool by name.
        """

        return self._tools.get(
            name.lower().strip()
        )

    # ------------------------------------------------------
    # MCP client
    # ------------------------------------------------------

    def get_client(self):
        """
        Return the combined MCP client.
        """

        return self.mcp_client