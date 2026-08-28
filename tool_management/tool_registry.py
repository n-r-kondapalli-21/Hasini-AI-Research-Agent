
import logging

from Tools.web_search import web_tools
from Tools.weather import weather_tools
from Tools.calculator import calculator_tool
from mcp_servers.mcp_tools import get_mcp_tools


from tool_management.permission_gated_tool import (create_permission_gated_tool,)

from tool_management.permission_registry import (discover_permissions,resolve_tool_server,)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all tools available to Hasini.

    The registry is responsible for:
        - Loading built-in tools
        - Loading MCP tools
        - Applying permission gates to MCP tools
        - Grouping tools by category
        - Providing access to individual or all tools
    """

    def __init__(self, confirmation_manager=None):

        self.confirmation_manager = confirmation_manager

        self._tools = {}
        self._categories = {}

        self.mcp_client = None

    async def initialize(self):
        """
        Load and register all available tools.
        """

        # --------------------------------------------------
        # Discover permissions before loading MCP tools
        # --------------------------------------------------

        discover_permissions()

        # --------------------------------------------------
        # Web tools
        # --------------------------------------------------

        self.register(
            category="web",
            tools=web_tools,
        )

        # --------------------------------------------------
        # Weather tools
        # --------------------------------------------------

        self.register(
            category="weather",
            tools=weather_tools,
        )
        #------------------------------------------------------
        #calculator tools
        #-----------------------------------------------------
        self.register(
            category="calculator",
            tools=calculator_tool,
        )



        # --------------------------------------------------
        # MCP tools
        # --------------------------------------------------

        mcp_client, mcp_tools = await get_mcp_tools()

        self.mcp_client = mcp_client

        gated_mcp_tools = []

        for tool in mcp_tools:

            server_id, method_name = resolve_tool_server(
                tool.name
            )

            gated_tool = create_permission_gated_tool(
                tool=tool,
                server_id=server_id,
                method_name=method_name,
                confirmation_manager=self.confirmation_manager,
            )

            gated_mcp_tools.append(gated_tool)

        self.register(
            category="mcp",
            tools=gated_mcp_tools,
        )

        logger.info(
            "Tool registry initialized with %d tools.",
            len(self._tools),
        )

        return self

    # ------------------------------------------------------
    # Registration
    # ------------------------------------------------------

    def register(self, category, tools):
        """
        Register one tool or multiple tools under a category.
        """

        category = category.lower().strip()

        if category not in self._categories:
            self._categories[category] = []

        # Accept a single tool or a collection of tools
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
        Return every registered tool.
        """

        return list(self._tools.values())

    # ------------------------------------------------------
    # Access category
    # ------------------------------------------------------

    def get_tools(self, category):
        """
        Return tools belonging to a category.
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
        Return a tool by name.
        """

        return self._tools.get(
            name.lower().strip()
        )

    # ------------------------------------------------------
    # Available categories
    # ------------------------------------------------------

    def categories(self):
        """
        Return all registered categories.
        """

        return list(
            self._categories.keys()
        )

    # ------------------------------------------------------
    # MCP client
    # ------------------------------------------------------

    def get_mcp_client(self):
        """
        Return the MCP client.
        """

        return self.mcp_client

