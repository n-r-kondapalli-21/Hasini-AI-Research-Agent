import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import github_token

logger = logging.getLogger(__name__)


MCP_SERVERS = {
    "filesystem": {
        "transport": "stdio",
        "command": "mcp-server-filesystem",
        "args": [
            r"D:\Hasini_Ai_Research_Agent\AI_Research"
        ],
    },

    "github": {
        "transport": "streamable_http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {
            "Authorization": f"Bearer {github_token}",
        },
    },
}


async def get_mcp_tools():
    """
    Connect to each configured MCP server independently.

    If a server fails to connect or list tools (bad token, network
    down, binary missing, etc.), log a warning and skip it instead
    of losing tools from every other server.
    """

    tools = []
    working_servers = {}

    for server_name, server_config in MCP_SERVERS.items():

        try:
            single_client = MultiServerMCPClient(
                {server_name: server_config},
                tool_name_prefix=True,
            )

            server_tools = await single_client.get_tools()

            tools.extend(server_tools)
            working_servers[server_name] = server_config

            logger.info(
                "MCP server '%s' connected: %d tool(s) loaded.",
                server_name,
                len(server_tools),
            )

        except Exception as e:
            logger.warning(
                "MCP server '%s' failed to connect and will be skipped: %s",
                server_name,
                e,
            )

    # A single combined client, built only from servers that
    # actually connected, so downstream code can keep using
    # `client` normally (e.g. for reconnects, closing, etc.).
    client = MultiServerMCPClient(
        working_servers,
        tool_name_prefix=True,
    )

    return client, tools


def get_active_mcp_servers():
    return list(MCP_SERVERS.keys())