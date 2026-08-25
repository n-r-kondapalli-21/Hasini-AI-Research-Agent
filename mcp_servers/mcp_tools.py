from langchain_mcp_adapters.client import MultiServerMCPClient

from config import github_token


MCP_SERVERS = {
    "filesystem": {
        "transport": "stdio",
        "command": "mcp-server-filesystem",
        "args": [
            r"D:\Hasini_Ai_Research_Agent\AI_Research"
        ],
    },

    # "github": {
    #     "transport": "streamable_http",
    #     "url": "https://api.githubcopilot.com/mcp/",
    #     "headers": {
    #         "Authorization": f"Bearer {github_token}",
    #     },
    # },
}


async def get_mcp_tools():

    client = MultiServerMCPClient(
        MCP_SERVERS,
        tool_name_prefix=True,
    )

    tools = await client.get_tools()

    return client, tools


def get_active_mcp_servers():
    return list(MCP_SERVERS.keys())