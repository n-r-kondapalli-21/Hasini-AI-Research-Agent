"""
MCP server client utilities.

Responsible for:
- Loading MCP server configuration from mcp_config/mcp_servers.json.
- Connecting to configured MCP servers independently.
- Discovering MCP tools.
- Returning a combined MCP client.
- Reporting active configured MCP servers.

Environment-variable substitution is enabled using ${VAR_NAME} syntax.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (
    ENABLE_MCP_FILESYSTEM,
    ENABLE_MCP_GITHUB,
    ENABLE_MCP_OPENALGO,
)
from permissions.permission_registry import register_server_discovery_func

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Store the list of successfully connected MCP servers
_connected_mcp_servers: list[str] = []


def get_connected_mcp_servers() -> list[str]:
    """
    Return the list of MCP servers that successfully connected.
    """
    return _connected_mcp_servers


# Register the connected servers discovery function with the permission system
register_server_discovery_func(get_connected_mcp_servers)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_SERVER_CONFIG = (
    Path(__file__).resolve().parent
    / "mcp_servers.json"
)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def substitute_env_vars(value):
    """
    Recursively substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} syntax. If the environment variable is not set,
    the placeholder is left unchanged.
    
    Args:
        value: Configuration value (str, dict, list, or other)
        
    Returns:
        Configuration value with environment variables substituted
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(
                    "Environment variable '%s' not set, keeping placeholder",
                    var_name,
                )
                return match.group(0)  # Keep the placeholder
            return env_value
        
        return re.sub(pattern, replace_var, value)
    
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    
    else:
        return value


def load_mcp_servers() -> dict:
    """
    Load MCP server configuration from JSON and filter by enable/disable flags.

    Environment-variable substitution is enabled using ${VAR_NAME} syntax.
    """

    try:
        with MCP_SERVER_CONFIG.open(
            "r",
            encoding="utf-8",
        ) as file:
            servers = json.load(file)

    except FileNotFoundError:
        logger.error(
            "MCP server configuration file not found: %s",
            MCP_SERVER_CONFIG,
        )
        return {}

    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in MCP server configuration '%s': %s",
            MCP_SERVER_CONFIG,
            e,
        )
        return {}

    except Exception as e:
        logger.error(
            "Failed to load MCP server configuration: %s",
            e,
        )
        return {}

    if not isinstance(servers, dict):
        logger.error(
            "MCP server configuration must contain a JSON object."
        )
        return {}

    # Substitute environment variables in the configuration
    servers = substitute_env_vars(servers)

    # Filter servers based on enable/disable flags
    filtered_servers = {}
    
    for server_name, server_config in servers.items():
        server_enabled = True
        
        if server_name == "filesystem" and not ENABLE_MCP_FILESYSTEM:
            server_enabled = False
        elif server_name == "github" and not ENABLE_MCP_GITHUB:
            server_enabled = False
        elif server_name == "openalgo" and not ENABLE_MCP_OPENALGO:
            server_enabled = False
        
        if server_enabled:
            filtered_servers[server_name] = server_config
            logger.info("MCP server '%s' enabled.", server_name)
        else:
            logger.info("MCP server '%s' disabled.", server_name)
    
    return filtered_servers


# ---------------------------------------------------------------------------
# MCP tool discovery
# ---------------------------------------------------------------------------

async def get_mcp_tools():
    """
    Connect to each configured MCP server independently.

    A failure in one MCP server does not prevent other MCP servers
    from loading.

    Returns:
        tuple:
            (combined MCP client, discovered MCP tools)
    """

    global _connected_mcp_servers

    mcp_servers = load_mcp_servers()

    if not mcp_servers:
        logger.warning(
            "No MCP servers are configured."
        )

        return (
            MultiServerMCPClient(
                {},
                tool_name_prefix=True,
            ),
            [],
        )

    tools = []
    working_servers = {}
    _connected_mcp_servers = []

    for server_name, server_config in mcp_servers.items():

        try:

            logger.info(
                "Connecting to MCP server '%s'...",
                server_name,
            )

            single_client = MultiServerMCPClient(
                {
                    server_name: server_config,
                },
                tool_name_prefix=True,
            )

            server_tools = await single_client.get_tools()

            tools.extend(server_tools)

            working_servers[server_name] = server_config
            _connected_mcp_servers.append(server_name)

            logger.info(
                "MCP server '%s' connected: %d tool(s) loaded.",
                server_name,
                len(server_tools),
            )

        except Exception as e:

            logger.exception(
                "MCP server '%s' failed to connect and will be skipped.",
                server_name,
            )

    # ------------------------------------------------------------------
    # Combined client
    # ------------------------------------------------------------------

    client = MultiServerMCPClient(
        working_servers,
        tool_name_prefix=True,
    )

    logger.info(
        "MCP initialization complete: %d server(s), %d tool(s).",
        len(working_servers),
        len(tools),
    )

    return client, tools


# ---------------------------------------------------------------------------
# Active MCP servers
# ---------------------------------------------------------------------------

def get_active_mcp_servers() -> list[str]:
    """
    Return configured MCP server IDs.

    These are the server IDs defined in mcp_servers.json.
    """

    servers = load_mcp_servers()

    return list(
        servers.keys()
    )