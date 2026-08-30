
"""
MCP permission registry and server discovery.

Permission definitions are loaded from:
    mcps/mcp_permissions/mcp_permissions.json

This module is responsible for:
- Loading MCP permission configuration.
- Discovering active MCP servers.
- Assigning HIGH to unknown MCP servers.
- Validating MCP permission tiers.
- Resolving MCP tool names to server + method.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path

from mcps.mcp_tools import get_active_mcp_servers


logger = logging.getLogger("permissions")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERMISSION_CONFIG = (
    Path(__file__).resolve().parent
    / "mcp_permissions.json"
)


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------

RISK_TIERS = {
    "LOW",
    "MEDIUM",
    "HIGH",
}

DEFAULT_UNKNOWN_TIER = "HIGH"


# ---------------------------------------------------------------------------
# Runtime registries
# ---------------------------------------------------------------------------

REGISTRY: dict = {}
_RESOLVED_REGISTRY: dict = {}


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_permission_registry() -> dict:
    """
    Load MCP permission definitions from JSON.

    Invalid or missing configuration fails closed by returning
    an empty registry. Unknown MCP servers will subsequently
    receive HIGH permission.
    """

    global REGISTRY

    try:
        with PERMISSION_CONFIG.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Permission configuration root must be an object."
            )

        REGISTRY = data

        logger.info(
            "Loaded MCP permission registry: %d server(s)",
            len(REGISTRY),
        )

    except FileNotFoundError:

        REGISTRY = {}

        logger.error(
            "MCP permission file not found: %s "
            "— all MCP servers will default to HIGH",
            PERMISSION_CONFIG,
        )

    except json.JSONDecodeError as e:

        REGISTRY = {}

        logger.error(
            "Invalid JSON in %s: %s "
            "— all MCP servers will default to HIGH",
            PERMISSION_CONFIG,
            e,
        )

    except Exception as e:

        REGISTRY = {}

        logger.error(
            "Failed to load MCP permission registry: %s "
            "— all MCP servers will default to HIGH",
            e,
        )

    return REGISTRY


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_tier(tier: str) -> str:
    """
    Validate an MCP permission tier.

    Invalid configuration always falls back to HIGH.
    """

    tier = str(tier).upper()

    if tier not in RISK_TIERS:

        logger.warning(
            "Invalid MCP risk tier '%s' detected "
            "— defaulting to HIGH",
            tier,
        )

        return DEFAULT_UNKNOWN_TIER

    return tier


def _validate_server_config(
    server_id: str,
    config: dict,
) -> dict:
    """
    Validate and normalize one MCP server permission configuration.
    """

    if not isinstance(config, dict):

        logger.warning(
            "Invalid permission configuration for MCP server '%s' "
            "— defaulting to HIGH",
            server_id,
        )

        return {
            "default": DEFAULT_UNKNOWN_TIER,
            "methods": {},
        }

    default_tier = _validate_tier(
        config.get(
            "default",
            DEFAULT_UNKNOWN_TIER,
        )
    )

    methods = config.get(
        "methods",
        {},
    )

    if not isinstance(methods, dict):

        logger.warning(
            "Invalid method permission configuration for MCP server '%s' "
            "— ignoring method overrides",
            server_id,
        )

        methods = {}

    validated_methods = {}

    for pattern, tier in methods.items():

        validated_methods[str(pattern)] = (
            _validate_tier(tier)
        )

    return {
        "default": default_tier,
        "methods": validated_methods,
    }


# ---------------------------------------------------------------------------
# MCP server discovery
# ---------------------------------------------------------------------------

def discover_permissions():
    """
    Discover currently active MCP servers and build the
    runtime permission registry.

    Unknown MCP servers automatically receive HIGH.
    """

    global _RESOLVED_REGISTRY

    load_permission_registry()

    _RESOLVED_REGISTRY = {}

    active_servers = get_active_mcp_servers()

    logger.info(
        "Discovering MCP servers..."
    )

    for server_id in active_servers:

        if server_id in REGISTRY:

            config = _validate_server_config(
                server_id,
                REGISTRY[server_id],
            )

            _RESOLVED_REGISTRY[server_id] = config

            logger.info(
                "Registered MCP server '%s' → %s",
                server_id,
                config["default"],
            )

        else:

            _RESOLVED_REGISTRY[server_id] = {
                "default": DEFAULT_UNKNOWN_TIER,
                "methods": {},
            }

            logger.warning(
                "Unregistered MCP server '%s' detected "
                "— defaulting to HIGH",
                server_id,
            )

    _log_resolved_registry()

    return _RESOLVED_REGISTRY


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_resolved_registry() -> None:
    """
    Log a compact runtime MCP permission table.
    """

    if not _RESOLVED_REGISTRY:

        logger.info(
            "Permission tier table: <no MCP servers>"
        )

        return

    parts = []

    for server_id, config in _RESOLVED_REGISTRY.items():

        parts.append(
            f"{server_id}={config['default']}"
        )

    logger.info(
        "Permission tier table: %s",
        ", ".join(parts),
    )


# ---------------------------------------------------------------------------
# Server permissions
# ---------------------------------------------------------------------------

def get_server_permission(
    server_id: str,
) -> dict:
    """
    Return the resolved permission configuration for an MCP server.

    If the runtime registry has not yet been initialized,
    permission discovery is performed automatically.
    """

    if not _RESOLVED_REGISTRY:

        discover_permissions()

    return _RESOLVED_REGISTRY.get(
        server_id,
        {
            "default": DEFAULT_UNKNOWN_TIER,
            "methods": {},
        },
    )


# ---------------------------------------------------------------------------
# Tool permission
# ---------------------------------------------------------------------------

def get_permission_tier(
    server_id: str,
    method_name: str | None = None,
) -> str:
    """
    Resolve the final permission tier for an MCP server + method.

    Method-level patterns take priority over the server default.
    """

    config = get_server_permission(
        server_id
    )

    if method_name:

        for pattern, tier in config["methods"].items():

            if fnmatch.fnmatch(
                method_name,
                pattern,
            ):

                return tier

    return config["default"]


# ---------------------------------------------------------------------------
# Runtime registry
# ---------------------------------------------------------------------------

def get_resolved_registry() -> dict:
    """
    Return a copy of the current runtime MCP permission registry.
    """

    return {
        server_id: {
            "default": config["default"],
            "methods": dict(
                config["methods"]
            ),
        }
        for server_id, config in _RESOLVED_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# Tool → MCP server resolution
# ---------------------------------------------------------------------------

def resolve_tool_server(
    tool_name: str,
) -> tuple[str, str]:
    """
    Resolve an adapter-prefixed MCP tool name.

    Example:

        openalgo_get_quote
        ->
        ("openalgo", "get_quote")

    Returns:

        (server_id, method_name)

    Unknown or unresolvable tools return:

        ("unknown", tool_name)
    """

    if not _RESOLVED_REGISTRY:

        discover_permissions()

    if not isinstance(tool_name, str):

        return (
            "unknown",
            str(tool_name),
        )

    server_ids = sorted(
        _RESOLVED_REGISTRY.keys(),
        key=len,
        reverse=True,
    )

    for server_id in server_ids:

        prefix = f"{server_id}_"

        if tool_name.startswith(prefix):

            method_name = tool_name[
                len(prefix):
            ]

            if not method_name:
                break

            return (
                server_id,
                method_name,
            )

    return (
        "unknown",
        tool_name,
    )
