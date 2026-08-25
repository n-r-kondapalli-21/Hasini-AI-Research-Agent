"""
Permission registry and MCP server discovery.

This module is responsible for:
- Defining known MCP server risk tiers.
- Discovering currently registered MCP servers.
- Assigning HIGH to unknown servers.
- Warning loudly about unclassified servers.
- Providing a resolved runtime permission table.

It does NOT decide individual tool calls.
That responsibility belongs to permissions.py.
"""

import logging
import fnmatch

from mcp_servers.mcp_tools import get_active_mcp_servers


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger("permissions")


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
# Static permission registry
# ---------------------------------------------------------------------------

REGISTRY = {
    "filesystem": {
        "default": "MEDIUM",
        "methods": {
            "read_*": "LOW",
            "list_*": "LOW",
            "get_*": "LOW",
            "search_*": "LOW",

            "delete_*": "MEDIUM",
            "move_*": "MEDIUM",
            "write_*": "MEDIUM",
            "edit_*": "MEDIUM",
            "create_*": "MEDIUM",
        },
    },


    # Example for future servers:
    #
    # "github": {
    #     "default": "MEDIUM",
    #     "methods": {
    #         "delete_*": "HIGH",
    #     },
    # },
}


# ---------------------------------------------------------------------------
# Runtime resolved registry
# ---------------------------------------------------------------------------

_RESOLVED_REGISTRY = {}


def _validate_tier(tier: str) -> str:
    """
    Validate a permission tier.

    Invalid configuration falls back to HIGH rather than
    accidentally making a tool less restrictive.
    """
    tier = str(tier).upper()

    if tier not in RISK_TIERS:
        logger.warning(
            "[permissions] Invalid risk tier '%s' detected — "
            "defaulting to HIGH",
            tier,
        )
        return DEFAULT_UNKNOWN_TIER

    return tier


def _validate_server_config(server_id: str, config: dict) -> dict:
    """
    Validate and normalize one registry entry.
    """

    if not isinstance(config, dict):
        logger.warning(
            "[permissions] Invalid configuration for MCP server '%s' "
            "— defaulting to HIGH",
            server_id,
        )

        return {
            "default": DEFAULT_UNKNOWN_TIER,
            "methods": {},
        }

    default_tier = _validate_tier(
        config.get("default", DEFAULT_UNKNOWN_TIER)
    )

    methods = config.get("methods", {})

    if not isinstance(methods, dict):
        logger.warning(
            "[permissions] Invalid method configuration for '%s' "
            "— ignoring method overrides",
            server_id,
        )
        methods = {}

    validated_methods = {}

    for pattern, tier in methods.items():
        validated_methods[pattern] = _validate_tier(tier)

    return {
        "default": default_tier,
        "methods": validated_methods,
    }


def discover_permissions():
    """
    Discover currently registered MCP servers and build the
    runtime permission registry.

    Unknown servers are automatically assigned HIGH.
    """

    global _RESOLVED_REGISTRY

    _RESOLVED_REGISTRY = {}

    active_servers = get_active_mcp_servers()

    logger.info("[permissions] Discovering MCP servers...")

    for server_id in active_servers:

        if server_id in REGISTRY:

            config = _validate_server_config(
                server_id,
                REGISTRY[server_id],
            )

            _RESOLVED_REGISTRY[server_id] = config

            logger.info(
                "[permissions] Registered MCP server '%s' → %s",
                server_id,
                config["default"],
            )

        else:

            _RESOLVED_REGISTRY[server_id] = {
                "default": DEFAULT_UNKNOWN_TIER,
                "methods": {},
            }

            logger.warning(
                "[permissions] Unregistered MCP server '%s' detected "
                "— defaulting to HIGH until classified in "
                "permission_registry.py",
                server_id,
            )

    _log_resolved_registry()

    return _RESOLVED_REGISTRY


def _log_resolved_registry():
    """
    Print a compact runtime permission table.
    """

    if not _RESOLVED_REGISTRY:
        logger.info(
            "[permissions] Permission tier table: <no MCP servers>"
        )
        return

    parts = []

    for server_id, config in _RESOLVED_REGISTRY.items():

        parts.append(
            f"{server_id}={config['default']}"
        )

    logger.info(
        "[permissions] Permission tier table: %s",
        ", ".join(parts),
    )


def get_server_permission(server_id: str) -> dict:
    """
    Return the resolved permission configuration for a server.

    Automatically performs discovery if the runtime registry
    has not been initialized yet.
    """

    if not _RESOLVED_REGISTRY:
        discover_permissions()

    if server_id not in _RESOLVED_REGISTRY:
        return {
            "default": DEFAULT_UNKNOWN_TIER,
            "methods": {},
        }

    return _RESOLVED_REGISTRY[server_id]


def get_permission_tier(
    server_id: str,
    method_name: str | None = None,
) -> str:
    """
    Resolve the final risk tier for a server + method.

    Method-level patterns take priority over the server default.
    """

    config = get_server_permission(server_id)

    if method_name:
        for pattern, tier in config["methods"].items():
            if fnmatch.fnmatch(method_name, pattern):
                return tier

    return config["default"]


def get_resolved_registry() -> dict:
    """
    Return a copy of the current runtime permission registry.
    """

    return {
        server_id: {
            "default": config["default"],
            "methods": dict(config["methods"]),
        }
        for server_id, config in _RESOLVED_REGISTRY.items()
    }



def resolve_tool_server(tool_name: str) -> tuple[str, str]:
    """
    Resolve an adapter-prefixed tool name into:

        (server_id, method_name)

    Example:

        filesystem_read_multiple_files
        ->
        ("filesystem", "read_multiple_files")

    Server IDs come from the active permission registry rather
    than a hardcoded tool list.
    """

    if not _RESOLVED_REGISTRY:
        discover_permissions()

    # Longest server ID first protects against overlapping
    # server names.
    server_ids = sorted(
        _RESOLVED_REGISTRY.keys(),
        key=len,
        reverse=True,
    )

    for server_id in server_ids:

        prefix = f"{server_id}_"

        if tool_name.startswith(prefix):

            method_name = tool_name[len(prefix):]

            if not method_name:
                break

            return server_id, method_name

    # Unknown/unresolvable tool.
    #
    # Do NOT guess the server.
    return "unknown", tool_name