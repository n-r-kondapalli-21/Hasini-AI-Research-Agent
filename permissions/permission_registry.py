"""
Permission registry and tool discovery.

Permission definitions are loaded from:
    permissions/permissions.json

This module is responsible for:
- Loading permission configuration.
- Discovering active tool servers.
- Assigning HIGH to unknown tool servers.
- Validating permission tiers.
- Resolving tool names to server + method.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import Callable


logger = logging.getLogger("permissions")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERMISSION_CONFIG = (
    Path(__file__).resolve().parent
    / "permissions.json"
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

# Callable to get active tool servers - must be set by the tool system
_get_active_servers_func: Callable[[], list[str]] | None = None


def register_server_discovery_func(func: Callable[[], list[str]]) -> None:
    """
    Register a function that returns active tool server IDs.
    
    This allows different tool systems (MCP, browser, email, etc.) to
    provide their own server discovery mechanism.
    
    Args:
        func: A callable that returns a list of server IDs (strings)
    """
    global _get_active_servers_func
    _get_active_servers_func = func


def _get_active_servers() -> list[str]:
    """
    Get active tool server IDs using the registered discovery function.
    
    Returns an empty list if no discovery function is registered.
    """
    if _get_active_servers_func is None:
        logger.warning(
            "No server discovery function registered. "
            "Returning empty server list."
        )
        return []
    
    try:
        return _get_active_servers_func()
    except Exception as e:
        logger.error(
            "Server discovery function raised an exception: %s",
            e,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_permission_registry() -> dict:
    """
    Load permission definitions from JSON.

    Invalid or missing configuration fails closed by returning
    an empty registry. Unknown tool servers will subsequently
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
            "Loaded permission registry: %d server(s)",
            len(REGISTRY),
        )

    except FileNotFoundError:

        REGISTRY = {}

        logger.error(
            "Permission file not found: %s "
            "— all tool servers will default to HIGH",
            PERMISSION_CONFIG,
        )

    except json.JSONDecodeError as e:

        REGISTRY = {}

        logger.error(
            "Invalid JSON in %s: %s "
            "— all tool servers will default to HIGH",
            PERMISSION_CONFIG,
            e,
        )

    except Exception as e:

        REGISTRY = {}

        logger.error(
            "Failed to load permission registry: %s "
            "— all tool servers will default to HIGH",
            e,
        )

    return REGISTRY


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_tier(tier: str) -> str:
    """
    Validate a permission tier.

    Invalid configuration always falls back to HIGH.
    """

    tier = str(tier).upper()

    if tier not in RISK_TIERS:

        logger.warning(
            "Invalid risk tier '%s' detected "
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
    Validate and normalize one tool server permission configuration.
    """

    if not isinstance(config, dict):

        logger.warning(
            "Invalid permission configuration for tool server '%s' "
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
            "Invalid method permission configuration for tool server '%s' "
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
# Tool server discovery
# ---------------------------------------------------------------------------

def discover_permissions():
    """
    Discover currently active tool servers and build the
    runtime permission registry.

    Unknown tool servers automatically receive HIGH.
    """

    global _RESOLVED_REGISTRY

    load_permission_registry()

    _RESOLVED_REGISTRY = {}

    active_servers = _get_active_servers()

    logger.info(
        "Discovering tool servers..."
    )

    for server_id in active_servers:

        if server_id in REGISTRY:

            config = _validate_server_config(
                server_id,
                REGISTRY[server_id],
            )

            _RESOLVED_REGISTRY[server_id] = config

            logger.info(
                "Registered tool server '%s' → %s",
                server_id,
                config["default"],
            )

        else:

            _RESOLVED_REGISTRY[server_id] = {
                "default": DEFAULT_UNKNOWN_TIER,
                "methods": {},
            }

            logger.warning(
                "Unregistered tool server '%s' detected "
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
    Log a compact runtime permission table.
    """

    if not _RESOLVED_REGISTRY:

        logger.info(
            "Permission tier table: <no tool servers>"
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
    Return the resolved permission configuration for a tool server.

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
    Resolve the final permission tier for a tool server + method.

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
    Return a copy of the current runtime permission registry.
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
# Tool → server resolution
# ---------------------------------------------------------------------------

def resolve_tool_server(
    tool_name: str,
) -> tuple[str, str]:
    """
    Resolve an adapter-prefixed tool name to server + method.

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
