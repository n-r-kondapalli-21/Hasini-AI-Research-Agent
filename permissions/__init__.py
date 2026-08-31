"""
Centralized permission system for tool-agnostic permission gating.

This package provides a unified permission layer that can be used by
any tool system (MCP, browser automation, email, etc.) to enforce
permission-based access control with confirmation requirements.

Permission tiers:
- LOW: Execute immediately without confirmation
- MEDIUM: Require normal confirmation
- HIGH: Require strict confirmation

The permission registry resolves tool/action identifiers to permission tiers,
and the permission-gated wrapper enforces confirmation requirements.
"""

from .permissions import ToolCall, PermissionDecision, check
from .permission_registry import (
    discover_permissions,
    get_permission_tier,
    get_server_permission,
    get_resolved_registry,
    resolve_tool_server,
)
from .permission_gated_tool import create_permission_gated_tool

__all__ = [
    "ToolCall",
    "PermissionDecision",
    "check",
    "discover_permissions",
    "get_permission_tier",
    "get_server_permission",
    "get_resolved_registry",
    "resolve_tool_server",
    "create_permission_gated_tool",
]
