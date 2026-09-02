"""
Operation-level permission context management.

Provides a thread-safe and asyncio-safe (ContextVar) mechanism for scoping
permission approvals to a single logical operation / user turn.

When an operation is approved for a tool server (e.g. 'browser'), sub-actions
belonging to that server execute under the temporary approval context without
re-prompting the user. When the context exits or the turn ends, the approval
automatically expires.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

audit_logger = logging.getLogger("hasini.permissions")

# Numeric hierarchy for risk tiers (higher number = higher risk)
TIER_HIERARCHY = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


@dataclass
class GrantedScope:
    server_id: str
    max_tier: str
    description: str
    granted_at: float = field(default_factory=time.time)
    executed_actions: list[dict[str, Any]] = field(default_factory=list)


class PermissionContextManager:
    """
    Manages active operation permission contexts for the current execution task.
    """

    def __init__(self) -> None:
        self._scopes: dict[str, GrantedScope] = {}

    def grant_approval(
        self,
        server_id: str,
        tier: str = "MEDIUM",
        description: str = "",
    ) -> GrantedScope:
        """
        Grant an operation-level permission approval for a tool server.
        """
        tier_upper = str(tier).upper().strip()
        if tier_upper not in TIER_HIERARCHY:
            tier_upper = "HIGH"

        scope = GrantedScope(
            server_id=server_id,
            max_tier=tier_upper,
            description=description or f"Approved operation for {server_id}",
        )
        self._scopes[server_id] = scope

        audit_logger.info(
            "🔒 [OPERATION APPROVAL GRANTED] Server: '%s', Max Tier: '%s', Operation: '%s'",
            server_id,
            tier_upper,
            scope.description,
        )
        return scope

    def is_approved(self, server_id: str, required_tier: str) -> bool:
        """
        Check if a tool action is approved under an active operation context.
        """
        if server_id not in self._scopes:
            return False

        scope = self._scopes[server_id]
        req_level = TIER_HIERARCHY.get(str(required_tier).upper(), 3)
        granted_level = TIER_HIERARCHY.get(scope.max_tier, 3)

        return req_level <= granted_level

    def record_action(
        self,
        server_id: str,
        method_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """
        Record a sub-action executed under the active operation context for auditing.
        """
        if server_id in self._scopes:
            scope = self._scopes[server_id]
            scope.executed_actions.append(
                {
                    "method_name": method_name,
                    "arguments": arguments,
                    "timestamp": time.time(),
                }
            )
            audit_logger.info(
                "⚡ [SUB-ACTION EXECUTED UNDER OPERATION] %s.%s (Operation: '%s', Sub-action #%d)",
                server_id,
                method_name,
                scope.description,
                len(scope.executed_actions),
            )

    def revoke_approval(self, server_id: str) -> None:
        """Revoke approval for a specific server."""
        if server_id in self._scopes:
            scope = self._scopes.pop(server_id)
            audit_logger.info(
                "🔒 [OPERATION APPROVAL EXPIRED] Server: '%s', Operation: '%s', Total Executed Sub-actions: %d",
                server_id,
                scope.description,
                len(scope.executed_actions),
            )

    def clear(self) -> None:
        """Clear all active operation approvals."""
        for server_id in list(self._scopes.keys()):
            self.revoke_approval(server_id)


# Process-level default manager instance
_default_context_manager = PermissionContextManager()

# ContextVar to maintain isolated context per task/turn if explicitly set
_permission_context_var: ContextVar[PermissionContextManager] = ContextVar(
    "permission_context_var"
)


def get_permission_context() -> PermissionContextManager:
    """Get the active PermissionContextManager for the current task/turn."""
    try:
        return _permission_context_var.get()
    except LookupError:
        return _default_context_manager


def set_permission_context(mgr: PermissionContextManager | None = None) -> PermissionContextManager:
    """Set an isolated permission context manager for the current asyncio context."""
    if mgr is None:
        mgr = PermissionContextManager()
    _permission_context_var.set(mgr)
    return mgr


def clear_permission_context() -> None:
    """Clear all active operation approvals."""
    try:
        mgr = _permission_context_var.get()
        mgr.clear()
    except LookupError:
        _default_context_manager.clear()

