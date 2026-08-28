"""
Permission decision models and checks for tool execution.

The existing permission policy is preserved:
    LOW    -> execute immediately
    MEDIUM -> normal confirmation
    HIGH   -> strict confirmation

Unexpected permission tiers continue to fail safely as HIGH.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tool_management.permission_registry import get_permission_tier


logger = logging.getLogger("hasini.permissions")


@dataclass(frozen=True)
class ToolCall:
    """
    Normalized representation of a tool call.

    Attributes:
        server_id: MCP server that owns the tool.
        method_name: Actual MCP/LangChain tool name.
        arguments: Arguments supplied by the agent.
    """

    server_id: str
    method_name: str
    arguments: dict[str, Any]


class PermissionDecision:
    """Result of a permission check."""

    def __init__(
        self,
        tier: str,
        requires_confirmation: bool,
        confirmation_level: str | None = None,
    ) -> None:
        self.tier = tier
        self.requires_confirmation = requires_confirmation
        self.confirmation_level = confirmation_level

    def __repr__(self) -> str:
        return (
            "PermissionDecision("
            f"tier={self.tier!r}, "
            f"requires_confirmation={self.requires_confirmation!r}, "
            f"confirmation_level={self.confirmation_level!r}"
            ")"
        )


def check(tool_call: ToolCall) -> PermissionDecision:
    """
    Determine the permission required for a tool call.

    LOW:
        Execute immediately.

    MEDIUM:
        Require normal confirmation.

    HIGH:
        Require strict confirmation.

    Any unexpected tier is treated conservatively as HIGH.
    """

    if not isinstance(tool_call, ToolCall):
        raise TypeError(
            "tool_call must be an instance of ToolCall"
        )

    if not tool_call.server_id:
        raise ValueError("ToolCall.server_id cannot be empty")

    if not tool_call.method_name:
        raise ValueError("ToolCall.method_name cannot be empty")

    try:
        tier = get_permission_tier(
            tool_call.server_id,
            tool_call.method_name,
        )
    except Exception:
        logger.exception(
            "Failed to determine permission tier for %s.%s. "
            "Applying HIGH permission as a safety fallback.",
            tool_call.server_id,
            tool_call.method_name,
        )
        return PermissionDecision(
            tier="HIGH",
            requires_confirmation=True,
            confirmation_level="strict",
        )

    if tier == "LOW":
        decision = PermissionDecision(
            tier="LOW",
            requires_confirmation=False,
        )

    elif tier == "MEDIUM":
        decision = PermissionDecision(
            tier="MEDIUM",
            requires_confirmation=True,
            confirmation_level="normal",
        )

    else:
        # Conservative fallback:
        # Anything unexpected is treated as HIGH.
        logger.warning(
            "Unexpected permission tier %r for %s.%s; "
            "treating it as HIGH.",
            tier,
            tool_call.server_id,
            tool_call.method_name,
        )

        decision = PermissionDecision(
            tier="HIGH",
            requires_confirmation=True,
            confirmation_level="strict",
        )

    logger.debug(
        "Permission decision for %s.%s: %s",
        tool_call.server_id,
        tool_call.method_name,
        decision,
    )

    return decision
