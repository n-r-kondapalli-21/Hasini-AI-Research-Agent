from dataclasses import dataclass
from typing import Any

from permission_registry import get_permission_tier


@dataclass
class ToolCall:
    """
    Normalized representation of a tool call.

    server_id:
        MCP server that owns the tool.

    method_name:
        Actual MCP/LangChain tool name.

    arguments:
        Arguments supplied by the agent.
    """

    server_id: str
    method_name: str
    arguments: dict[str, Any]


class PermissionDecision:
    """
    Result of a permission check.
    """

    def __init__(
        self,
        tier: str,
        requires_confirmation: bool,
        confirmation_level: str | None = None,
    ):
        self.tier = tier
        self.requires_confirmation = requires_confirmation
        self.confirmation_level = confirmation_level

    def __repr__(self):
        return (
            "PermissionDecision("
            f"tier={self.tier!r}, "
            f"requires_confirmation="
            f"{self.requires_confirmation!r}, "
            f"confirmation_level="
            f"{self.confirmation_level!r}"
            ")"
        )

def check(tool_call: ToolCall) -> PermissionDecision:
    """
    Hard permission decision.

    LOW:
        Execute immediately.

    MEDIUM:
        Require normal confirmation.

    HIGH:
        Require strict confirmation.
    """

    tier = get_permission_tier(
        tool_call.server_id,
        tool_call.method_name,
    )

    if tier == "LOW":

        return PermissionDecision(
            tier="LOW",
            requires_confirmation=False,
        )

    if tier == "MEDIUM":

        return PermissionDecision(
            tier="MEDIUM",
            requires_confirmation=True,
            confirmation_level="normal",
        )

    # Conservative fallback.
    #
    # Anything unexpected is treated as HIGH.
    return PermissionDecision(
        tier="HIGH",
        requires_confirmation=True,
        confirmation_level="strict",
    )