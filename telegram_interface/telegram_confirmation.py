"""
Telegram-based confirmation manager for permission-gated tools.

Provides Telegram-based interaction for user confirmation when a
permission-gated tool action requires permission (MEDIUM or HIGH tier).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

# Import confirmation exceptions from voice.confirmation
# These are used by the agent runtime for proper error handling
from voice.confirmation import ConfirmationRejected, ConfirmationTimeout

logger = logging.getLogger("hasini.telegram_confirmation")


class TelegramConfirmationManager:
    """
    Telegram-based confirmation manager for permission-gated tools.

    Uses a hybrid system:
    1. Check for pre-approved permissions (via /grant command)
    2. If not pre-approved, request instant permission with "y" or "yes" response
    """

    def __init__(
        self,
        send_message_callback: Callable[[str], None],
        permission_check_callback: Optional[Callable[[str], bool]] = None,
        set_pending_action_callback: Optional[Callable[[str, str], None]] = None,
        timeout_seconds: float = 60.0,
    ):
        """
        Initialize the Telegram confirmation manager.

        Args:
            send_message_callback: Function to send messages to Telegram chat.
            permission_check_callback: Optional function to check if user has pre-approved permission.
                Takes (tier) argument and returns bool.
            set_pending_action_callback: Optional function to store pending action for retry.
                Takes (tier, action_description) arguments.
            timeout_seconds: How long to wait for user input before timing out.
        """
        self.send_message_callback = send_message_callback
        self.permission_check_callback = permission_check_callback
        self.set_pending_action_callback = set_pending_action_callback
        self.timeout_seconds = timeout_seconds
        self._pending_confirmation: Optional[asyncio.Event] = None
        self._confirmation_result: Optional[bool] = None
        self._current_tier: str = "MEDIUM"

    async def wait_for_confirmation(
        self,
        tier: str,
        action_description: str,
    ) -> bool:
        """
        Check if the user has pre-approved permission or request instant permission.

        Args:
            tier: Permission tier ("MEDIUM" or "HIGH").
            action_description: Description of the action requiring permission.

        Returns:
            True if user has pre-approved permission or grants instant permission.

        Raises:
            ConfirmationRejected if user denies permission or times out.
        """
        tier_upper = str(tier).upper().strip()
        clean_action = str(action_description).rstrip(".")
        self._current_tier = tier_upper

        # Check if user has pre-approved permission
        if self.permission_check_callback and self.permission_check_callback(tier_upper):
            logger.info(f"User has pre-approved {tier_upper} permission")
            return True

        # User doesn't have pre-approved permission, request instant permission
        logger.info(f"Requesting instant {tier_upper} permission for: {clean_action}")

        # Store the pending action for potential retry
        if self.set_pending_action_callback:
            self.set_pending_action_callback(tier_upper, clean_action)

        # Create confirmation message
        if tier_upper == "HIGH":
            message = (
                f"⚠️ HIGH RISK PERMISSION REQUIRED\n\n"
                f"Action: {clean_action}\n\n"
                f"⚠️ This action could potentially cause data loss or security issues.\n\n"
                f"Reply with 'y' or 'yes' to ALLOW this action (valid for this action only).\n"
                f"Reply with 'n' or 'no' to DENY this action.\n\n"
                f"Or pre-approve permissions with: /grant {tier_upper.lower()}"
            )
        else:
            message = (
                f"🔒 PERMISSION REQUIRED ({tier_upper})\n\n"
                f"Action: {clean_action}\n\n"
                f"Reply with 'y' or 'yes' to ALLOW this action (valid for this action only).\n"
                f"Reply with 'n' or 'no' to DENY this action.\n\n"
                f"Or pre-approve permissions with: /grant {tier_upper.lower()}"
            )

        # Send confirmation request
        self.send_message_callback(message)

        # Wait for user response
        self._pending_confirmation = asyncio.Event()
        self._confirmation_result = None

        try:
            await asyncio.wait_for(
                self._pending_confirmation.wait(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.send_message_callback("❌ Confirmation timed out.")
            raise ConfirmationTimeout()
        finally:
            self._pending_confirmation = None

        if self._confirmation_result:
            self.send_message_callback("✅ Action confirmed.")
            return True
        else:
            self.send_message_callback("❌ Action rejected by user.")
            raise ConfirmationRejected()

    def handle_confirmation_response(self, response: str) -> None:
        """
        Handle user confirmation response from Telegram.

        Args:
            response: User's text response.
        """
        if self._pending_confirmation is None:
            return

        response_lower = response.strip().lower()

        if self._current_tier == "HIGH":
            self._confirmation_result = response_lower in {"yes", "confirm"}
        else:
            self._confirmation_result = response_lower in {
                "y",
                "yes",
                "confirm",
                "ok",
            }

        self._pending_confirmation.set()
