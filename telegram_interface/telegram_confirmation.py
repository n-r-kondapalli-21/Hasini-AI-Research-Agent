"""
Telegram-based confirmation manager for permission-gated tools.

Provides Telegram-based interaction for user confirmation when a
permission-gated tool action requires permission (MEDIUM or HIGH tier).

Confirmation state is keyed per user_id, because the bot runs with
concurrent_updates=True (see telegram_main.py), so more than one user's
request can be suspended waiting on a confirmation at the same time.

wait_for_confirmation() is called from deep inside LangGraph's tool
execution (permission_gated_tool.py), which has no direct notion of
"which Telegram user is this" and calls this method with only
(tier, action_description). So `user_id` here is optional: if the
caller doesn't supply it, we resolve it via `get_current_user_id_callback`,
which HasiniTelegramBot wires up to whichever user's request is
currently in flight.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Optional

# Import confirmation exceptions from voice.confirmation
# These are used by the agent runtime for proper error handling
from voice.confirmation import ConfirmationRejected, ConfirmationTimeout

logger = logging.getLogger("hasini.telegram_confirmation")

# Sentinel key used only if we truly cannot determine a user_id (no
# user_id argument AND no callback AND no active user set). Keeps a
# single shared slot as a last-resort fallback rather than crashing,
# matching the old single-user behavior for anything that doesn't
# know about per-user tracking.
_UNKNOWN_USER = -1


class TelegramConfirmationManager:
    """
    Telegram-based confirmation manager for permission-gated tools.

    Uses a hybrid system:
    1. Check for pre-approved permissions (via /grant command)
    2. If not pre-approved, request instant permission with "y" or "yes" response

    All per-request state is keyed by user_id so concurrent requests from
    different users don't interfere with each other.
    """

    def __init__(
        self,
        send_message_callback: Callable[[str, bool], None],
        permission_check_callback: Optional[Callable[[str], bool]] = None,
        set_pending_action_callback: Optional[Callable[[str, str], None]] = None,
        timeout_seconds: float = 100.0,
        get_current_user_id_callback: Optional[Callable[[], Optional[int]]] = None,
    ):
        """
        Initialize the Telegram confirmation manager.

        Args:
            send_message_callback: Function to send messages to Telegram chat.
                Signature: (message: str, is_new_request: bool) -> None.
                `is_new_request` is True ONLY for the initial permission-request
                prompt (the one that actually needs a y/n reply). It is False
                for follow-up status notices ("confirmed", "rejected", "timed
                out"). The caller MUST only mark the user as "awaiting
                confirmation" when is_new_request is True — marking it for
                every message (including the follow-up status notices) causes
                the very next unrelated message from the user to be wrongly
                treated as a confirmation reply. See telegram_main.py's
                _send_admin_message for the correct handling.
            permission_check_callback: Optional function to check if user has pre-approved permission.
                Takes (tier) argument and returns bool.
            set_pending_action_callback: Optional function to store pending action for retry.
                Takes (tier, action_description) arguments.
            timeout_seconds: How long to wait for user input before timing out.
            get_current_user_id_callback: Optional zero-arg function returning the
                Telegram user_id of whichever request is currently in flight.
                Used as a fallback to resolve user_id when wait_for_confirmation()
                is called without one directly (e.g. from deep inside LangGraph's
                tool execution, which has no notion of "Telegram user").
        """
        self.send_message_callback = send_message_callback
        self.permission_check_callback = permission_check_callback
        self.set_pending_action_callback = set_pending_action_callback
        self.timeout_seconds = timeout_seconds
        self.get_current_user_id_callback = get_current_user_id_callback

        # Per-user confirmation state, so concurrent requests from
        # different users don't share/overwrite each other's state.
        self._pending_confirmations: Dict[int, asyncio.Event] = {}
        self._confirmation_results: Dict[int, bool] = {}
        self._current_tiers: Dict[int, str] = {}

    def _resolve_user_id(self, user_id: Optional[int]) -> int:
        """Resolve the effective user_id to key confirmation state under."""
        if user_id is not None:
            return user_id
        if self.get_current_user_id_callback is not None:
            resolved = self.get_current_user_id_callback()
            if resolved is not None:
                return resolved
        logger.warning(
            "wait_for_confirmation() called with no user_id and no resolvable "
            "current user; falling back to a shared confirmation slot. "
            "Concurrent requests from different users may interfere."
        )
        return _UNKNOWN_USER

    def is_awaiting_confirmation(self, user_id: Optional[int] = None) -> bool:
        """
        Return True if a wait_for_confirmation() call is currently blocked
        waiting on a response for this specific user (or the resolved
        current user, if user_id is omitted).

        Used by callers (e.g. the Telegram message handler) to distinguish
        between "the original request is still alive and listening" and
        "that confirmation window already timed out / closed", so a late
        reply doesn't trigger a duplicate retry of the original action.
        """
        resolved = self._resolve_user_id(user_id)
        return resolved in self._pending_confirmations

    async def wait_for_confirmation(
        self,
        tier: str,
        action_description: str,
        user_id: Optional[int] = None,
    ) -> bool:
        """
        Check if the user has pre-approved permission or request instant permission.

        Args:
            tier: Permission tier ("MEDIUM" or "HIGH").
            action_description: Description of the action requiring permission.
            user_id: Telegram user ID this confirmation is for. Optional —
                if omitted (as it is when called from permission_gated_tool.py),
                it's resolved via get_current_user_id_callback.

        Returns:
            True if user has pre-approved permission or grants instant permission.

        Raises:
            ConfirmationRejected if user denies permission or times out.
        """
        resolved_user_id = self._resolve_user_id(user_id)

        tier_upper = str(tier).upper().strip()
        clean_action = str(action_description).rstrip(".")
        self._current_tiers[resolved_user_id] = tier_upper

        # Check if user has pre-approved permission
        if self.permission_check_callback and self.permission_check_callback(tier_upper):
            logger.info(f"User {resolved_user_id} has pre-approved {tier_upper} permission")
            return True

        # User doesn't have pre-approved permission, request instant permission
        logger.info(
            f"Requesting instant {tier_upper} permission from user "
            f"{resolved_user_id} for: {clean_action}"
        )

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

        # Send confirmation request. is_new_request=True: this is the ONE
        # message that should mark the user as "awaiting confirmation".
        self.send_message_callback(message, True)

        # Wait for user response
        event = asyncio.Event()
        self._pending_confirmations[resolved_user_id] = event
        self._confirmation_results[resolved_user_id] = False

        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Follow-up status notice — is_new_request=False, must NOT
            # re-arm "awaiting confirmation" for this user.
            self.send_message_callback("❌ Confirmation timed out.", False)
            raise ConfirmationTimeout()
        finally:
            # Always clean up this user's pending state, whether we got a
            # reply, timed out, or raised for some other reason.
            self._pending_confirmations.pop(resolved_user_id, None)

        if self._confirmation_results.pop(resolved_user_id, False):
            # Follow-up status notice — is_new_request=False.
            self.send_message_callback("✅ Action confirmed.", False)
            return True
        else:
            self._confirmation_results.pop(resolved_user_id, None)
            # Follow-up status notice — is_new_request=False.
            self.send_message_callback("❌ Action rejected by user.", False)
            raise ConfirmationRejected()

    def handle_confirmation_response(self, response: str, user_id: Optional[int] = None) -> None:
        """
        Handle a user's confirmation response from Telegram.

        Args:
            response: User's text response.
            user_id: Telegram user ID the response came from. Optional —
                resolved via get_current_user_id_callback if omitted, though
                callers that know the user_id (like telegram_main.py) should
                always pass it explicitly.
        """
        resolved_user_id = self._resolve_user_id(user_id)

        event = self._pending_confirmations.get(resolved_user_id)
        if event is None:
            # Nothing is currently waiting for this user (already timed
            # out, already resolved, or never asked). Caller is expected
            # to check is_awaiting_confirmation() first and avoid retrying
            # the original action in this case.
            return

        response_lower = response.strip().lower()
        tier = self._current_tiers.get(resolved_user_id, "MEDIUM")

        if tier == "HIGH":
            self._confirmation_results[resolved_user_id] = response_lower in {"yes", "confirm"}
        else:
            self._confirmation_results[resolved_user_id] = response_lower in {
                "y",
                "yes",
                "confirm",
                "ok",
            }

        event.set()