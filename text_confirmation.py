"""
Text-mode confirmation manager for permission-gated tools.

Provides terminal-based interaction for user confirmation when a
permission-gated tool action requires permission (MEDIUM or HIGH tier).
"""

from __future__ import annotations

import asyncio
import logging

# Import confirmation exceptions from voice.confirmation
# These are used by the agent runtime for proper error handling
from voice.confirmation import ConfirmationRejected, ConfirmationTimeout

logger = logging.getLogger("hasini.text_confirmation")


class TextConfirmationManager:
    """
    Terminal/text-based confirmation manager for permission-gated tools in text mode.

    Prompts the user in the terminal when a tool action requires permission confirmation.
    """

    def __init__(self, timeout_seconds: float = 15.0):
        """
        Initialize the text confirmation manager.
        
        Args:
            timeout_seconds: How long to wait for user input before timing out.
        """
        self.timeout_seconds = timeout_seconds

    async def wait_for_confirmation(
        self,
        tier: str,
        action_description: str,
    ) -> bool:
        """
        Prompt the user in the terminal for permission confirmation.

        Args:
            tier: Permission tier ("MEDIUM" or "HIGH").
            action_description: Description of the action requiring permission.

        Returns:
            True if confirmed by user.

        Raises:
            ConfirmationRejected if rejected by user or input interrupted.
            ConfirmationTimeout if confirmation input times out.
        """
        tier_upper = str(tier).upper().strip()
        clean_action = str(action_description).rstrip(".")

        print("\n" + "=" * 50)
        if tier_upper == "HIGH":
            print("⚠️  [HIGH RISK PERMISSION REQUIRED]")
            print(f"Action: {clean_action}")
            print("Type 'yes' or 'confirm' to ALLOW this high-risk action.")
        else:
            print("🔒 [PERMISSION REQUIRED]")
            print(f"Action: {clean_action}")
            print("Allow this action? (y/n)")
        print("=" * 50)

        loop = asyncio.get_running_loop()
        try:
            user_resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: input("Confirmation: ").strip().lower(),
                ),
                timeout=self.timeout_seconds,
            )
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Confirmation cancelled.")
            raise ConfirmationRejected()
        except asyncio.TimeoutError:
            print("\n❌ Confirmation timed out.")
            raise ConfirmationTimeout()

        if tier_upper == "HIGH":
            if user_resp in {"yes", "confirm"}:
                print("✅ Action confirmed.")
                return True
            else:
                print("❌ Action rejected by user.")
                raise ConfirmationRejected()
        else:
            if user_resp in {"y", "yes", "confirm", "ok"}:
                print("✅ Action confirmed.")
                return True
            else:
                print("❌ Action rejected by user.")
                raise ConfirmationRejected()
