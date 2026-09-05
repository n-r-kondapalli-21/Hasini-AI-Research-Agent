"""
Hasini Telegram interface entrypoint.

Compatible with the current main.py and voice_main.py interfaces:

    agent, registry = await create_research_agent()

The Telegram layer uses the same research agent, Agent_stream, and
ConversationMemory as the terminal main.py while providing a Telegram bot interface.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from agent_runtime import create_research_agent, Agent_stream
from services.conversation_memory import ConversationMemory
from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT, TELEGRAM_BOT_TOKEN
from telegram_interface.telegram_confirmation import TelegramConfirmationManager
from typing import Iterable


logger = logging.getLogger("hasini.telegram_main")


def _configure_logging() -> None:
    """Configure application-wide logging before startup work begins."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Keep noisy third-party loggers from overwhelming the application logs.
    for logger_name in (
        "httpx",
        "httpcore",
        "urllib3",
        "telegram",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger.info("Logging initialized.")


class HasiniTelegramBot:
    """
    Telegram bot interface for Hasini AI Research Agent.

    This class manages the Telegram bot lifecycle and connects Telegram messages
    to the existing agent runtime.
    """

    def __init__(self):
        self.application: Optional[Application] = None
        self.agent = None
        self.registry = None
        self.memory = None
        self.confirmation_manager = None
        self._user_memories = {}  # Dictionary to store per-user conversation memories
        self._current_update = None  # Store current update for sending messages during agent processing
        self._current_user_id = None  # Store current user ID for confirmation
        self._user_permissions = {}  # Dictionary to store per-user permission pre-approvals
        self._pending_actions = {}  # Dictionary to store pending actions for retry
        self._awaiting_confirmation = {}  # Dictionary to track users awaiting confirmation
        self._last_user_message = {}  # Dictionary to store last user message for retry

    async def initialize(self) -> None:
        """Initialize the Telegram bot and research agent."""
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set in environment variables. "
                "Please add it to your .env file."
            )

        logger.info("Initializing Telegram bot...")

        # Create Telegram confirmation manager with permission check callback
        self.confirmation_manager = TelegramConfirmationManager(
            send_message_callback=self._send_admin_message,
            permission_check_callback=lambda tier: self._has_user_permission(self._current_user_id, tier) if self._current_user_id else False,
            set_pending_action_callback=self._set_pending_action,
            timeout_seconds=60.0,  # Increased timeout for Telegram users
        )

        # Create research agent
        logger.info("Creating research agent for Telegram mode...")
        self.agent, self.registry = await create_research_agent(
            confirmation_manager=self.confirmation_manager,
            mode="text",  # Use text mode for Telegram
        )

        logger.info("Research agent created successfully.")

        # Create Telegram application
        self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Register handlers
        self.application.add_handler(CommandHandler("start", self._handle_start))
        self.application.add_handler(CommandHandler("help", self._handle_help))
        self.application.add_handler(
            CommandHandler("clear", self._handle_clear)
        )
        self.application.add_handler(
            CommandHandler("tools", self._handle_tools)
        )
        self.application.add_handler(
            CommandHandler("tool", self._handle_tool)
        )
        self.application.add_handler(
            CommandHandler("grant", self._handle_grant)
        )
        self.application.add_handler(
            CommandHandler("revoke", self._handle_revoke)
        )
        self.application.add_handler(
            CommandHandler("permissions", self._handle_permissions)
        )
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        logger.info("Telegram bot initialized successfully.")

        # Initialize the application
        await self.application.initialize()
        await self.application.start()

    def _get_user_memory(self, user_id: int) -> ConversationMemory:
        """Get or create conversation memory for a specific user."""
        if user_id not in self._user_memories:
            self._user_memories[user_id] = ConversationMemory(
                history_limit=MEMORY_HISTORY_LIMIT,
                enabled=MEMORY_ENABLED,
            )
        return self._user_memories[user_id]

    def _grant_user_permission(self, user_id: int, tier: str) -> None:
        """Grant permission pre-approval for a specific tier to a user."""
        if user_id not in self._user_permissions:
            self._user_permissions[user_id] = set()
        self._user_permissions[user_id].add(tier.upper())
        logger.info(f"Granted {tier.upper()} permission pre-approval to user {user_id}")

    def _revoke_user_permission(self, user_id: int, tier: str) -> None:
        """Revoke permission pre-approval for a specific tier from a user."""
        if user_id in self._user_permissions:
            self._user_permissions[user_id].discard(tier.upper())
            logger.info(f"Revoked {tier.upper()} permission pre-approval from user {user_id}")

    def _has_user_permission(self, user_id: int, tier: str) -> bool:
        """Check if a user has pre-approved permission for a specific tier."""
        if user_id not in self._user_permissions:
            return False
        return tier.upper() in self._user_permissions[user_id]

    def _set_pending_action(self, tier: str, action_description: str) -> None:
        """Store the pending action for potential retry after permission is granted."""
        if self._current_user_id:
            self._pending_actions[self._current_user_id] = {
                'tier': tier,
                'action': action_description,
                'user_message': self._last_user_message.get(self._current_user_id, ''),
                'timestamp': asyncio.get_event_loop().time()
            }
            logger.info(f"Stored pending action for user {self._current_user_id}: {action_description}")

    def _get_pending_action(self, user_id: int) -> Optional[dict]:
        """Get and clear the pending action for a user."""
        return self._pending_actions.pop(user_id, None)

    def _set_awaiting_confirmation(self, user_id: int) -> None:
        """Mark a user as awaiting a confirmation response."""
        self._awaiting_confirmation[user_id] = True

    def _clear_awaiting_confirmation(self, user_id: int) -> None:
        """Clear the awaiting confirmation state for a user."""
        self._awaiting_confirmation.pop(user_id, None)

    def _is_awaiting_confirmation(self, user_id: int) -> bool:
        """Check if a user is awaiting a confirmation response."""
        return self._awaiting_confirmation.get(user_id, False)

    def _format_tools_list(self, tools: Iterable, title: str = "Available Tools") -> str:
        """Format a list of registered tools for Telegram display."""
        try:
            tools = list(tools)
            message = f"📋 {title}\n"
            message += "-" * 30 + "\n"

            if not tools:
                message += "No tools found."
            else:
                for tool in tools:
                    tool_name = getattr(tool, "name", "<unnamed tool>")
                    message += f"• {tool_name}\n"

            message += "-" * 30
            return message

        except Exception:
            logger.exception("Failed to format tools list.")
            return "❌ Failed to display tools."

    def _format_categories(self, registry) -> str:
        """Format all currently registered tool categories for Telegram display."""
        try:
            categories = registry.categories()
            message = "📂 Available Tool Categories\n"
            message += "-" * 30 + "\n"

            if not categories:
                message += "No tool categories found."
            else:
                for category in categories:
                    tools = registry.get_tools(category)
                    message += f"• {category} ({len(tools)} tools)\n"

            message += "-" * 30
            return message

        except Exception:
            logger.exception("Failed to format tool categories.")
            return "❌ Failed to display tool categories."

    def _format_tool_details(self, tool, tool_name: str) -> str:
        """Format detailed information about a specific tool for Telegram display."""
        try:
            message = "🔧 Tool Details\n"
            message += "-" * 30 + "\n"
            message += f"Name: {getattr(tool, 'name', tool_name)}\n"

            description = getattr(tool, "description", None)
            if description:
                message += f"\nDescription:\n{description}\n"

            message += "-" * 30
            return message

        except Exception:
            logger.exception("Failed to format tool details.")
            return "❌ Failed to display tool details."

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /start command."""
        if update.message:
            user_name = update.effective_user.first_name or "there"
            welcome_message = (
                f"Hello {user_name}! 👋\n\n"
                "I'm Hasini, your AI Research Agent. I can help you with:\n"
                "- Web searches\n"
                "- Weather information\n"
                "- Calculations\n"
                "- And more via integrated tools\n\n"
                "Commands:\n"
                "/help - Show this help message\n"
                "/clear - Clear conversation history\n"
                "/tools - List available tool categories\n"
                "/tools <category> - List tools in a category\n"
                "/tool <tool_name> - Inspect a specific tool\n"
                "/grant <tier> - Pre-approve permissions (low/medium/high)\n"
                "/revoke <tier> - Revoke permission pre-approval\n"
                "/permissions - Show your current permissions\n\n"
                "Just send me a message to get started!"
            )
            await update.message.reply_text(welcome_message)

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /help command."""
        if update.message:
            help_message = (
                "🤖 Hasini AI Research Agent Help\n\n"
                "Available commands:\n"
                "/start - Start the bot\n"
                "/help - Show this help message\n"
                "/clear - Clear conversation history\n"
                "/tools - List available tool categories\n"
                "/tools <category> - List tools in a category\n"
                "/tool <tool_name> - Inspect a specific tool\n"
                "/grant <tier> - Pre-approve permissions (low/medium/high)\n"
                "/revoke <tier> - Revoke permission pre-approval\n"
                "/permissions - Show your current permissions\n\n"
                "I can help you with:\n"
                "- Web searches\n"
                "- Weather information\n"
                "- Calculations\n"
                "- And more via integrated tools\n\n"
                "Just send me a message to get started!"
            )
            await update.message.reply_text(help_message)

    async def _handle_clear(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /clear command."""
        if update.message:
            user_id = update.effective_user.id
            if user_id in self._user_memories:
                self._user_memories[user_id].clear()
                await update.message.reply_text("Conversation history cleared. 🗑️")
            else:
                await update.message.reply_text("No conversation history to clear.")

    async def _handle_tools(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /tools command to list tool categories or tools in a category."""
        if not update.message:
            return

        try:
            # Get the category argument if provided
            if context.args and len(context.args) > 0:
                category = context.args[0].strip()
                tools = self.registry.get_tools(category)

                if tools:
                    message = self._format_tools_list(
                        tools, f"{category.title()} Tools"
                    )
                else:
                    message = f"No tools found for category '{category}'.\n\n"
                    message += self._format_categories(self.registry)
            else:
                # Show all categories
                message = self._format_categories(self.registry)

            await update.message.reply_text(message)

        except Exception as e:
            logger.exception("Error handling /tools command")
            await update.message.reply_text("Failed to retrieve tools information.")

    async def _handle_tool(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /tool command to inspect a specific tool."""
        if not update.message:
            return

        try:
            # Get the tool name argument
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    "Please provide a tool name. Example: /tool openalgo_get_quote"
                )
                return

            tool_name = context.args[0].strip()

            if not tool_name:
                await update.message.reply_text("Tool name cannot be empty.")
                return

            tool = self.registry.get_tool(tool_name)

            if tool:
                message = self._format_tool_details(tool, tool_name)
            else:
                message = f"Tool '{tool_name}' not found."

            await update.message.reply_text(message)

        except Exception as e:
            logger.exception("Error handling /tool command")
            await update.message.reply_text("Failed to retrieve tool information.")

    async def _handle_grant(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /grant command to pre-approve permissions."""
        if not update.message:
            return

        user_id = update.effective_user.id

        try:
            # Get the tier argument
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    "Please provide a permission tier. Example: /grant medium\n"
                    "Available tiers: low, medium, high"
                )
                return

            tier = context.args[0].strip().upper()

            if tier not in {"LOW", "MEDIUM", "HIGH"}:
                await update.message.reply_text(
                    "Invalid permission tier. Available tiers: low, medium, high"
                )
                return

            self._grant_user_permission(user_id, tier)
            await update.message.reply_text(f"✅ Granted {tier} permission pre-approval.")

        except Exception as e:
            logger.exception("Error handling /grant command")
            await update.message.reply_text("Failed to grant permission.")

    async def _handle_revoke(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /revoke command to revoke permission pre-approvals."""
        if not update.message:
            return

        user_id = update.effective_user.id

        try:
            # Get the tier argument
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    "Please provide a permission tier. Example: /revoke medium\n"
                    "Available tiers: low, medium, high"
                )
                return

            tier = context.args[0].strip().upper()

            if tier not in {"LOW", "MEDIUM", "HIGH"}:
                await update.message.reply_text(
                    "Invalid permission tier. Available tiers: low, medium, high"
                )
                return

            self._revoke_user_permission(user_id, tier)
            await update.message.reply_text(f"✅ Revoked {tier} permission pre-approval.")

        except Exception as e:
            logger.exception("Error handling /revoke command")
            await update.message.reply_text("Failed to revoke permission.")

    async def _handle_permissions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /permissions command to show current permission pre-approvals."""
        if not update.message:
            return

        user_id = update.effective_user.id

        try:
            if user_id not in self._user_permissions or not self._user_permissions[user_id]:
                message = "You have no pre-approved permissions."
            else:
                permissions = sorted(self._user_permissions[user_id])
                message = "Your pre-approved permissions:\n"
                for perm in permissions:
                    message += f"• {perm}\n"

            await update.message.reply_text(message)

        except Exception as e:
            logger.exception("Error handling /permissions command")
            await update.message.reply_text("Failed to retrieve permissions.")

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle text messages from users."""
        if not update.message or not update.message.text:
            return

        user_id = update.effective_user.id
        user_message = update.message.text.strip()

        if not user_message:
            return

        logger.info(f"Received message from user {user_id}: {user_message}")

        # Check if user is awaiting a confirmation response
        if self._is_awaiting_confirmation(user_id):
            self._clear_awaiting_confirmation(user_id)
            # Handle confirmation response
            self.confirmation_manager.handle_confirmation_response(user_message)
            await update.message.reply_text("Response recorded. Retrying action...")
            
            # Retry the original action
            pending_action = self._get_pending_action(user_id)
            if pending_action and pending_action['user_message']:
                # Re-process the original request
                memory = self._get_user_memory(user_id)
                await update.message.chat.send_action("typing")
                
                try:
                    # Store current update and user ID for sending messages during agent processing
                    self._current_update = update
                    self._current_user_id = user_id

                    response_received = False
                    full_response = ""

                    async for chunk in Agent_stream(
                        pending_action['user_message'],  # Use the original user message
                        self.agent,
                        memory,
                    ):
                        response_received = True
                        full_response += chunk

                    # Clear the current update and user ID
                    self._current_update = None
                    self._current_user_id = None

                    if response_received and full_response:
                        await update.message.reply_text(full_response)
                        logger.info(f"Retried action succeeded for user {user_id}")
                    else:
                        await update.message.reply_text(
                            "I didn't generate a response on retry. Please try again."
                        )
                        logger.warning(f"No response generated on retry for user {user_id}")

                except Exception as e:
                    logger.exception(f"Error on retry for user {user_id}")
                    await update.message.reply_text(
                        "Sorry, I encountered an error on retry. Please try again."
                    )
            else:
                await update.message.reply_text("No pending action to retry. Please send your command again.")
            return

        # Store the user message for potential retry
        self._last_user_message[user_id] = user_message

        # Get user-specific conversation memory
        memory = self._get_user_memory(user_id)

        # Send typing indicator
        await update.message.chat.send_action("typing")

        # Process message through agent
        response_received = False
        full_response = ""

        try:
            # Store current update and user ID for sending messages during agent processing
            self._current_update = update
            self._current_user_id = user_id

            async for chunk in Agent_stream(
                user_message,
                self.agent,
                memory,
            ):
                response_received = True
                full_response += chunk

            # Clear the current update and user ID
            self._current_update = None
            self._current_user_id = None

            if response_received and full_response:
                # Send the complete response
                await update.message.reply_text(full_response)
                logger.info(f"Sent response to user {user_id}")
            else:
                await update.message.reply_text(
                    "I didn't generate a response. Please try again."
                )
                logger.warning(f"No response generated for user {user_id}")

        except asyncio.CancelledError:
            logger.info(f"Agent response cancelled for user {user_id}")
            await update.message.reply_text("Response was cancelled. Please try again.")

        except Exception as e:
            logger.exception(f"Error processing message for user {user_id}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your request. Please try again."
            )

    def _send_admin_message(self, message: str) -> None:
        """
        Send a message to the user for confirmation.

        This is used by the confirmation manager to send confirmation requests.
        The message is sent immediately if there's a current update context.
        """
        if self._current_update and self._current_update.message:
            # Set the user state to awaiting confirmation
            if self._current_user_id:
                self._set_awaiting_confirmation(self._current_user_id)
            
            # Use asyncio to send the message immediately
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the message to be sent
                    asyncio.create_task(
                        self._current_update.message.reply_text(message)
                    )
                else:
                    logger.warning("Event loop not running, cannot send confirmation message")
            except Exception as e:
                logger.exception(f"Failed to send confirmation message: {e}")
        else:
            logger.warning("No current update context, cannot send confirmation message")

    async def run(self) -> None:
        """Run the Telegram bot."""
        if not self.application:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        logger.info("Starting Telegram bot polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

        logger.info("Telegram bot is running. Press Ctrl+C to stop.")

        # Keep the bot running
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            logger.info("Bot shutdown requested.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self.application:
            logger.info("Stopping Telegram bot...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped.")


async def main() -> int:
    """Main entry point for the Telegram bot."""
    _configure_logging()

    logger.info("Starting Hasini Telegram Bot...")

    bot = HasiniTelegramBot()

    shutdown_event = asyncio.Event()

    def _request_shutdown(sig_name: str) -> None:
        logger.info("Received %s, shutting down...", sig_name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                _request_shutdown,
                sig.name,
            )
        except (NotImplementedError, RuntimeError):
            logger.debug(
                "Signal handler for %s is not supported by this event loop.",
                sig,
            )

    try:
        await bot.initialize()

        # Start the bot
        await bot.application.updater.start_polling()

        logger.info("Hasini Telegram Bot started successfully.")
        logger.info("Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1

    except Exception:
        logger.exception("Telegram bot failed to start.")
        return 1

    finally:
        logger.info("Shutting down...")
        if bot.application:
            await bot.application.updater.stop()
        logger.info("Hasini Telegram Bot stopped.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        exit_code = 130

    sys.exit(exit_code)
