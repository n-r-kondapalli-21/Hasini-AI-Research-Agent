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

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger("hasini.text_confirmation")
_console = Console()


class TextConfirmationManager:
    """
    Terminal/text-based confirmation manager for permission-gated tools in text mode.

    Prompts the user in the terminal when a tool action requires permission confirmation.
    """

    def __init__(self, timeout_seconds: float = 15.0, console: Console | None = None):
        """
        Initialize the text confirmation manager.

        Args:
            timeout_seconds: How long to wait for user input before timing out.
            console: Rich Console instance.
        """
        self.timeout_seconds = timeout_seconds
        self.console = console or _console
        # Injected by main.py to point at the active Rich spinner before each
        # agent call. We stop it here before calling input() so that Rich's
        # Live display doesn't hold the terminal in a mode that swallows stdin.
        self._active_status = None

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

        if tier_upper == "HIGH":
            panel_content = (
                f"[bold white]Action:[/bold white] [bright_red]{clean_action}[/bright_red]\n\n"
                f"[bold yellow]⚠️  This action has significant impact. Type 'yes' or 'confirm' to ALLOW.[/bold yellow]"
            )
            panel = Panel(
                panel_content,
                title="🔥 [bold white on red] HIGH RISK PERMISSION REQUIRED [/bold white on red]",
                border_style="bold red",
                expand=False,
            )
        else:
            panel_content = (
                f"[bold white]Action:[/bold white] [yellow]{clean_action}[/yellow]\n\n"
                f"[bold cyan]Allow this action? (y/n)[/bold cyan]"
            )
            panel = Panel(
                panel_content,
                title="🔒 [bold black on yellow] PERMISSION REQUIRED [/bold black on yellow]",
                border_style="yellow",
                expand=False,
            )

        # ── CRITICAL FIX ────────────────────────────────────────────────────
        # Stop the Rich spinner/status *before* printing the panel and calling
        # input(). When a Rich Live display (console.status) is active it holds
        # the terminal in a mode that intercepts raw stdin — the user's y/n
        # keystrokes are swallowed and never reach input(). main.py keeps
        # _active_status pointing to the current spinner; we stop it here and
        # clear the reference so the finally block in main.py is a safe no-op.
        if self._active_status is not None:
            try:
                self._active_status.stop()
            except Exception:
                logger.debug("Failed to stop active status before confirmation prompt.", exc_info=True)
            self._active_status = None
        # ────────────────────────────────────────────────────────────────────

        self.console.print()
        self.console.print(panel)

        loop = asyncio.get_running_loop()
        try:
            user_resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: input("Confirmation > ").strip().lower(),
                ),
                timeout=self.timeout_seconds,
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[bold red]❌ Confirmation cancelled by user.[/bold red]")
            raise ConfirmationRejected()
        except asyncio.TimeoutError:
            self.console.print("\n[bold red]❌ Confirmation timed out.[/bold red]")
            raise ConfirmationTimeout()

        if tier_upper == "HIGH":
            if user_resp in {"yes", "confirm"}:
                self.console.print("[bold green]✅ Action confirmed.[/bold green]")
                return True
            else:
                self.console.print("[bold red]❌ Action rejected by user.[/bold red]")
                raise ConfirmationRejected()
        else:
            if user_resp in {"y", "yes", "confirm", "ok"}:
                self.console.print("[bold green]✅ Action confirmed.[/bold green]")
                return True
            else:
                self.console.print("[bold red]❌ Action rejected by user.[/bold red]")
                raise ConfirmationRejected()
