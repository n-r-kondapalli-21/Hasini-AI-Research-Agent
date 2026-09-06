"""
Hasini Voice Assistant JARVIS HUD UI Bridge.

Launches the local HTTP & WebSocket server and connects VoiceStateMachine,
CommandListener, and AgentController to the JARVIS Arc Reactor visualizer HUD.
"""

from __future__ import annotations

import asyncio
import subprocess
import logging
from pathlib import Path
from typing import Optional

from .state_machine import VoiceState, VoiceStateMachine
from .ui_server import UIServer

logger = logging.getLogger("hasini.voice.ui")


def _map_voice_state(state: VoiceState) -> str:
    """Map VoiceState enum to HUD state string."""
    if state in (VoiceState.LISTENING, VoiceState.WAKE_DETECTED):
        return "LISTENING"
    if state in (
        VoiceState.TRANSCRIBING,
        VoiceState.THINKING,
        VoiceState.TOOL_EXECUTION,
        VoiceState.CONFIRM_WAIT,
        VoiceState.RESPONDING,
    ):
        return "PROCESSING"
    if state == VoiceState.SPEAKING:
        return "SPEAKING"
    return "IDLE"


class VoiceUI:
    """
    JARVIS Visualizer HUD UI Bridge.

    Manages the local UIServer and updates state and text data without blocking
    the voice processing pipeline.
    """

    def __init__(
        self,
        state_machine: Optional[VoiceStateMachine] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        auto_open_browser: bool = True,
    ):
        self.state_machine = state_machine
        self.server = UIServer(host=host, port=port)
        self.auto_open_browser = auto_open_browser
        self.voice_state: VoiceState = (
            state_machine.state if state_machine else VoiceState.IDLE
        )
        self._running: bool = False

        if self.state_machine:
            self.state_machine.add_listener(self._on_state_change)

    def _on_state_change(self, old_state: VoiceState, new_state: VoiceState) -> None:
        """Handle state machine transition callbacks."""
        self.voice_state = new_state
        hud_state = _map_voice_state(new_state)

        # Clear error on new listening session
        if new_state in (VoiceState.WAKE_DETECTED, VoiceState.LISTENING) and old_state == VoiceState.IDLE:
            self.server.update_state(state=hud_state, error=None)
        else:
            self.server.update_state(state=hud_state)

    def set_user_command(self, command: str) -> None:
        """Update user command text on HUD."""
        self.server.update_state(user_command=command.strip())

    def set_agent_response(self, text: str) -> None:
        """Set Hasini agent response text on HUD."""
        self.server.update_state(agent_response=text)

    def append_agent_response(self, chunk: str) -> None:
        """Stream response chunk/token to HUD."""
        self.server.send_token(chunk)

    def clear_agent_response(self) -> None:
        """Clear current agent response text on HUD."""
        self.server.update_state(agent_response="")

    def set_error(self, message: str) -> None:
        """Set error state and alert message on HUD."""
        self.server.update_state(state="ERROR", error=message)

    async def start(self) -> None:
        """Start the JARVIS HUD server and launch Chrome."""
        if self._running:
            return
        self._running = True

        await self.server.start()

        url = f"http://{self.server.host}:{self.server.port}"
        logger.info("JARVIS Visualizer HUD UI ready at %s", url)

        if self.auto_open_browser:
            try:
                chrome_paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    rf"{Path.home()}\AppData\Local\Google\Chrome\Application\chrome.exe",
                ]

                chrome_path = next(
                    (path for path in chrome_paths if Path(path).exists()),
                    None,
                )

                if chrome_path:
                    subprocess.Popen(
                        [chrome_path, url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    logger.info("Opened JARVIS HUD GUI in Google Chrome.")
                else:
                    logger.warning("Google Chrome executable was not found.")

            except Exception as e:
                logger.warning("Failed to open Chrome automatically: %s", e)

    async def stop(self) -> None:
        """Stop the JARVIS HUD UI server."""
        if not self._running:
            return
        self._running = False

        if self.state_machine:
            self.state_machine.remove_listener(self._on_state_change)

        await self.server.stop()
        logger.info("JARVIS HUD UI stopped.")
