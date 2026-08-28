import asyncio
import logging

from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)


logger = logging.getLogger("hasini.voice.response_controller")


class ResponseController:
    """
    Owns the currently active agent/TTS response.

    Responsible for:
        - tracking agent task
        - tracking TTS tasks
        - cancelling response generation
        - cancelling audio playback
        - returning the voice system to LISTENING
    """

    def __init__(
        self,
        audio_queue,
        audio_player,
        state_machine: VoiceStateMachine,
    ):
        self.audio_queue = audio_queue
        self.audio_player = audio_player
        self.state_machine = state_machine

        self.agent_task: asyncio.Task | None = None
        self.tts_tasks: list[asyncio.Task] = []

    async def cancel_response(self) -> None:
        """Cancel the active agent/TTS response and prepare for new input."""
        logger.info("Cancelling current response...")

        # ------------------------------------------
        # Cancel agent generation
        # ------------------------------------------
        agent_task = self.agent_task

        if agent_task is not None and not agent_task.done():
            logger.debug("Cancelling active agent task.")
            agent_task.cancel()

            try:
                await agent_task
            except asyncio.CancelledError:
                logger.debug("Agent task cancelled successfully.")
            except Exception:
                logger.exception(
                    "Error while cancelling agent task."
                )

        self.agent_task = None

        # ------------------------------------------
        # Cancel TTS synthesis
        # ------------------------------------------
        active_tts_tasks = list(self.tts_tasks)
        self.tts_tasks.clear()

        for task in active_tts_tasks:
            if task.done():
                continue

            logger.debug("Cancelling active TTS task.")
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                logger.debug("TTS task cancelled successfully.")
            except Exception:
                logger.exception(
                    "Error while cancelling TTS task."
                )

        # ------------------------------------------
        # Stop current playback
        # ------------------------------------------
        try:
            await self.audio_player.cancel()
        except asyncio.CancelledError:
            logger.debug("Audio playback cancellation was cancelled.")
            raise
        except Exception:
            logger.exception(
                "Failed to cancel current audio playback."
            )

        # ------------------------------------------
        # Clear queued audio
        # ------------------------------------------
        try:
            await self.audio_queue.cancel()
        except asyncio.CancelledError:
            logger.debug("Audio queue cancellation was cancelled.")
            raise
        except Exception:
            logger.exception(
                "Failed to clear queued audio."
            )

        # ------------------------------------------
        # State transition
        # ------------------------------------------
        try:
            if self.state_machine.state == VoiceState.USER_INTERRUPT:
                self.state_machine.handle_event(
                    VoiceEvent.TTS_CANCELLED
                )

            if self.state_machine.state == VoiceState.STOP_TTS:
                self.state_machine.handle_event(
                    VoiceEvent.LISTENING_STARTED
                )

        except Exception:
            logger.exception(
                "Failed to update voice state after response cancellation."
            )
            raise

        logger.info("Ready for new command.")
