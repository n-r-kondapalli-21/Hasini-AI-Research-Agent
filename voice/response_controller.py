import asyncio

from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)


class ResponseController:
    """
    Owns the currently active agent/TTS response.

    Responsible for:
        - tracking agent task
        - tracking TTS task
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

        self.agent_task = None

        self.tts_tasks = []

    async def cancel_response(self):

        print(
            "\n🛑 Cancelling current response..."
        )

        # ------------------------------------------
        # Cancel agent generation
        # ------------------------------------------

        if (
            self.agent_task is not None
            and not self.agent_task.done()
        ):

            self.agent_task.cancel()

        # ------------------------------------------
        # Cancel TTS synthesis
        # ------------------------------------------

        for task in self.tts_tasks:

            if not task.done():

                task.cancel()

        self.tts_tasks.clear()

        # ------------------------------------------
        # Stop current playback
        # ------------------------------------------

        await self.audio_player.cancel()

        # ------------------------------------------
        # Clear queued audio
        # ------------------------------------------

        await self.audio_queue.cancel()

        # ------------------------------------------
        # State transition
        # ------------------------------------------

        if (
            self.state_machine.state
            == VoiceState.USER_INTERRUPT
        ):

            self.state_machine.handle_event(
                VoiceEvent.TTS_CANCELLED
            )

        if (
            self.state_machine.state
            == VoiceState.STOP_TTS
        ):

            self.state_machine.handle_event(
                VoiceEvent.LISTENING_STARTED
            )

        print(
            "🎤 Ready for new command."
        )