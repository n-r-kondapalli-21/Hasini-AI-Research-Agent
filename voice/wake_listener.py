import asyncio

import numpy as np

from .audio_broadcaster import AudioBroadcaster
from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)
from .providers.wakeword.base import WakeWordProvider


class WakeWordListener:
    """
    Continuously listens for the configured wake word.

    Responsibilities:
        - Consume microphone frames.
        - Convert audio to the format expected by
          the wake-word provider.
        - Detect the wake word.
        - Notify the state machine.

    Does NOT:
        - perform STT
        - call the LLM
        - execute tools
        - generate TTS
    """

    # 80 ms at 16 kHz.
    # This matches the frame size used successfully
    # during our wake-word calibration.
    WAKEWORD_FRAME_SAMPLES = 1280

    def __init__(
        self,
        broadcaster: AudioBroadcaster,
        wakeword: WakeWordProvider,
        state_machine: VoiceStateMachine,
    ):

        self.broadcaster = broadcaster

        self.wakeword = wakeword

        self.state_machine = state_machine

        # Independent queue for this consumer.
        self.queue = broadcaster.subscribe()

        self.running = False

        self.task = None

    # ============================================================
    # START
    # ============================================================

    async def start(self):

        if self.running:
            return

        self.running = True

        self.task = asyncio.create_task(
            self._listen_loop()
        )

        print(
            "👂 Wake-word listener started."
        )

    # ============================================================
    # LISTENING LOOP
    # ============================================================

    async def _listen_loop(self):

        buffer = np.empty(
            0,
            dtype=np.float32,
        )

        try:

            while self.running:

                frame = await self.queue.get()

                # --------------------------------------------------
                # Accumulate microphone frames.
                # --------------------------------------------------

                buffer = np.concatenate(
                    (
                        buffer,
                        frame,
                    )
                )

                # --------------------------------------------------
                # Process complete 80 ms chunks.
                # --------------------------------------------------

                while (
                    len(buffer)
                    >= self.WAKEWORD_FRAME_SAMPLES
                ):

                    chunk = buffer[
                        :self.WAKEWORD_FRAME_SAMPLES
                    ]

                    buffer = buffer[
                        self.WAKEWORD_FRAME_SAMPLES:
                    ]

                    # --------------------------------------------------
                    # Wake word is only active while IDLE.
                    # --------------------------------------------------

                    if (
                        self.state_machine.state
                        != VoiceState.IDLE
                    ):

                        continue

                    # --------------------------------------------------
                    # Convert float32 [-1, 1]
                    # to int16 PCM.
                    # --------------------------------------------------

                    audio = np.clip(
                        chunk * 32767.0,
                        -32768,
                        32767,
                    ).astype(
                        np.int16
                    )

                    # --------------------------------------------------
                    # Run exactly one wake-word inference.
                    # --------------------------------------------------

                    score = (
                        self.wakeword.score_frame(
                            audio
                        )
                    )

                    if (
                        score
                        < self.wakeword.threshold
                    ):

                        continue

                    # --------------------------------------------------
                    # Wake word detected.
                    # --------------------------------------------------

                    print(
                        f"\n🟢 Hey Jarvis detected!"
                        f" Score: {score:.3f}"
                    )

                    self.wakeword.reset()

                    self.state_machine.handle_event(
                        VoiceEvent.WAKE_WORD
                    )

                    # Stop processing additional wake-word
                    # chunks until the voice system returns
                    # to IDLE.

        except asyncio.CancelledError:

            pass

        except Exception as e:

            print(
                f"\n❌ Wake-word listener error: {e}"
            )

    # ============================================================
    # STOP
    # ============================================================

    async def stop(self):

        if not self.running:
            return

        self.running = False

        if self.task is not None:

            self.task.cancel()

            try:

                await self.task

            except asyncio.CancelledError:

                pass

        self.task = None

        print(
            "👂 Wake-word listener stopped."
        )