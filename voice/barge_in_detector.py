import asyncio
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

from .audio_broadcaster import AudioBroadcaster

from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)

from .providers.vad.base import VADProvider

from .config import (
    BARGE_IN_VAD_THRESHOLD,
    BARGE_IN_REQUIRED_SPEECH_FRAMES,
    BARGE_IN_GRACE_PERIOD_MS,
    BARGE_IN_MIN_RMS,
)


class BargeInDetector:
    """
    Conservative speech detector used only while Hasini
    is speaking.

    Normal command VAD and barge-in VAD intentionally have
    different sensitivity requirements.

    Barge-in pipeline:

        microphone frame
              ↓
          RMS gate
              ↓
       Silero probability
              ↓
       0.85 threshold
              ↓
      sustained speech
              ↓
        USER_SPEECH
              ↓
      cancel response
    """

    def __init__(
        self,
        broadcaster: AudioBroadcaster,
        vad: VADProvider,
        state_machine: VoiceStateMachine,
        agent_controller,
    ):

        self.broadcaster = broadcaster

        self.vad = vad

        self.state_machine = state_machine

        self.agent_controller = agent_controller

        self.queue = broadcaster.subscribe()

        self.running = False

        self.task = None

        # --------------------------------------------------
        # Detection state
        # --------------------------------------------------

        self.speech_frames = 0

        self.interrupt_triggered = False

        # --------------------------------------------------
        # Speaking timing
        # --------------------------------------------------

        self.speaking_started_at = None

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
            "👂 Barge-in detector started."
        )

    # ============================================================
    # RMS
    # ============================================================

    @staticmethod
    def _rms(
        frame: np.ndarray,
    ) -> float:
        """
        Calculate RMS microphone energy.
        """

        if frame is None or len(frame) == 0:

            return 0.0

        frame = frame.astype(
            np.float32,
            copy=False,
        )

        return float(
            np.sqrt(
                np.mean(
                    np.square(frame)
                )
            )
        )

    # ============================================================
    # RESET DETECTION
    # ============================================================

    def _reset_detection(self):

        self.speech_frames = 0

    # ============================================================
    # LISTEN LOOP
    # ============================================================

    async def _listen_loop(self):

        try:

            while self.running:

                frame = await self.queue.get()

                # --------------------------------------------------
                # Only monitor microphone during SPEAKING.
                # --------------------------------------------------

                if (
                    self.state_machine.state
                    != VoiceState.SPEAKING
                ):

                    self._reset_detection()

                    self.interrupt_triggered = False

                    self.speaking_started_at = None

                    continue

                # --------------------------------------------------
                # Start timing the current speaking session.
                # --------------------------------------------------

                if self.speaking_started_at is None:

                    self.speaking_started_at = (
                        time.monotonic()
                    )

                    self._reset_detection()

                    continue

                # --------------------------------------------------
                # Grace period after TTS starts.
                # --------------------------------------------------

                elapsed = (
                    time.monotonic()
                    - self.speaking_started_at
                )

                if (
                    elapsed
                    <
                    BARGE_IN_GRACE_PERIOD_MS
                    / 1000.0
                ):

                    self._reset_detection()

                    continue

                # --------------------------------------------------
                # RMS energy gate.
                # --------------------------------------------------

                rms = self._rms(frame)

                if rms < BARGE_IN_MIN_RMS:

                    self._reset_detection()

                    continue

                # --------------------------------------------------
                # Get RAW Silero probability.
                # --------------------------------------------------

                try:

                    probability = (
                        self.vad
                        .process_probability(
                            frame
                        )
                    )

                except AttributeError:

                    print(
                        "\n❌ Barge-in VAD provider "
                        "does not expose "
                        "process_probability()."
                    )

                    print(
                        "Add process_probability() "
                        "to the Silero VAD provider."
                    )

                    self.running = False

                    break

                # --------------------------------------------------
                # Debug output.
                #
                # Uncomment when tuning.
                # --------------------------------------------------

                print(
                    f"Barge VAD | "
                    f"prob={probability:.3f} | "
                    f"rms={rms:.4f} | "
                    f"frames={self.speech_frames}"
                )

                # --------------------------------------------------
                # Speech probability threshold.
                # --------------------------------------------------

                if (
                    probability
                    >= BARGE_IN_VAD_THRESHOLD
                ):

                    self.speech_frames += 1

                else:

                    # Any weak frame breaks the
                    # sustained-speech requirement.
                    self._reset_detection()

                # --------------------------------------------------
                # Confirm sustained speech.
                # --------------------------------------------------

                if (
                    self.speech_frames
                    >= BARGE_IN_REQUIRED_SPEECH_FRAMES
                    and not self.interrupt_triggered
                ):

                    self.interrupt_triggered = True

                    print(
                        "\n🛑 BARGE-IN: "
                        "Confirmed user speech."
                    )

                    print(
                        f"   Probability: "
                        f"{probability:.3f}"
                    )

                    print(
                        f"   RMS: "
                        f"{rms:.4f}"
                    )

                    print(
                        f"   Speech frames: "
                        f"{self.speech_frames}"
                    )

                    # --------------------------------------------------
                    # SPEAKING → USER_INTERRUPT
                    # --------------------------------------------------

                    self.state_machine.handle_event(
                        VoiceEvent.USER_SPEECH
                    )

                    # --------------------------------------------------
                    # Cancel active response.
                    # --------------------------------------------------

                    asyncio.create_task(
                        self.agent_controller
                        .cancel_response()
                    )

                    self.vad.reset()

                    self._reset_detection()

        except asyncio.CancelledError:

            pass

        except Exception as e:

            print(
                f"\n❌ Barge-in detector error: {e}"
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

        self._reset_detection()

        self.interrupt_triggered = False

        self.speaking_started_at = None

        print(
            "👂 Barge-in detector stopped."
        )