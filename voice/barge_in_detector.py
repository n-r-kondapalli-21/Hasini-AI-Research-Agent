import asyncio
import logging
import time

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


logger = logging.getLogger("hasini.voice.barge_in")


class BargeInDetector:
    """
    Conservative speech detector used only while Hasini is speaking.

    Pipeline:

        microphone frame
              ↓
          RMS gate
              ↓
       Silero probability
              ↓
       threshold check
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
        self.task: asyncio.Task | None = None

        # Detection state
        self.speech_frames = 0
        self.interrupt_triggered = False

        # Speaking timing
        self.speaking_started_at: float | None = None

    async def start(self) -> None:
        """Start the barge-in detector."""
        if self.running:
            logger.debug("Barge-in detector is already running.")
            return

        self.running = True

        try:
            self.task = asyncio.create_task(
                self._listen_loop(),
                name="barge-in-detector",
            )
        except Exception:
            self.running = False
            logger.exception("Failed to create barge-in detector task.")
            raise

        logger.info("Barge-in detector started.")

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        """Calculate RMS microphone energy."""
        if frame is None or len(frame) == 0:
            return 0.0

        try:
            frame = frame.astype(np.float32, copy=False)

            return float(
                np.sqrt(
                    np.mean(
                        np.square(frame)
                    )
                )
            )
        except Exception:
            logger.exception("Failed to calculate microphone RMS.")
            return 0.0

    def _reset_detection(self) -> None:
        """Reset sustained speech detection."""
        self.speech_frames = 0

    def _reset_session_state(self) -> None:
        """Reset all state associated with the current speaking session."""
        self._reset_detection()
        self.interrupt_triggered = False
        self.speaking_started_at = None

    async def _listen_loop(self) -> None:
        """Continuously monitor microphone audio for user interruption."""
        try:
            while self.running:
                frame = await self.queue.get()

                # --------------------------------------------------
                # Only monitor microphone during SPEAKING.
                # --------------------------------------------------
                if self.state_machine.state != VoiceState.SPEAKING:
                    self._reset_session_state()
                    continue

                # --------------------------------------------------
                # Start timing the current speaking session.
                # --------------------------------------------------
                if self.speaking_started_at is None:
                    self.speaking_started_at = time.monotonic()
                    self._reset_detection()
                    continue

                # --------------------------------------------------
                # Grace period after TTS starts.
                # --------------------------------------------------
                elapsed = (
                    time.monotonic()
                    - self.speaking_started_at
                )

                grace_period_seconds = (
                    BARGE_IN_GRACE_PERIOD_MS / 1000.0
                )

                if elapsed < grace_period_seconds:
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
                    probability = self.vad.process_probability(frame)

                except AttributeError:
                    logger.error(
                        "Barge-in VAD provider does not expose "
                        "process_probability(). "
                        "Add process_probability() to the Silero "
                        "VAD provider."
                    )

                    self.running = False
                    break

                except Exception:
                    logger.exception(
                        "Barge-in VAD probability processing failed."
                    )
                    self._reset_detection()
                    continue

                # --------------------------------------------------
                # Per-frame debug information.
                #
                # DEBUG keeps normal terminal output clean.
                # Enable DEBUG logging when tuning the detector.
                # --------------------------------------------------
                logger.debug(
                    "Barge VAD | prob=%.3f | rms=%.4f | frames=%d",
                    probability,
                    rms,
                    self.speech_frames,
                )

                # --------------------------------------------------
                # Speech probability threshold.
                # --------------------------------------------------
                if probability >= BARGE_IN_VAD_THRESHOLD:
                    self.speech_frames += 1
                else:
                    # Any weak frame breaks sustained speech.
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

                    logger.info(
                        "BARGE-IN confirmed: user speech detected "
                        "(prob=%.3f, rms=%.4f, frames=%d).",
                        probability,
                        rms,
                        self.speech_frames,
                    )

                    # --------------------------------------------------
                    # SPEAKING → USER_INTERRUPT
                    # --------------------------------------------------
                    try:
                        self.state_machine.handle_event(
                            VoiceEvent.USER_SPEECH
                        )
                    except Exception:
                        logger.exception(
                            "Failed to transition state machine "
                            "after barge-in detection."
                        )
                        self._reset_detection()
                        continue

                    # --------------------------------------------------
                    # Cancel active response.
                    # --------------------------------------------------
                    try:
                        asyncio.create_task(
                            self.agent_controller.cancel_response(),
                            name="cancel-agent-response",
                        )
                    except Exception:
                        logger.exception(
                            "Failed to schedule agent response cancellation."
                        )

                    # --------------------------------------------------
                    # Reset VAD state.
                    # --------------------------------------------------
                    try:
                        self.vad.reset()
                    except Exception:
                        logger.exception(
                            "Failed to reset barge-in VAD."
                        )

                    self._reset_detection()

        except asyncio.CancelledError:
            logger.debug("Barge-in detector task cancelled.")
            raise

        except Exception:
            logger.exception("Unexpected error in barge-in detector loop.")

        finally:
            logger.debug("Barge-in detector listen loop exited.")

    async def stop(self) -> None:
        """Stop the barge-in detector and clean up its task."""
        if not self.running and self.task is None:
            logger.debug("Barge-in detector is already stopped.")
            return

        self.running = False

        task = self.task
        self.task = None

        if task is not None:
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Barge-in detector task cancelled successfully.")
            except Exception:
                logger.exception(
                    "Error while stopping barge-in detector task."
                )

        self._reset_session_state()

        logger.info("Barge-in detector stopped.")
