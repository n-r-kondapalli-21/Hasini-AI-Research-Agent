"""
Continuously listens for the configured wake word.

Responsibilities:
    - Consume microphone frames.
    - Convert audio to the format expected by the wake-word provider.
    - Detect the wake word.
    - Notify the state machine.

Does NOT:
    - perform STT
    - call the LLM
    - execute tools
    - generate TTS

Enhancements over the original version:
  - `logging` instead of `print`.
  - A single bad chunk (inference error, malformed audio) no longer
    kills the *entire* listener for the rest of the session — the
    original wrapped the whole `_listen_loop` in one try/except, so any
    exception on any chunk silently ended wake-word detection for good
    (the loop simply exited; nothing restarted it). Per-chunk processing
    is now individually guarded, logged, and skipped so the loop keeps
    running.
  - `start()`'s background task now has a "done" callback that logs if
    it ever exits unexpectedly (crashed or finished without being
    cancelled), instead of failing silently with no trace that
    wake-word detection quietly stopped.
  - The fire-and-forget ack-sound playback task is tracked and its
    exceptions are logged instead of vanishing silently (an
    uncaught exception in a task nobody awaits is normally swallowed by
    asyncio entirely).
  - `queue.get()` and `wakeword.score_frame(...)` calls are defensively
    handled so a transient failure doesn't propagate into a full loop
    exit.
  - Ack sound loading logs via `logging` and degrades gracefully;
    unchanged in behavior, just no longer using `print`/bare `except`.
"""

import asyncio
import logging
import os
import random
import wave

import numpy as np

from .audio_broadcaster import AudioBroadcaster
from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)
from .providers.wakeword.base import WakeWordProvider
from .audio_io import play_audio


logger = logging.getLogger("hasini.voice.wakeword_listener")


class WakeWordListener:
    """
    Continuously listens for the configured wake word.
    """

    # 80 ms at 16 kHz.
    # This matches the frame size used successfully during our
    # wake-word calibration.
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
        self.task: asyncio.Task | None = None

        # Track fire-and-forget ack playback tasks so exceptions in
        # them get logged instead of silently disappearing, and so
        # they don't get garbage-collected mid-flight.
        self._background_tasks: set[asyncio.Task] = set()

        self.ack_sounds = []
        self._load_ack_sounds()

    # ============================================================
    # ACK SOUNDS
    # ============================================================

    def _load_ack_sounds(self) -> None:
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        ack_files = [
            ("ack_yes.wav", "Yes?"),
            ("ack_listening.wav", "I'm listening."),
            ("ack_go_ahead.wav", "Go ahead."),
            ("ack_yes_listening.wav", "Yes, I'm listening."),
        ]

        self.ack_sounds = []
        for filename, label in ack_files:
            filepath = os.path.join(assets_dir, filename)
            if not os.path.exists(filepath):
                continue
            try:
                with wave.open(filepath, "rb") as w:
                    rate = w.getframerate()
                    frames = w.readframes(w.getnframes())
                    audio_np = (
                        np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    self.ack_sounds.append((audio_np, rate, label))
            except Exception:
                logger.warning("Could not load ack sound %s", filename, exc_info=True)

        # Fallback to ack_short.wav if voice files are missing.
        if not self.ack_sounds:
            chime_path = os.path.join(assets_dir, "ack_short.wav")
            if os.path.exists(chime_path):
                try:
                    with wave.open(chime_path, "rb") as w:
                        rate = w.getframerate()
                        frames = w.readframes(w.getnframes())
                        audio_np = (
                            np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        self.ack_sounds.append((audio_np, rate, "chime"))
                except Exception:
                    logger.warning(
                        "Could not load fallback ack sound %s", chime_path, exc_info=True
                    )

        if not self.ack_sounds:
            logger.warning(
                "No wake-word acknowledgement sounds available; "
                "detections will proceed silently."
            )

    def _play_random_ack(self) -> None:
        if not self.ack_sounds:
            return

        audio, rate, label = random.choice(self.ack_sounds)
        logger.info('Wake acknowledgement: "%s"', label)

        task = asyncio.create_task(asyncio.to_thread(play_audio, audio, rate))
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task, label=label):
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.warning(
                    "Ack sound playback failed for %r: %s", label, exc, exc_info=exc
                )

        task.add_done_callback(_on_done)

    # ============================================================
    # START
    # ============================================================

    async def start(self) -> None:
        if self.running:
            logger.debug("Wake-word listener is already running.")
            return

        self.running = True
        self.task = asyncio.create_task(self._listen_loop())
        self.task.add_done_callback(self._on_listen_loop_done)

        logger.info("Wake-word listener started.")

    def _on_listen_loop_done(self, task: asyncio.Task) -> None:
        """Surface unexpected exits of the listener loop.

        Without this, a bug that let an exception escape `_listen_loop`
        (or the loop returning early for any other reason) would leave
        `self.running` still True with nobody actually listening, and
        no indication in the logs that wake-word detection had died.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Wake-word listen loop exited unexpectedly: %s", exc, exc_info=exc
            )
        elif self.running:
            logger.warning(
                "Wake-word listen loop exited while still marked as running."
            )

    # ============================================================
    # LISTENING LOOP
    # ============================================================

    async def _listen_loop(self) -> None:
        buffer = np.empty(0, dtype=np.float32)

        while self.running:
            try:
                frame = await self.queue.get()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Failed to read audio frame from queue; retrying", exc_info=True
                )
                continue

            try:
                buffer = np.concatenate((buffer, frame))
            except (TypeError, ValueError):
                logger.warning(
                    "Discarding malformed audio frame from broadcaster", exc_info=True
                )
                continue

            # ------------------------------------------------------
            # Process complete 80 ms chunks.
            # ------------------------------------------------------
            while len(buffer) >= self.WAKEWORD_FRAME_SAMPLES:
                chunk = buffer[: self.WAKEWORD_FRAME_SAMPLES]
                buffer = buffer[self.WAKEWORD_FRAME_SAMPLES :]

                try:
                    self._process_chunk(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A single bad chunk (bad inference, unexpected
                    # provider error, etc.) must not end wake-word
                    # detection for the rest of the session.
                    logger.warning(
                        "Error processing wake-word chunk; skipping", exc_info=True
                    )

        logger.debug("Wake-word listen loop exiting (running=False).")

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Run wake-word detection on a single 80ms chunk."""

        # Wake word is only active while IDLE.
        if self.state_machine.state != VoiceState.IDLE:
            return

        # Convert float32 [-1, 1] to int16 PCM.
        audio = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)

        # Run exactly one wake-word inference.
        score = self.wakeword.score_frame(audio)

        if score < self.wakeword.threshold:
            return

        # Wake word detected.
        logger.info("Hey Jarvis detected! Score: %.3f", score)

        try:
            self.wakeword.reset()
        except Exception:
            # Non-fatal: worst case is a slightly stale internal
            # model state on the next inference, not a crash.
            logger.warning("Failed to reset wake-word model state", exc_info=True)

        self.state_machine.handle_event(VoiceEvent.WAKE_WORD)

        # Non-blocking preloaded voice acknowledgement playback.
        self._play_random_ack()

        # Stop processing additional wake-word chunks until the voice
        # system returns to IDLE (handled by the state check above).

    # ============================================================
    # STOP
    # ============================================================

    async def stop(self) -> None:
        if not self.running:
            logger.debug("Wake-word listener is already stopped.")
            return

        self.running = False

        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "Wake-word listener task raised during shutdown", exc_info=True
                )

        self.task = None

        # Give any in-flight ack playback tasks a moment to finish/log
        # rather than abandoning them; cancel stragglers.
        if self._background_tasks:
            for t in list(self._background_tasks):
                t.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        logger.info("Wake-word listener stopped.")