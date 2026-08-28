"""
Hasini Voice Command Listener.

Handles microphone input after wake-word detection and after a barge-in
interruption.

Responsibilities:
    - Wait for WAKE_DETECTED.
    - Handle post-barge-in LISTENING.
    - Start command listening.
    - Use VAD to detect speech.
    - Collect command audio.
    - Detect end of speech.
    - Transcribe captured audio.

Does NOT:
    - call the LLM directly
    - execute tools
    - generate TTS
    - manage conversation history
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque

import numpy as np

from .audio_broadcaster import AudioBroadcaster
from .state_machine import VoiceEvent, VoiceState, VoiceStateMachine
from .providers.vad.base import VADProvider
from .providers.stt.base import STTProvider
from .agent_controller import AgentController


logger = logging.getLogger("hasini.voice.command_listener")


def is_stop_command(text: str) -> bool:
    """
    Check whether the complete user utterance is a stop/cancel directive.

    Stop detection intentionally uses exact normalized phrases rather than
    checking whether a stop word appears anywhere in a longer command.

    This prevents commands such as:
        "How do I stop a Python process?"
    from being treated as a request to stop Hasini.
    """
    if not text:
        return False

    clean_text = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    clean_text = re.sub(r"\s+", " ", clean_text)

    stop_phrases = {
        "stop",
        "shut up",
        "quiet",
        "cancel",
        "pause",
        "halt",
        "never mind",
        "nevermind",
        "silence",
        "enough",
        "exit",
        "quit",
        "goodbye",
        "stop talking",
        "be quiet",
    }

    return clean_text in stop_phrases


def is_interruption_filler(text: str) -> bool:
    """
    Check whether user speech is a transient pause/filler after barge-in.
    """
    if not text:
        return True

    clean_text = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    clean_text = re.sub(r"\s+", " ", clean_text)

    fillers = {
        "wait",
        "hold on",
        "hey",
        "listen",
        "um",
        "uh",
        "hasini",
        "jarvis",
    }

    return clean_text in fillers


class CommandListener:
    """
    Handles microphone input after wake-word detection and barge-in.

    The listener keeps microphone frames in a rolling buffer so that speech
    occurring while barge-in detection is being confirmed can be recovered.
    """

    def __init__(
        self,
        broadcaster: AudioBroadcaster,
        vad: VADProvider,
        stt: STTProvider,
        agent_controller: AgentController,
        state_machine: VoiceStateMachine,
    ):
        self.broadcaster = broadcaster
        self.vad = vad
        self.stt = stt
        self.agent_controller = agent_controller
        self.state_machine = state_machine

        # Microphone subscription.
        self.queue = broadcaster.subscribe()

        self.running = False
        self.task: asyncio.Task | None = None

        # Background transcription/agent tasks are tracked so shutdown can
        # cancel them cleanly instead of leaving detached tasks alive.
        self.background_tasks: set[asyncio.Task] = set()

        # Command audio.
        self.audio_buffer: list[np.ndarray] = []

        # Wake-word suppression.
        self.wake_suppression_seconds = 0.6
        self.listen_started_at = 0.0

        # Command session state.
        self.listening_session_active = False

        # This describes the current captured session only.
        # It is cleared before background transcription starts; the captured
        # value is passed explicitly to _transcribe().
        self.interrupted_session = False

        # True after VAD reports speech start.
        self.speech_started = False

        # Barge-in pre-buffer.
        #
        # VAD frame = 32 ms.
        # 40 frames ~= 1.28 seconds of rolling microphone history.
        self.pre_buffer_frames = 40
        self.recent_frames = deque(maxlen=self.pre_buffer_frames)

        # Pre-speech preroll used to preserve the beginning of a command.
        self.preroll_frames = deque(maxlen=15)

    # ============================================================
    # START
    # ============================================================

    async def start(self):
        """Start the command listener task."""
        if self.running:
            return

        self.running = True

        try:
            self.task = asyncio.create_task(
                self._listen_loop(),
                name="command-listener",
            )
        except Exception:
            self.running = False
            logger.exception("Failed to create command listener task.")
            raise

        logger.info("Command listener started.")

    # ============================================================
    # BACKGROUND TASK MANAGEMENT
    # ============================================================

    def _track_background_task(
        self,
        coroutine,
        name: str,
    ) -> asyncio.Task:
        """
        Create and track a background task.

        Tracking prevents transcription/agent tasks from surviving
        CommandListener.stop().
        """
        task = asyncio.create_task(coroutine, name=name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    async def _cancel_background_tasks(self) -> None:
        """Cancel and await all outstanding background tasks."""
        tasks = list(self.background_tasks)

        if not tasks:
            return

        logger.debug(
            "Cancelling %d command background task(s).",
            len(tasks),
        )

        for task in tasks:
            if not task.done():
                task.cancel()

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result,
                asyncio.CancelledError,
            ):
                logger.warning(
                    "Background command task ended with an error: %r",
                    result,
                )

        self.background_tasks.clear()

    # ============================================================
    # MAIN LISTENING LOOP
    # ============================================================

    async def _listen_loop(self):
        """Continuously consume microphone frames and route them by state."""
        try:
            while self.running:
                frame = await self.queue.get()

                if frame is None:
                    logger.debug("Ignoring empty microphone frame.")
                    continue

                # Always maintain the rolling microphone history before
                # state-specific processing.
                try:
                    self.recent_frames.append(frame.copy())
                except AttributeError:
                    logger.warning(
                        "Ignoring microphone frame without copy() support."
                    )
                    continue

                state = self.state_machine.state

                # ==================================================
                # Normal wake-word flow
                # ==================================================

                if (
                    state == VoiceState.WAKE_DETECTED
                    and not self.listening_session_active
                ):
                    await self._start_listening_session(
                        interrupted=False
                    )
                    self._process_frame(frame)
                    continue

                # ==================================================
                # Barge-in flow
                # ==================================================

                if (
                    state == VoiceState.LISTENING
                    and not self.listening_session_active
                ):
                    await self._start_listening_session(
                        interrupted=True
                    )

                    # _start_listening_session() replays the buffered frames,
                    # including the current frame. Do not process it again.
                    continue

                # ==================================================
                # Continue active command session
                # ==================================================

                if (
                    state == VoiceState.LISTENING
                    and self.listening_session_active
                ):
                    self._process_frame(frame)

        except asyncio.CancelledError:
            logger.debug("Command listener loop cancelled.")
            raise

        except Exception:
            logger.exception(
                "Command listener loop failed unexpectedly."
            )

            self.running = False
            self.listening_session_active = False
            self.interrupted_session = False
            self.audio_buffer.clear()
            self.recent_frames.clear()
            self.preroll_frames.clear()
            self.state_machine.reset()

    # ============================================================
    # START LISTENING SESSION
    # ============================================================

    async def _start_listening_session(
        self,
        interrupted: bool = False,
    ):
        """Initialize a new command capture session."""
        try:
            self.vad.reset()
        except Exception:
            logger.exception("Failed to reset command VAD.")
            raise

        self.audio_buffer = []
        self.preroll_frames.clear()
        self.speech_started = False

        self.listen_started_at = time.monotonic()

        self.listening_session_active = True
        self.interrupted_session = interrupted

        # ==================================================
        # NORMAL WAKE-WORD SESSION
        # ==================================================

        if not interrupted:
            if self.state_machine.can_handle(
                VoiceEvent.LISTENING_STARTED
            ):
                self.state_machine.handle_event(
                    VoiceEvent.LISTENING_STARTED
                )

            logger.info("Listening for command...")
            return

        # ==================================================
        # BARGE-IN SESSION
        # ==================================================

        logger.info("Listening after interruption.")

        buffered_frames = list(self.recent_frames)
        self.recent_frames.clear()

        logger.debug(
            "Recovering %d recent audio frames.",
            len(buffered_frames),
        )

        for buffered_frame in buffered_frames:
            if not self.running:
                break

            if self.state_machine.state != VoiceState.LISTENING:
                break

            self._process_frame(buffered_frame)

            if not self.listening_session_active:
                break

    # ============================================================
    # PROCESS AUDIO FRAME
    # ============================================================

    def _process_frame(self, frame):
        """Process one microphone frame through command VAD."""
        if not self.listening_session_active:
            return

        # ============================================================
        # Wake-word suppression
        # ============================================================

        if not self.interrupted_session:
            elapsed = time.monotonic() - self.listen_started_at

            if elapsed < self.wake_suppression_seconds:
                return

            # If the user never starts speaking after wake detection,
            # disengage after 8 seconds.
            if elapsed > 8.0 and not self.speech_started:
                logger.info(
                    "Post-wake silence timeout (8s). Disengaging."
                )

                self.listening_session_active = False
                self.interrupted_session = False
                self.audio_buffer.clear()
                self.preroll_frames.clear()
                self.recent_frames.clear()
                self.state_machine.reset()
                return

        # ------------------------------------------------------------
        # Maintain rolling pre-speech preroll.
        # ------------------------------------------------------------

        self.preroll_frames.append(frame.copy())

        # ============================================================
        # Run command VAD
        # ============================================================

        min_silence_ms = (
            1200 if self.interrupted_session else None
        )

        try:
            result = self.vad.process(
                frame,
                min_silence_ms=min_silence_ms,
            )
        except Exception:
            logger.exception(
                "Command VAD processing failed."
            )

            self.listening_session_active = False
            self.interrupted_session = False
            self.audio_buffer.clear()
            self.preroll_frames.clear()
            self.state_machine.reset()
            return

        if result is None:
            return

        # ============================================================
        # Speech started
        # ============================================================

        if "start" in result:
            logger.info("Speech detected.")

            self.speech_started = True

            # Preserve recent frames so the beginning of speech is not lost.
            for p_frame in self.preroll_frames:
                self.audio_buffer.append(p_frame.copy())

            self.preroll_frames.clear()
            return

        # ============================================================
        # Speech continues
        # ============================================================

        if "speech" in result:
            self.speech_started = True
            self.audio_buffer.append(frame.copy())
            return

        # ============================================================
        # Speech ended
        # ============================================================

        if "end" in result:
            logger.info("Speech complete.")

            # Include the final frame.
            self.audio_buffer.append(frame.copy())

            # Capture the session-specific interruption flag BEFORE clearing
            # it. This value must belong to this transcription task only.
            is_interrupted = self.interrupted_session

            self.listening_session_active = False
            self.interrupted_session = False
            self.speech_started = False

            # ------------------------------------------------------------
            # LISTENING -> TRANSCRIBING
            # ------------------------------------------------------------

            try:
                self.state_machine.handle_event(
                    VoiceEvent.SPEECH_COMPLETE
                )
            except Exception:
                logger.exception(
                    "Failed to transition to TRANSCRIBING state."
                )

                self.audio_buffer.clear()
                self.preroll_frames.clear()
                self.recent_frames.clear()
                self.state_machine.reset()
                return

            # ------------------------------------------------------------
            # Combine captured frames.
            # ------------------------------------------------------------

            if not self.audio_buffer:
                logger.warning(
                    "Speech ended but no audio was captured."
                )

                self.state_machine.reset()
                return

            try:
                audio = np.concatenate(self.audio_buffer)
            except Exception:
                logger.exception(
                    "Failed to combine captured command audio."
                )

                self.audio_buffer.clear()
                self.state_machine.reset()
                return

            self.audio_buffer = []
            self.recent_frames.clear()
            self.preroll_frames.clear()

            logger.debug(
                "Captured %d audio samples.",
                len(audio),
            )

            # ------------------------------------------------------------
            # Transcribe asynchronously.
            #
            # is_interrupted is passed explicitly so another session cannot
            # overwrite the metadata for this transcription task.
            # ------------------------------------------------------------

            try:
                self._track_background_task(
                    self._transcribe(
                        audio,
                        is_interrupted,
                    ),
                    "command-transcription",
                )
            except Exception:
                logger.exception(
                    "Failed to schedule command transcription."
                )
                self.state_machine.reset()

    # ============================================================
    # TRANSCRIPTION
    # ============================================================

    async def _transcribe(
        self,
        audio,
        is_interrupted: bool,
    ):
        """
        Transcribe one captured command.

        is_interrupted is immutable session metadata passed by the caller.
        It must not be read from self.interrupted_session because that value
        belongs to the next/current microphone session.
        """
        try:
            if (
                self.state_machine.state
                != VoiceState.TRANSCRIBING
            ):
                logger.debug(
                    "Ignoring transcription because state is %s.",
                    self.state_machine.state,
                )
                return

            logger.info("Transcribing command...")

            text = await asyncio.to_thread(
                self.stt.transcribe,
                audio,
            )

            text = text.strip() if text else ""

            logger.info(
                "Transcription: %s",
                text if text else "<empty>",
            )

            # ==================================================
            # Empty transcription / interruption filler
            # ==================================================

            if not text or (
                is_interrupted
                and is_interruption_filler(text)
            ):
                if is_interrupted:
                    logger.info(
                        "Interruption filler or pause recognized ('%s'). "
                        "Continuing to listen for the command.",
                        text or "silence",
                    )

                    self.listening_session_active = False
                    self.interrupted_session = True
                    self.speech_started = False

                    if self.state_machine.can_handle(
                        VoiceEvent.LISTENING_STARTED
                    ):
                        self.state_machine.handle_event(
                            VoiceEvent.LISTENING_STARTED
                        )
                    else:
                        self.state_machine.state = VoiceState.LISTENING

                    return

                logger.warning("No speech recognized.")
                self.listening_session_active = False
                self.interrupted_session = False
                self.state_machine.reset()
                return

            # ==================================================
            # Stop command handling
            # ==================================================

            if is_stop_command(text):
                logger.info(
                    "Stop command recognized: '%s'. Returning to IDLE.",
                    text,
                )

                self.listening_session_active = False
                self.interrupted_session = False
                self.state_machine.reset()
                return

            # ==================================================
            # TRANSCRIBING -> THINKING
            # ==================================================

            try:
                self.state_machine.handle_event(
                    VoiceEvent.TRANSCRIPTION_COMPLETE
                )
            except Exception:
                logger.exception(
                    "Failed to transition to THINKING state."
                )
                self.state_machine.reset()
                return

            # ------------------------------------------------------------
            # Process agent asynchronously.
            # ------------------------------------------------------------

            try:
                self._track_background_task(
                    self._process_agent(text),
                    "command-agent-processing",
                )
            except Exception:
                logger.exception(
                    "Failed to schedule agent processing."
                )
                self.state_machine.reset()

        except asyncio.CancelledError:
            logger.debug("Command transcription cancelled.")
            raise

        except Exception:
            logger.exception(
                "Transcription failed unexpectedly."
            )

            self.listening_session_active = False
            self.interrupted_session = False
            self.state_machine.reset()

    # ============================================================
    # AGENT
    # ============================================================

    async def _process_agent(self, text: str):
        """Pass the recognized command to AgentController."""
        try:
            response = await self.agent_controller.process(text)

            if response is None:
                logger.debug(
                    "Agent controller returned no response."
                )
                return

            logger.info("Agent response received.")

        except asyncio.CancelledError:
            logger.debug("Agent processing cancelled.")
            raise

        except Exception:
            logger.exception(
                "Agent processing error."
            )

    # ============================================================
    # STOP
    # ============================================================

    async def stop(self):
        """Stop the listener and all of its background work."""
        if not self.running:
            # Even if the main listener is already stopped, make sure
            # detached tasks are not left alive.
            await self._cancel_background_tasks()
            return

        self.running = False

        task = self.task
        self.task = None

        if task is not None:
            current_task = asyncio.current_task()

            if task is not current_task:
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(
                        "Command listener task cancelled successfully."
                    )
                except Exception:
                    logger.exception(
                        "Error while stopping command listener task."
                    )

        await self._cancel_background_tasks()

        self.audio_buffer.clear()
        self.recent_frames.clear()
        self.preroll_frames.clear()

        self.listening_session_active = False
        self.interrupted_session = False
        self.speech_started = False

        logger.info("Command listener stopped.")
