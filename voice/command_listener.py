import asyncio
import re
import time
from collections import deque

import numpy as np


def is_stop_command(text: str) -> bool:
    """
    Check whether user speech is a stop or cancel directive.
    """
    if not text:
        return False

    clean_text = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    words = clean_text.split()

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

    if clean_text in stop_phrases:
        return True

    stop_words = {
        "stop",
        "quiet",
        "cancel",
        "pause",
        "halt",
        "silence",
        "enough",
        "exit",
        "quit",
        "goodbye",
    }

    return any(w in stop_words for w in words)


def is_interruption_filler(text: str) -> bool:
    """
    Check whether user speech is a transient pause/filler word after barge-in.
    """
    if not text:
        return True

    clean_text = re.sub(r"[^\w\s]", " ", text.lower()).strip()

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

from .audio_broadcaster import AudioBroadcaster

from .state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)

from .providers.vad.base import VADProvider
from .providers.stt.base import STTProvider

from .agent_controller import AgentController


class CommandListener:
    """
    Handles microphone input after wake-word detection
    and after a barge-in interruption.

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

        # ============================================================
        # Microphone subscription
        # ============================================================

        self.queue = broadcaster.subscribe()

        self.running = False

        self.task = None

        # ============================================================
        # Command audio
        # ============================================================

        self.audio_buffer = []

        # ============================================================
        # Wake-word suppression
        # ============================================================

        self.wake_suppression_seconds = 0.6

        self.listen_started_at = 0.0

        # ============================================================
        # Command session state
        # ============================================================

        self.listening_session_active = False

        self.interrupted_session = False

        # ============================================================
        # Barge-in pre-buffer
        # ============================================================
        #
        # VAD frame = 32 ms
        #
        # 25 frames × 32 ms = 800 ms
        #
        # This continuously stores the latest microphone audio.
        #
        # When barge-in is confirmed, the beginning of the user's
        # command can be recovered even though the barge-in detector
        # needed ~384 ms to confirm sustained speech.
        #
        # ============================================================

        self.pre_buffer_frames = 40

        self.recent_frames = deque(
            maxlen=self.pre_buffer_frames
        )

        self.preroll_frames = deque(
            maxlen=15
        )

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
            "🎧 Command listener started."
        )

    # ============================================================
    # MAIN LISTENING LOOP
    # ============================================================

    async def _listen_loop(self):

        try:

            while self.running:

                # --------------------------------------------------
                # Receive microphone frame
                # --------------------------------------------------

                frame = await self.queue.get()

                # --------------------------------------------------
                # ALWAYS maintain rolling microphone buffer.
                #
                # This must happen before checking the state.
                #
                # That way, when the user starts speaking during
                # SPEAKING, the beginning of their command is already
                # available when barge-in is confirmed.
                # --------------------------------------------------

                self.recent_frames.append(
                    frame.copy()
                )

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

                    self._process_frame(
                        frame
                    )

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

                    # IMPORTANT:
                    #
                    # Do NOT call _process_frame(frame) here.
                    #
                    # _start_listening_session() already replays
                    # the recent buffered frames, including this
                    # frame.
                    #
                    continue

                # ==================================================
                # Continue active command session
                # ==================================================

                if (
                    state == VoiceState.LISTENING
                    and self.listening_session_active
                ):

                    self._process_frame(
                        frame
                    )

                    continue

        except asyncio.CancelledError:

            pass

        except Exception as e:

            print(
                f"\n❌ Command listener error: {e}"
            )

    # ============================================================
    # START LISTENING SESSION
    # ============================================================

    async def _start_listening_session(
        self,
        interrupted: bool = False,
    ):

        # --------------------------------------------------
        # Reset command VAD state
        # --------------------------------------------------

        self.vad.reset()

        self.audio_buffer = []

        self.preroll_frames.clear()

        self.listen_started_at = (
            time.monotonic()
        )

        self.listening_session_active = True

        self.interrupted_session = interrupted

        # ==================================================
        # NORMAL WAKE-WORD SESSION
        # ==================================================

        if not interrupted:

            self.state_machine.handle_event(
                VoiceEvent.LISTENING_STARTED
            )

            print(
                "🎤 Listening for command..."
            )

            return

        # ==================================================
        # BARGE-IN SESSION
        # ==================================================

        print(
            "🎤 Listening after interruption..."
        )

        # --------------------------------------------------
        # Recover buffered microphone audio.
        #
        # This contains the beginning of the user's speech
        # that happened while BargeInDetector was confirming
        # the interruption.
        # --------------------------------------------------

        buffered_frames = list(
            self.recent_frames
        )

        self.recent_frames.clear()

        print(
            f"📼 Recovering "
            f"{len(buffered_frames)} "
            f"recent audio frames."
        )

        # --------------------------------------------------
        # Feed buffered audio through command VAD.
        # --------------------------------------------------

        for buffered_frame in buffered_frames:

            if not self.running:
                break

            if (
                self.state_machine.state
                != VoiceState.LISTENING
            ):
                break

            self._process_frame(
                buffered_frame
            )

            # --------------------------------------------------
            # If speech already ended while processing the
            # recovered frames, stop replaying.
            # --------------------------------------------------

            if not self.listening_session_active:

                break

    # ============================================================
    # PROCESS AUDIO FRAME
    # ============================================================

    def _process_frame(
        self,
        frame,
    ):

        # ============================================================
        # Wake-word suppression
        # ============================================================

        # Only normal wake-word sessions need suppression.
        #
        # Barge-in sessions must process audio immediately because
        # the user is already speaking and there is no wake word.
        #

        if not self.interrupted_session:

            elapsed = (
                time.monotonic()
                - self.listen_started_at
            )

            if (
                elapsed
                < self.wake_suppression_seconds
            ):

                return

        # --------------------------------------------------
        # Maintain rolling pre-speech preroll buffer
        # --------------------------------------------------
        self.preroll_frames.append(frame.copy())

        # ============================================================
        # Run command VAD
        # ============================================================

        min_silence_ms = 1200 if self.interrupted_session else None

        result = self.vad.process(
            frame,
            min_silence_ms=min_silence_ms,
        )

        if result is None:

            return

        # ============================================================
        # Speech started
        # ============================================================

        if "start" in result:

            print(
                "🗣️ Speech detected."
            )

            # Prepend recent preroll frames so initial speech onset is preserved.
            for p_frame in self.preroll_frames:
                self.audio_buffer.append(p_frame.copy())

            self.preroll_frames.clear()

            return

        # ============================================================
        # Speech continues
        # ============================================================

        if "speech" in result:

            self.audio_buffer.append(
                frame.copy()
            )

            return

        # ============================================================
        # Speech ended
        # ============================================================

        if "end" in result:

            print(
                "✅ Speech complete."
            )

            # Include final frame.
            self.audio_buffer.append(
                frame.copy()
            )

            # --------------------------------------------------
            # End current listening session.
            # --------------------------------------------------

            self.listening_session_active = False

            # Note: self.interrupted_session is intentionally NOT cleared here.
            # It must be preserved until _transcribe() runs so _transcribe() can
            # determine whether this audio segment originated from an interruption.

            # --------------------------------------------------
            # LISTENING → TRANSCRIBING
            # --------------------------------------------------

            self.state_machine.handle_event(
                VoiceEvent.SPEECH_COMPLETE
            )

            # --------------------------------------------------
            # Combine captured frames.
            # --------------------------------------------------

            audio = np.concatenate(
                self.audio_buffer
            )

            self.audio_buffer = []

            # --------------------------------------------------
            # The command has now been extracted.
            # We no longer need the old microphone frames.
            # --------------------------------------------------

            self.recent_frames.clear()

            print(
                f"📦 Captured "
                f"{len(audio)} samples."
            )

            # --------------------------------------------------
            # Transcribe asynchronously.
            # --------------------------------------------------

            asyncio.create_task(
                self._transcribe(
                    audio
                )
            )

    # ============================================================
    # TRANSCRIPTION
    # ============================================================

    async def _transcribe(
        self,
        audio,
    ):

        is_interrupted = self.interrupted_session
        self.interrupted_session = False

        try:

            # --------------------------------------------------
            # Verify state.
            # --------------------------------------------------

            if (
                self.state_machine.state
                != VoiceState.TRANSCRIBING
            ):

                return

            print(
                "📝 Transcribing..."
            )

            # --------------------------------------------------
            # Run STT outside event loop.
            # --------------------------------------------------

            text = await asyncio.to_thread(
                self.stt.transcribe,
                audio,
            )

            text = (
                text.strip()
                if text
                else ""
            )

            print(
                f"📝 Transcription: {text}"
            )

            # ==================================================
            # Empty transcription or Interruption Filler
            # ==================================================

            if not text or (is_interrupted and is_interruption_filler(text)):

                if is_interrupted:

                    print(
                        f"🎤 Interruption filler or pause recognized ('{text or 'silence'}'). "
                        "Continuing to listen for your command..."
                    )

                    self.listening_session_active = False

                    self.interrupted_session = True

                    if self.state_machine.can_handle(
                        VoiceEvent.LISTENING_STARTED
                    ):
                        self.state_machine.handle_event(
                            VoiceEvent.LISTENING_STARTED
                        )
                    else:
                        self.state_machine.state = VoiceState.LISTENING

                    return

                else:

                    print(
                        "⚠️ No speech recognized."
                    )

                    self.listening_session_active = False

                    self.interrupted_session = False

                    self.state_machine.reset()

                    return

            # ==================================================
            # Stop command handling
            # ==================================================

            if is_stop_command(text):

                print(
                    f"🛑 Stop command recognized: '{text}'. Returning to IDLE."
                )

                self.listening_session_active = False

                self.interrupted_session = False

                self.state_machine.reset()

                return

            # ==================================================
            # TRANSCRIBING → THINKING
            # ==================================================

            self.state_machine.handle_event(
                VoiceEvent.TRANSCRIPTION_COMPLETE
            )

            # --------------------------------------------------
            # Process agent asynchronously.
            # --------------------------------------------------

            asyncio.create_task(
                self._process_agent(
                    text
                )
            )

        except asyncio.CancelledError:

            pass

        except Exception as e:

            print(
                f"\n❌ Transcription error: {e}"
            )

            self.listening_session_active = False

            self.interrupted_session = False

            self.state_machine.reset()

    # ============================================================
    # AGENT
    # ============================================================

    async def _process_agent(
        self,
        text: str,
    ):

        try:

            response = await (
                self.agent_controller.process(
                    text
                )
            )

            if response is None:

                return

            print(
                f"🤖 Agent response:\n{response}"
            )

        except asyncio.CancelledError:

            raise

        except Exception as e:

            print(
                f"❌ Agent processing error: {e}"
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

        self.audio_buffer = []

        self.recent_frames.clear()

        self.listening_session_active = False

        self.interrupted_session = False

        print(
            "🎧 Command listener stopped."
        )