import sys
from enum import Enum, auto

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class VoiceState(Enum):
    """
    High-level state of the voice system.
    """

    IDLE = auto()

    WAKE_DETECTED = auto()

    LISTENING = auto()

    TRANSCRIBING = auto()

    THINKING = auto()

    TOOL_EXECUTION = auto()

    CONFIRM_WAIT = auto()

    RESPONDING = auto()

    SPEAKING = auto()

    USER_INTERRUPT = auto()

    STOP_TTS = auto()


class VoiceEvent(Enum):
    """
    Events that can cause a state transition.
    """

    WAKE_WORD = auto()

    LISTENING_STARTED = auto()

    SPEECH_COMPLETE = auto()

    TRANSCRIPTION_COMPLETE = auto()

    THINKING_STARTED = auto()

    TOOL_STARTED = auto()

    TOOL_COMPLETED = auto()

    RESPONSE_STARTED = auto()

    RESPONSE_COMPLETE = auto()

    TTS_STARTED = auto()

    TTS_COMPLETE = auto()

    USER_SPEECH = auto()

    STOP_REQUESTED = auto()

    TTS_CANCELLED = auto()


class VoiceStateMachine:
    """
    Explicit state machine for the voice system.

    The state machine owns state transitions.

    The voice manager/controller owns actual task lifecycle.
    """

    def __init__(self):

        self.state = VoiceState.IDLE

        self._transition_table = {

            # --------------------------------------------------
            # IDLE
            # --------------------------------------------------

            VoiceState.IDLE: {

                VoiceEvent.WAKE_WORD:
                    VoiceState.WAKE_DETECTED,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # WAKE DETECTED
            # --------------------------------------------------

            VoiceState.WAKE_DETECTED: {

                VoiceEvent.LISTENING_STARTED:
                    VoiceState.LISTENING,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # LISTENING
            # --------------------------------------------------

            VoiceState.LISTENING: {

                VoiceEvent.SPEECH_COMPLETE:
                    VoiceState.TRANSCRIBING,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # TRANSCRIBING
            # --------------------------------------------------

            VoiceState.TRANSCRIBING: {

                VoiceEvent.TRANSCRIPTION_COMPLETE:
                    VoiceState.THINKING,

                VoiceEvent.LISTENING_STARTED:
                    VoiceState.LISTENING,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # THINKING
            # --------------------------------------------------

            VoiceState.THINKING: {

                VoiceEvent.TOOL_STARTED:
                    VoiceState.TOOL_EXECUTION,

                VoiceEvent.RESPONSE_STARTED:
                    VoiceState.RESPONDING,

                VoiceEvent.USER_SPEECH:
                    VoiceState.USER_INTERRUPT,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # TOOL EXECUTION
            # --------------------------------------------------

            VoiceState.TOOL_EXECUTION: {

                VoiceEvent.TOOL_COMPLETED:
                    VoiceState.RESPONDING,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # CONFIRMATION
            # --------------------------------------------------

            VoiceState.CONFIRM_WAIT: {

                VoiceEvent.RESPONSE_STARTED:
                    VoiceState.RESPONDING,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # RESPONDING
            # --------------------------------------------------

            VoiceState.RESPONDING: {

                VoiceEvent.TTS_STARTED:
                    VoiceState.SPEAKING,

                VoiceEvent.USER_SPEECH:
                    VoiceState.USER_INTERRUPT,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # SPEAKING
            # --------------------------------------------------

            VoiceState.SPEAKING: {

                VoiceEvent.USER_SPEECH:
                    VoiceState.USER_INTERRUPT,

                VoiceEvent.TTS_COMPLETE:
                    VoiceState.IDLE,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # USER INTERRUPT
            # --------------------------------------------------

            VoiceState.USER_INTERRUPT: {

                VoiceEvent.TTS_CANCELLED:
                    VoiceState.STOP_TTS,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },

            # --------------------------------------------------
            # STOP TTS
            # --------------------------------------------------

            VoiceState.STOP_TTS: {

                VoiceEvent.LISTENING_STARTED:
                    VoiceState.LISTENING,

                VoiceEvent.STOP_REQUESTED:
                    VoiceState.IDLE,
            },
        }

    # ============================================================
    # HANDLE EVENT
    # ============================================================

    def handle_event(
        self,
        event: VoiceEvent,
    ) -> VoiceState:

        transitions = self._transition_table.get(
            self.state,
            {},
        )

        next_state = transitions.get(
            event
        )

        if next_state is None:

            raise ValueError(
                f"Invalid voice transition: "
                f"{self.state.name} "
                f"+ {event.name}"
            )

        previous_state = self.state

        self.state = next_state

        print(
            f"🔄 Voice state: "
            f"{previous_state.name} "
            f"→ {self.state.name}"
        )

        return self.state

    # ============================================================
    # CHECK EVENT
    # ============================================================

    def can_handle(
        self,
        event: VoiceEvent,
    ) -> bool:

        return event in self._transition_table.get(
            self.state,
            {},
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        self.state = VoiceState.IDLE