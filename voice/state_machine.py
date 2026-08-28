import logging
from enum import Enum, auto


logger = logging.getLogger("hasini.voice.state_machine")


class VoiceState(Enum):
    """High-level state of the voice system."""

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
    """Events that can cause a state transition."""

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
            # IDLE
            VoiceState.IDLE: {
                VoiceEvent.WAKE_WORD: VoiceState.WAKE_DETECTED,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # WAKE DETECTED
            VoiceState.WAKE_DETECTED: {
                VoiceEvent.LISTENING_STARTED: VoiceState.LISTENING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # LISTENING
            VoiceState.LISTENING: {
                VoiceEvent.LISTENING_STARTED: VoiceState.LISTENING,
                VoiceEvent.SPEECH_COMPLETE: VoiceState.TRANSCRIBING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # TRANSCRIBING
            VoiceState.TRANSCRIBING: {
                VoiceEvent.TRANSCRIPTION_COMPLETE: VoiceState.THINKING,
                VoiceEvent.LISTENING_STARTED: VoiceState.LISTENING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # THINKING
            VoiceState.THINKING: {
                VoiceEvent.TOOL_STARTED: VoiceState.TOOL_EXECUTION,
                VoiceEvent.RESPONSE_STARTED: VoiceState.RESPONDING,
                VoiceEvent.USER_SPEECH: VoiceState.USER_INTERRUPT,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # TOOL EXECUTION
            VoiceState.TOOL_EXECUTION: {
                VoiceEvent.TOOL_COMPLETED: VoiceState.RESPONDING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # CONFIRMATION
            VoiceState.CONFIRM_WAIT: {
                VoiceEvent.RESPONSE_STARTED: VoiceState.RESPONDING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # RESPONDING
            VoiceState.RESPONDING: {
                VoiceEvent.TTS_STARTED: VoiceState.SPEAKING,
                VoiceEvent.USER_SPEECH: VoiceState.USER_INTERRUPT,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # SPEAKING
            VoiceState.SPEAKING: {
                VoiceEvent.USER_SPEECH: VoiceState.USER_INTERRUPT,
                VoiceEvent.TTS_COMPLETE: VoiceState.LISTENING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # USER INTERRUPT
            VoiceState.USER_INTERRUPT: {
                VoiceEvent.TTS_CANCELLED: VoiceState.STOP_TTS,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },

            # STOP TTS
            VoiceState.STOP_TTS: {
                VoiceEvent.LISTENING_STARTED: VoiceState.LISTENING,
                VoiceEvent.STOP_REQUESTED: VoiceState.IDLE,
            },
        }

        logger.debug(
            "Voice state machine initialized in %s state.",
            self.state.name,
        )

    def handle_event(self, event: VoiceEvent) -> VoiceState:
        """
        Handle an event and transition to the next state.

        Raises:
            TypeError: If event is not a VoiceEvent.
            ValueError: If the event is invalid for the current state.
        """
        if not isinstance(event, VoiceEvent):
            raise TypeError(
                f"event must be a VoiceEvent, got {type(event).__name__}"
            )

        transitions = self._transition_table.get(self.state, {})
        next_state = transitions.get(event)

        if next_state is None:
            logger.error(
                "Invalid voice transition: %s + %s",
                self.state.name,
                event.name,
            )
            raise ValueError(
                f"Invalid voice transition: "
                f"{self.state.name} + {event.name}"
            )

        previous_state = self.state
        self.state = next_state

        logger.info(
            "Voice state: %s → %s",
            previous_state.name,
            self.state.name,
        )

        return self.state

    def can_handle(self, event: VoiceEvent) -> bool:
        """Return whether the current state can handle the given event."""
        if not isinstance(event, VoiceEvent):
            return False

        return event in self._transition_table.get(
            self.state,
            {},
        )

    def reset(self) -> None:
        """Reset the voice state machine to IDLE."""
        previous_state = self.state
        self.state = VoiceState.IDLE

        if previous_state != VoiceState.IDLE:
            logger.info(
                "Voice state reset: %s → IDLE",
                previous_state.name,
            )
        else:
            logger.debug("Voice state machine already in IDLE state.")
