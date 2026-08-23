from ..state_machine import (
    VoiceEvent,
    VoiceState,
    VoiceStateMachine,
)
def test_barge_in():

    machine = VoiceStateMachine()

    machine.handle_event(
        VoiceEvent.WAKE_WORD
    )

    machine.handle_event(
        VoiceEvent.LISTENING_STARTED
    )

    machine.handle_event(
        VoiceEvent.SPEECH_COMPLETE
    )

    machine.handle_event(
        VoiceEvent.TRANSCRIPTION_COMPLETE
    )

    machine.handle_event(
        VoiceEvent.RESPONSE_STARTED
    )

    machine.handle_event(
        VoiceEvent.TTS_STARTED
    )

    assert (
        machine.state
        == VoiceState.SPEAKING
    )

    machine.handle_event(
        VoiceEvent.USER_SPEECH
    )

    assert (
        machine.state
        == VoiceState.USER_INTERRUPT
    )

    print(
        "✅ Barge-in state transition passed."
    )

def main():

    machine = VoiceStateMachine()

    assert machine.state == VoiceState.IDLE

    machine.handle_event(
        VoiceEvent.WAKE_WORD
    )

    assert (
        machine.state
        == VoiceState.WAKE_DETECTED
    )

    machine.handle_event(
        VoiceEvent.LISTENING_STARTED
    )

    assert (
        machine.state
        == VoiceState.LISTENING
    )

    machine.handle_event(
        VoiceEvent.SPEECH_COMPLETE
    )

    assert (
        machine.state
        == VoiceState.TRANSCRIBING
    )

    machine.handle_event(
        VoiceEvent.TRANSCRIPTION_COMPLETE
    )

    assert (
        machine.state
        == VoiceState.THINKING
    )

    machine.handle_event(
        VoiceEvent.RESPONSE_STARTED
    )

    assert (
        machine.state
        == VoiceState.RESPONDING
    )

    machine.handle_event(
        VoiceEvent.TTS_STARTED
    )

    assert (
        machine.state
        == VoiceState.SPEAKING
    )

    machine.handle_event(
        VoiceEvent.TTS_COMPLETE
    )

    assert (
        machine.state
        == VoiceState.IDLE
    )

    print("\n✅ State machine test passed.")


if __name__ == "__main__":
    test_barge_in()
    main()