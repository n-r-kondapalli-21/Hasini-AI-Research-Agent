import asyncio
import sys
import os

sys.path.insert(0, ".")
from voice.state_machine import VoiceStateMachine, VoiceState, VoiceEvent
from voice.command_listener import is_stop_command


def test_multiturn_state_transitions():
    """Verify that TTS_COMPLETE transitions SPEAKING -> LISTENING for continuous multi-turn session."""
    sm = VoiceStateMachine()
    assert sm.state == VoiceState.IDLE

    # Turn 1 start
    sm.handle_event(VoiceEvent.WAKE_WORD)
    assert sm.state == VoiceState.WAKE_DETECTED

    sm.handle_event(VoiceEvent.LISTENING_STARTED)
    assert sm.state == VoiceState.LISTENING

    sm.handle_event(VoiceEvent.SPEECH_COMPLETE)
    assert sm.state == VoiceState.TRANSCRIBING

    sm.handle_event(VoiceEvent.TRANSCRIPTION_COMPLETE)
    assert sm.state == VoiceState.THINKING

    sm.handle_event(VoiceEvent.RESPONSE_STARTED)
    assert sm.state == VoiceState.RESPONDING

    sm.handle_event(VoiceEvent.TTS_STARTED)
    assert sm.state == VoiceState.SPEAKING

    # Response finishes -> automatically transitions to LISTENING for Turn 2 (NOT IDLE)
    sm.handle_event(VoiceEvent.TTS_COMPLETE)
    assert sm.state == VoiceState.LISTENING, "Multi-turn failed: TTS_COMPLETE did not transition to LISTENING!"

    # Turn 2 start without saying Hey Jarvis again
    sm.handle_event(VoiceEvent.SPEECH_COMPLETE)
    assert sm.state == VoiceState.TRANSCRIBING

    sm.handle_event(VoiceEvent.TRANSCRIPTION_COMPLETE)
    assert sm.state == VoiceState.THINKING

    sm.handle_event(VoiceEvent.RESPONSE_STARTED)
    assert sm.state == VoiceState.RESPONDING

    sm.handle_event(VoiceEvent.TTS_STARTED)
    assert sm.state == VoiceState.SPEAKING

    sm.handle_event(VoiceEvent.TTS_COMPLETE)
    assert sm.state == VoiceState.LISTENING, "Multi-turn failed: Turn 2 TTS_COMPLETE did not transition to LISTENING!"

    # Exit command terminates multi-turn session -> resets to IDLE
    assert is_stop_command("goodbye") is True
    sm.reset()
    assert sm.state == VoiceState.IDLE

    print("✅ Multi-turn continuous conversation lifecycle test passed!")


if __name__ == "__main__":
    test_multiturn_state_transitions()
