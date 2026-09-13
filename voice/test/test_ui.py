"""
Unit test for Hasini JARVIS Voice UI component and UIServer.
"""

import asyncio
from voice.state_machine import VoiceState, VoiceStateMachine, VoiceEvent
from voice.ui import VoiceUI


def test_ui_state_transitions():
    sm = VoiceStateMachine()
    ui = VoiceUI(state_machine=sm, auto_open_browser=False)

    assert ui.voice_state == VoiceState.IDLE
    assert ui.server.current_state == "IDLE"

    sm.handle_event(VoiceEvent.WAKE_WORD)
    assert ui.voice_state == VoiceState.WAKE_DETECTED
    assert ui.server.current_state == "LISTENING"

    sm.handle_event(VoiceEvent.LISTENING_STARTED)
    assert ui.voice_state == VoiceState.LISTENING
    assert ui.server.current_state == "LISTENING"

    sm.handle_event(VoiceEvent.SPEECH_COMPLETE)
    assert ui.voice_state == VoiceState.TRANSCRIBING
    assert ui.server.current_state == "PROCESSING"

    sm.handle_event(VoiceEvent.TRANSCRIPTION_COMPLETE)
    assert ui.voice_state == VoiceState.THINKING
    assert ui.server.current_state == "PROCESSING"

    sm.handle_event(VoiceEvent.RESPONSE_STARTED)
    assert ui.voice_state == VoiceState.RESPONDING
    assert ui.server.current_state == "PROCESSING"

    sm.handle_event(VoiceEvent.TTS_STARTED)
    assert ui.voice_state == VoiceState.SPEAKING
    assert ui.server.current_state == "SPEAKING"

    sm.handle_event(VoiceEvent.TTS_COMPLETE)
    assert ui.voice_state == VoiceState.LISTENING
    assert ui.server.current_state == "LISTENING"

    sm.reset()
    assert ui.voice_state == VoiceState.IDLE
    assert ui.server.current_state == "IDLE"

    print("✅ JARVIS UI state transition test passed.")


def test_ui_text_updates():
    sm = VoiceStateMachine()
    ui = VoiceUI(state_machine=sm, auto_open_browser=False)

    ui.set_user_command("What is quantum computing?")
    assert ui.server.user_command == "What is quantum computing?"

    ui.append_agent_response("Quantum computing is ")
    ui.append_agent_response("a rapidly emerging technology...")
    assert ui.server.agent_response == "Quantum computing is a rapidly emerging technology..."

    ui.clear_agent_response()
    assert ui.server.agent_response == ""

    ui.set_error("Audio hardware disconnected")
    assert ui.server.current_state == "ERROR"
    assert ui.server.error_message == "Audio hardware disconnected"

    print("✅ JARVIS UI text update test passed.")


async def test_ui_async_lifecycle():
    sm = VoiceStateMachine()
    ui = VoiceUI(state_machine=sm, auto_open_browser=False)

    await ui.start()
    await asyncio.sleep(0.1)
    assert ui._running is True
    assert ui.server.server is not None

    await ui.stop()
    assert ui._running is False
    assert ui.server.server is None

    print("✅ JARVIS UI async server lifecycle test passed.")


def main():
    test_ui_state_transitions()
    test_ui_text_updates()
    asyncio.run(test_ui_async_lifecycle())
    print("\n✅ All JARVIS Voice UI tests passed successfully!")


if __name__ == "__main__":
    main()
