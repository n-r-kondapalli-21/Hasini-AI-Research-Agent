import asyncio
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from voice.state_machine import VoiceStateMachine, VoiceState, VoiceEvent
from voice.command_listener import CommandListener


class MockBroadcaster:
    def subscribe(self):
        return asyncio.Queue()


class MockVAD:
    def __init__(self):
        self.call_count = 0

    def reset(self):
        pass

    def process(self, frame, min_silence_ms=None):
        self.call_count += 1
        if self.call_count == 1:
            return {"start": True}
        elif self.call_count == 2:
            return {"end": True}
        return None


class MockSTT:
    def __init__(self, text=""):
        self.text = text

    def transcribe(self, audio):
        return self.text


class MockAgentController:
    def __init__(self, state_machine):
        self.state_machine = state_machine
        self.current_task = None
        self.tts_tasks = []
        self.generation = 0

    async def cancel_response(self):
        task_to_cancel = self.current_task
        self.current_task = None
        self.generation += 1

        if task_to_cancel is not None and not task_to_cancel.done():
            task_to_cancel.cancel()

        if self.state_machine.can_handle(VoiceEvent.USER_SPEECH):
            self.state_machine.handle_event(VoiceEvent.USER_SPEECH)
        if self.state_machine.can_handle(VoiceEvent.TTS_CANCELLED):
            self.state_machine.handle_event(VoiceEvent.TTS_CANCELLED)
        if self.state_machine.can_handle(VoiceEvent.LISTENING_STARTED):
            self.state_machine.handle_event(VoiceEvent.LISTENING_STARTED)


async def test_interruption_recovery():
    print("\n🧪 Testing Interruption Flow & Second Command Recovery...")

    state_machine = VoiceStateMachine()
    broadcaster = MockBroadcaster()
    vad = MockVAD()
    stt = MockSTT(text="")  # Empty transcription during interruption
    agent_controller = MockAgentController(state_machine)

    listener = CommandListener(
        broadcaster=broadcaster,
        vad=vad,
        stt=stt,
        agent_controller=agent_controller,
        state_machine=state_machine,
    )

    # 1. Start in SPEAKING state
    state_machine.state = VoiceState.SPEAKING
    assert state_machine.state == VoiceState.SPEAKING

    # 2. User interrupts
    await agent_controller.cancel_response()

    # Verify state machine transitioned to LISTENING
    assert state_machine.state == VoiceState.LISTENING, f"Expected LISTENING, got {state_machine.state}"
    print("✅ State transitioned SPEAKING → USER_INTERRUPT → STOP_TTS → LISTENING successfully.")

    # 3. Simulate empty audio / filler during interruption
    listener.interrupted_session = True
    listener.listening_session_active = True

    # Process frame to trigger end of initial interrupted audio segment
    frame = np.zeros(512, dtype=np.float32)
    listener._process_frame(frame)
    listener._process_frame(frame)

    # Wait briefly for async _transcribe task
    await asyncio.sleep(0.1)

    # Verify state machine is STILL in LISTENING state (not IDLE!)
    assert state_machine.state == VoiceState.LISTENING, f"Expected LISTENING after empty interruption, got {state_machine.state}"
    print("✅ System preserved LISTENING state after empty interruption audio (did NOT reset to IDLE).")

    # 4. Now user speaks second command
    listener.stt.text = "What is the capital of France?"
    vad.call_count = 0

    await listener._start_listening_session(interrupted=True)
    listener._process_frame(frame)
    listener._process_frame(frame)

    await asyncio.sleep(0.1)

    # Verify state transitioned to THINKING for the second command
    assert state_machine.state == VoiceState.THINKING, f"Expected THINKING for second command, got {state_machine.state}"
    print("✅ Second command ('What is the capital of France?') processed successfully into THINKING state!")

    print("\n🎉 ALL INTERRUPTION RECOVERY TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(test_interruption_recovery())
