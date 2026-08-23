import asyncio

from main import (
    create_research_agent,
    Agent_stream,
)

from services.conversation_memory import (
    ConversationMemory,
)

from ..audio_capture import AudioCapture
from ..audio_broadcaster import AudioBroadcaster
from ..audio_queue import OrderedAudioQueue

from ..providers.factory import (
    create_stt_provider,
    create_tts_provider,
    create_vad_provider,
    create_wakeword_provider,
)

from ..command_listener import CommandListener
from ..agent_controller import AgentController
from ..barge_in_detector import BargeInDetector
from ..state_machine import VoiceStateMachine
from ..wake_listener import WakeWordListener


async def main():

    # ==================================================
    # Core audio components
    # ==================================================

    capture = AudioCapture()

    broadcaster = AudioBroadcaster(
        capture
    )

    state_machine = VoiceStateMachine()

    # ==================================================
    # Providers
    # ==================================================

    wakeword = create_wakeword_provider()

    # VAD for normal command listening
    vad = create_vad_provider()

    # STT
    stt = create_stt_provider()

    # TTS
    tts = create_tts_provider()

    # ==================================================
    # Audio playback queue
    # ==================================================

    audio_queue = OrderedAudioQueue()

    # ==================================================
    # Separate VAD for barge-in
    # ==================================================

    barge_vad = create_vad_provider()

    # ==================================================
    # Existing research agent
    # ==================================================

    print(
        "\n🧠 Creating research agent..."
    )

    agent, mcp_client, all_tools = (
        await create_research_agent()
    )

    memory = ConversationMemory(
        history_limit=10,
        enabled=True,
    )

    # ==================================================
    # Agent controller
    # ==================================================

    agent_controller = AgentController(
        agent=agent,
        memory=memory,
        agent_stream=Agent_stream,
        tts=tts,
        state_machine=state_machine,
        audio_queue=audio_queue,
    )

    # ==================================================
    # Wake-word listener
    # ==================================================

    wake_listener = WakeWordListener(
        broadcaster=broadcaster,
        wakeword=wakeword,
        state_machine=state_machine,
    )

    # ==================================================
    # Command listener
    # ==================================================

    command_listener = CommandListener(
        broadcaster=broadcaster,
        vad=vad,
        stt=stt,
        agent_controller=agent_controller,
        state_machine=state_machine,
    )

    # ==================================================
    # Barge-in detector
    # ==================================================

    barge_in_detector = BargeInDetector(
        broadcaster=broadcaster,
        vad=barge_vad,
        state_machine=state_machine,
        agent_controller=agent_controller,
    )

    # ==================================================
    # Start system
    # ==================================================

    await capture.start()

    await broadcaster.start()

    await agent_controller.start()

    await wake_listener.start()

    await command_listener.start()

    await barge_in_detector.start()

    # ==================================================
    # Test information
    # ==================================================

    print(
        "\n" + "=" * 50
    )

    print(
        "🎤 Hasini Phase 3 Agent Test"
    )

    print(
        "=" * 50
    )

    print(
        "\nSay: Hey Jarvis"
    )

    print(
        "Then give your command."
    )

    print(
        "While Hasini is speaking, "
        "interrupt her to test barge-in."
    )

    print(
        "\nPress Ctrl+C to stop.\n"
    )

    # ==================================================
    # Main loop
    # ==================================================

    try:

        while True:

            await asyncio.sleep(1)

    finally:

        print(
            "\n🛑 Shutting down..."
        )

        # Stop listeners first.
        await barge_in_detector.stop()

        await command_listener.stop()

        await wake_listener.stop()

        # Stop agent / TTS / playback.
        await agent_controller.stop()

        # Stop shared audio infrastructure.
        await broadcaster.stop()

        await capture.stop()

        print(
            "✅ Hasini stopped."
        )


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print(
            "\n🛑 Test stopped."
        )