import asyncio

from main import (
    create_research_agent,
    Agent_stream,
)

from services.conversation_memory import (
    ConversationMemory,
)

from config import MEMORY_ENABLED, MEMORY_HISTORY_LIMIT

from .audio_capture import AudioCapture
from .audio_broadcaster import AudioBroadcaster
from .audio_queue import OrderedAudioQueue

from .providers.factory import (
    create_stt_provider,
    create_tts_provider,
    create_vad_provider,
    create_wakeword_provider,
)

from .confirmation_listener import VoiceConfirmationListener
from .confirmation import ConfirmationManager

from .command_listener import CommandListener
from .agent_controller import AgentController
from .barge_in_detector import BargeInDetector
from .state_machine import VoiceStateMachine
from .wake_listener import WakeWordListener


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

    # Separate VAD for barge-in
    barge_vad = create_vad_provider()

    # STT
    stt = create_stt_provider()

    # TTS
    tts = create_tts_provider()

    # ==================================================
    # Confirmation listener
    #
    # Uses the normal command VAD for now.
    # ==================================================

    confirmation_listener = VoiceConfirmationListener(
        vad,
        tts=tts,
    )

    confirmation_manager = ConfirmationManager(
        listen_callback=confirmation_listener.listen,
        speak_callback=confirmation_listener.speak,
        timeout_seconds=15,
    )

    # ==================================================
    # Audio playback queue
    # ==================================================

    audio_queue = OrderedAudioQueue()

    # ==================================================
    # Research agent
    #
    # IMPORTANT:
    # The confirmation manager must be supplied while
    # the gated MCP tools are being created.
    # ==================================================

    print(
        "\n🧠 Creating research agent..."
    )

    agent, mcp_client, all_tools = (
        await create_research_agent(
            confirmation_manager=confirmation_manager,
            mode="voice",
        )
    )

    memory = ConversationMemory(
        history_limit=MEMORY_HISTORY_LIMIT,
        enabled=MEMORY_ENABLED,
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
        "🎤 Hasini Phase 4 Agent Test"
    )

    print(
        "=" * 50
    )

    print(
        "\nSay: 'Hey Jarvis' ONCE to start a continuous conversation session."
    )

    print(
        "Hasini will respond and automatically keep listening for follow-up commands."
    )

    print(
        "For a MEDIUM/HIGH operation, Hasini will verbally ask for confirmation."
    )

    print(
        "Say 'exit', 'quit', or 'goodbye' (or wait 8s) to end the session."
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

        try:
            await mcp_client.close()
        except Exception:
            pass

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