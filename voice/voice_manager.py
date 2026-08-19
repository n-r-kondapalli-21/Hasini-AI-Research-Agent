import asyncio
import re

from .audio_io import record_audio, play_audio
from .stt import SpeechToText
from .tts import TextToSpeech
from .config import MAX_RECORD_SECONDS

from main import create_research_agent, Agent
from services.conversation_memory import ConversationMemory


def is_exit_command(text: str) -> bool:
    """
    Check whether the user wants to exit voice mode.
    """

    text = text.lower().strip()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    words = text.split()

    exit_commands = {
        "exit",
        "quit",
        "goodbye",
    }

    if text in exit_commands:
        return True

    return any(
        word in exit_commands
        for word in words
    )


async def voice_loop(agent):

    stt = SpeechToText()
    tts = TextToSpeech()

    memory = ConversationMemory(
        history_limit=10,
        enabled=True
    )

    while True:

        # --------------------------------------------------
        # Start a new voice command
        # --------------------------------------------------

        input("\nPress ENTER to speak...")

        while True:

            # ----------------------------------------------
            # Record
            # ----------------------------------------------

            audio = record_audio(MAX_RECORD_SECONDS)

            # ----------------------------------------------
            # STT
            # ----------------------------------------------

            text = stt.transcribe(audio)

            if not text:
                print("❌ I didn't hear anything.")
                print("🔄 Please try again.")
                continue

            print("\n" + "-" * 50)
            print(f"🎤 Transcription: {text}")
            print("-" * 50)

            # ----------------------------------------------
            # Exit
            # ----------------------------------------------

            if is_exit_command(text):
                print("\n👋 Goodbye!")
                return

            # ----------------------------------------------
            # Confirmation
            # ----------------------------------------------

            print("\nPress ENTER to send this command.")
            print("Press R + ENTER to record again.")
            print("Press Q + ENTER to cancel this command.")

            choice = input("\n>>> ").strip().lower()

            # ----------------------------------------------
            # Accept
            # ----------------------------------------------

            if choice == "":
                break

            # ----------------------------------------------
            # Record again
            # ----------------------------------------------

            if choice == "r":
                print("\n🔄 Recording again...")
                continue

            # ----------------------------------------------
            # Cancel
            # ----------------------------------------------

            if choice == "q":
                print("\n❌ Command cancelled.")
                break

            # ----------------------------------------------
            # Invalid choice
            # ----------------------------------------------

            print(
                "\n⚠️ Invalid option."
                "\nUse ENTER, R, or Q."
            )

        # --------------------------------------------------
        # If Q was selected, start a fresh command
        # --------------------------------------------------

        if choice == "q":
            continue

        # --------------------------------------------------
        # Send confirmed command to Hasini
        # --------------------------------------------------

        print("\n🤔 Hasini is thinking...")

        try:

            response = await Agent(
                text,
                agent,
                memory
            )

        except Exception as e:

            print(f"\n❌ Agent error: {e}")
            continue

        print(f"\n🤖 Hasini: {response}")

        # --------------------------------------------------
        # TTS
        # --------------------------------------------------

        try:

            audio_data, sample_rate = (tts.synthesize(response))

            if audio_data is not None:

                play_audio(
                    audio_data,
                    sample_rate
                )

        except Exception as e:

            print(f"\n❌ TTS error: {e}")


async def main():

    print("=" * 50)
    print("🤖 Hasini Voice Mode")
    print("=" * 50)

    print("\nConnecting to agent...")

    agent, mcp_client, all_tools = (
        await create_research_agent()
    )

    print("✅ Agent connected.")
    print("🎤 Voice mode ready.")
    print("Press Ctrl+C anytime to exit.")

    try:

        await voice_loop(agent)

    except (KeyboardInterrupt, EOFError):

        print("\n\n🛑 Voice mode interrupted.")
        print("👋 Goodbye!")

    finally:
        try:
            await mcp_client.close()

        except Exception:
            pass


if __name__ == "__main__":

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("👋 Hasini stopped.")