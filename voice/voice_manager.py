import asyncio
import re

import numpy as np

from .audio_io import record_until_silence
from .providers.factory import (create_stt_provider,create_tts_provider,create_vad_provider)
from .segmenter import ResponseSegmenter

from .audio_queue import OrderedAudioQueue

from .config import MAX_RECORD_SECONDS

from main import create_research_agent, Agent_stream
from services.conversation_memory import ConversationMemory


def is_exit_command(text: str) -> bool:
    """
    Check whether the user wants to exit voice mode.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

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


def _synthesize_blocking(tts, text):
    """
    Runs the (synchronous, CPU-bound) TTS generator to completion.
    Meant to be executed off the event loop via asyncio.to_thread.
    """

    audio_parts = []
    sample_rate = None

    for audio, rate in tts.synthesize_stream(text):

        if audio is None:
            continue

        audio_parts.append(audio)
        sample_rate = rate

    if not audio_parts:
        return None, None

    if len(audio_parts) == 1:
        final_audio = audio_parts[0]
    else:
        final_audio = np.concatenate(audio_parts)

    return final_audio, sample_rate


async def synthesize_chunk(
    tts,
    audio_queue,
    index,
    text,
):
    """
    Synthesize one response chunk and place
    it into the ordered playback queue.

    IMPORTANT: synthesis is offloaded to a worker thread via
    asyncio.to_thread. Running the blocking TTS generator directly
    inside this coroutine would freeze the event loop (and therefore
    mic capture + audio playback) for the entire duration of
    synthesis, defeating the purpose of pipelining chunks.

    On failure, we still `put()` an empty placeholder for this index.
    OrderedAudioQueue plays back strictly in index order, so silently
    skipping the put on error would permanently stall every later
    chunk waiting behind this one.
    """

    try:

        final_audio, sample_rate = await asyncio.to_thread(
            _synthesize_blocking,
            tts,
            text,
        )

        if final_audio is None:
            # Nothing synthesized (e.g. empty chunk) — still fill
            # the slot so the queue doesn't stall waiting for it.
            await audio_queue.put(index, np.array([], dtype=np.float32), 0)
            return

        await audio_queue.put(
            index,
            final_audio,
            sample_rate,
        )

    except Exception as e:

        print(
            f"\n❌ TTS chunk {index + 1} "
            f"error: {e}"
        )

        # Fill the slot on failure too, so playback doesn't deadlock
        # waiting on a chunk that will never arrive.
        try:
            await audio_queue.put(index, np.array([], dtype=np.float32), 0)
        except Exception as inner_e:
            print(
                f"\n❌ Failed to fill placeholder for chunk "
                f"{index + 1}: {inner_e}"
            )


async def process_voice_command(
    text,
    agent,
    memory,
    tts,
    audio_queue,
):
    """
    Process one complete voice command.

    LLM stream
        ↓
    Segmenter
        ↓
    Piper
        ↓
    Ordered audio queue
    """

    segmenter = ResponseSegmenter()

    tts_tasks = []

    chunk_index = 0

    # chunk indices restart at 0 for every command, but the shared
    # audio_queue's next_index only ever increases — without this
    # reset, every turn after the first would wait forever for an
    # index that never arrives. Safe here because the previous
    # turn's wait_until_finished() has already returned.
    await audio_queue.reset()

    print("\n🤔 Hasini is thinking...")

    try:

        async for token in Agent_stream(
            text,
            agent,
            memory,
        ):

            # ------------------------------------------
            # Feed token into segmenter
            # ------------------------------------------

            chunk = segmenter.add(token)

            if chunk:

                print(
                    f"\n🔊 Response chunk "
                    f"{chunk_index + 1}: "
                    f"{chunk}"
                )

                task = asyncio.create_task(
                    synthesize_chunk(
                        tts,
                        audio_queue,
                        chunk_index,
                        chunk,
                    )
                )

                tts_tasks.append(task)

                chunk_index += 1

        # ----------------------------------------------
        # Flush remaining text
        # ----------------------------------------------

        final_chunk = segmenter.flush()

        if final_chunk:

            print(
                f"\n🔊 Final response chunk: "
                f"{final_chunk}"
            )

            task = asyncio.create_task(
                synthesize_chunk(
                    tts,
                    audio_queue,
                    chunk_index,
                    final_chunk,
                )
            )

            tts_tasks.append(task)

    except Exception as e:

        print(
            f"\n❌ Agent/voice error: {e}"
        )

    finally:

        # ----------------------------------------------
        # Always wait for whatever TTS tasks were spawned,
        # even if the agent stream itself errored out —
        # otherwise they keep running in the background
        # untracked, and the ordered queue can end up with
        # audio still in flight when we reopen the mic.
        # ----------------------------------------------

        if tts_tasks:

            await asyncio.gather(
                *tts_tasks,
                return_exceptions=True,
            )

        # ----------------------------------------------
        # IMPORTANT:
        # Wait until all generated audio has actually
        # finished playing before opening the microphone.
        # ----------------------------------------------

        await audio_queue.wait_until_finished()


async def voice_loop(agent):

    print("\n🧠 Loading voice components...")

    vad = create_vad_provider()

    stt = create_stt_provider()

    tts = create_tts_provider()
    
    memory = ConversationMemory(
        history_limit=10,
        enabled=True,
    )

    audio_queue = OrderedAudioQueue()

    # --------------------------------------------------
    # Start speaker worker
    # --------------------------------------------------

    playback_task = asyncio.create_task(
        audio_queue.play_loop()
    )

    print("\n" + "=" * 50)
    print("🎤 Hasini Phase 2 Voice Mode")
    print("=" * 50)

    print(
        "\nSpeak naturally."
        "\nSilence will automatically end your command."
        "\nSay 'exit' or 'quit' to stop."
    )

    try:

        while True:

            # ------------------------------------------
            # Listen
            # ------------------------------------------

            audio = await asyncio.to_thread(
                record_until_silence,
                vad,
                MAX_RECORD_SECONDS,
            )

            if audio is None or len(audio) == 0:

                print(
                    "\n❌ No speech detected."
                )

                continue

            # ------------------------------------------
            # STT
            # ------------------------------------------

            print(
                "\n🧠 Transcribing..."
            )

            text = await asyncio.to_thread(
                stt.transcribe,
                audio,
            )

            if not text:

                print(
                    "\n❌ I couldn't understand that."
                )

                continue

            print(
                "\n" + "-" * 50
            )

            print(
                f"🎤 You: {text}"
            )

            print(
                "-" * 50
            )

            # ------------------------------------------
            # Exit
            # ------------------------------------------

            if is_exit_command(text):

                print(
                    "\n👋 Goodbye!"
                )

                break

            # ------------------------------------------
            # Process command
            # ------------------------------------------

            await process_voice_command(
                text,
                agent,
                memory,
                tts,
                audio_queue,
            )

    finally:

        # Wait for generated audio to finish
        await audio_queue.close()

        await playback_task


async def main():

    print("=" * 50)
    print("🤖 Hasini Phase 2 Voice Mode")
    print("=" * 50)

    print(
        "\nConnecting to agent..."
    )

    agent, mcp_client, all_tools = (
        await create_research_agent()
    )

    print(
        "✅ Agent connected."
    )

    print(
        "🎤 Voice mode ready."
    )

    print(
        "Press Ctrl+C anytime to exit."
    )

    try:

        await voice_loop(agent)

    except (
        KeyboardInterrupt,
        EOFError,
    ):

        print(
            "\n\n🛑 Voice mode interrupted."
        )

        print(
            "👋 Goodbye!"
        )

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

        print(
            "👋 Hasini stopped."
        )