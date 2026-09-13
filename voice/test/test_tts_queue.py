import asyncio
import time

from ..providers.factory import create_tts_provider
from ..audio_queue import OrderedAudioQueue


async def synthesize_chunk(
    tts,
    player,
    index,
    text,
):
    print(
        f"\n🧠 START TTS chunk {index + 1}"
    )

    start = time.perf_counter()

    audio_parts = []

    for audio, sample_rate in (
        tts.synthesize_stream(text)
    ):
        audio_parts.append(
            (audio, sample_rate)
        )

    latency = (
        time.perf_counter() - start
    )

    print(
        f"⚡ Chunk {index + 1} generated "
        f"in {latency:.3f}s"
    )

    # Combine Piper audio pieces.
    if not audio_parts:
        return

    audio_arrays = [
        audio
        for audio, _ in audio_parts
    ]

    audio = audio_arrays[0]

    if len(audio_arrays) > 1:

        import numpy as np

        audio = np.concatenate(
            audio_arrays
        )

    sample_rate = audio_parts[0][1]

    await player.put(
        index,
        audio,
        sample_rate,
    )

    print(
        f"✅ Chunk {index + 1} queued"
    )


async def main():

    print("=" * 60)
    print("PHASE 2.4 - ORDERED TTS TEST")
    print("=" * 60)

    tts = create_tts_provider()

    player = OrderedAudioQueue()

    playback_task = asyncio.create_task(
        player.play_loop()
    )

    chunks = [
        "This is the first sentence. "
        "It must always play first.",

        "This is the second sentence. "
        "It must always play after the first.",

        "This is the third sentence. "
        "It must always play after the second.",
    ]

    # --------------------------------------------------
    # Generate chunks concurrently
    # --------------------------------------------------

    tasks = []

    for index, text in enumerate(chunks):

        tasks.append(
            asyncio.create_task(
                synthesize_chunk(
                    tts,
                    player,
                    index,
                    text,
                )
            )
        )

    await asyncio.gather(*tasks)

    print(
        "\n⏳ Closing audio queue..."
    )

    await player.close()

    await playback_task

    print(
        "\n" + "=" * 60
    )

    print(
        "PHASE 2.4 ORDERED TEST COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    asyncio.run(main())