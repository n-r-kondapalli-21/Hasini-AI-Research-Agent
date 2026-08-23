import asyncio
import numpy as np

from ..audio_queue import OrderedAudioQueue


async def main():

    queue = OrderedAudioQueue()

    # --------------------------------------------------
    # Start playback loop
    # --------------------------------------------------

    playback_task = asyncio.create_task(
        queue.play_loop()
    )

    # --------------------------------------------------
    # Generate a long test tone
    # --------------------------------------------------

    sample_rate = 22050

    duration = 10

    frequency = 440

    samples = int(
        sample_rate * duration
    )

    t = np.arange(samples) / sample_rate

    audio = (
        0.2
        * np.sin(
            2 * np.pi * frequency * t
        )
    ).astype(
        np.float32
    )

    # --------------------------------------------------
    # Queue audio
    # --------------------------------------------------

    await queue.put(
        index=0,
        audio=audio,
        sample_rate=sample_rate,
    )

    print(
        "\n" + "=" * 50
    )

    print(
        "🔊 Audio cancellation test"
    )

    print(
        "=" * 50
    )

    print(
        "\nYou should hear a 440 Hz tone."
    )

    print(
        "It should automatically stop after 2 seconds."
    )

    # --------------------------------------------------
    # Let audio play
    # --------------------------------------------------

    await asyncio.sleep(2)

    print(
        "\n🛑 Cancelling playback..."
    )

    # --------------------------------------------------
    # Cancel
    # --------------------------------------------------

    await queue.cancel()

    print(
        "✅ Cancellation requested."
    )

    # Give sounddevice/playback thread time
    # to finish its cleanup.
    await asyncio.sleep(0.5)

    # --------------------------------------------------
    # Stop playback loop
    # --------------------------------------------------

    await queue.close()

    await playback_task

    print(
        "✅ Audio cancellation test finished."
    )


if __name__ == "__main__":

    asyncio.run(main())