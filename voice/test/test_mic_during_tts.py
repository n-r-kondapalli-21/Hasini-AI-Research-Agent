import asyncio
import numpy as np

from ..audio_capture import AudioCapture
from ..audio_broadcaster import AudioBroadcaster


async def main():

    capture = AudioCapture()

    broadcaster = AudioBroadcaster(
        capture
    )

    queue = broadcaster.subscribe()

    await capture.start()
    await broadcaster.start()

    print("\n" + "=" * 50)
    print("🎤 Microphone TTS Leakage Test")
    print("=" * 50)

    print(
        "\nSpeak nothing."
    )

    print(
        "Play Hasini/TTS audio normally."
    )

    print(
        "Watching microphone RMS..."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    try:

        while True:

            frame = await queue.get()

            frame = np.asarray(
                frame,
                dtype=np.float32,
            )

            if len(frame) == 0:
                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(frame)
                    )
                )
            )

            peak = float(
                np.max(
                    np.abs(frame)
                )
            )

            print(
                f"Mic RMS: {rms:.4f} | "
                f"Peak: {peak:.4f}"
            )

    finally:

        await broadcaster.stop()

        await capture.stop()


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print(
            "\n🛑 Test stopped."
        )