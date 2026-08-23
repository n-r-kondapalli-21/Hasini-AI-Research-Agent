import asyncio

from ..audio_capture import AudioCapture
from ..audio_broadcaster import AudioBroadcaster


async def main():

    capture = AudioCapture()

    broadcaster = AudioBroadcaster(
        capture
    )

    wake_queue = (
        broadcaster.subscribe()
    )

    vad_queue = (
        broadcaster.subscribe()
    )

    await capture.start()

    await broadcaster.start()

    print(
        "\n🎤 Broadcaster test running..."
    )

    try:

        for i in range(20):

            wake_frame = (
                await wake_queue.get()
            )

            vad_frame = (
                await vad_queue.get()
            )

            print(
                f"Frame {i + 1}: "
                f"wake={len(wake_frame)}, "
                f"vad={len(vad_frame)}"
            )

    finally:

        await broadcaster.stop()

        await capture.stop()


if __name__ == "__main__":

    asyncio.run(main())