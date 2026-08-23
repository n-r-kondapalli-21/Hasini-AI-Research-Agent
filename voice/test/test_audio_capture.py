import asyncio

from ..audio_capture import AudioCapture


async def main():

    capture = AudioCapture()

    await capture.start()

    print("\n🎤 Audio capture test running.")
    print("Speak for a few seconds.")
    print("Press Ctrl+C to stop.\n")

    try:

        for i in range(100):

            frame = await capture.get_frame()

            print(
                f"Frame {i + 1}: "
                f"{len(frame)} samples"
            )

    finally:

        await capture.stop()


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print(
            "\n🛑 Test interrupted."
        )