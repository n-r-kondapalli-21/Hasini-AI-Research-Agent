import asyncio
import threading

import sounddevice as sd

from .config import (
    INPUT_SAMPLE_RATE,
    INPUT_CHANNELS,
    AUDIO_DEVICE,
    VAD_FRAME_SAMPLES,
)


class AudioCapture:
    """
    Owns exactly one microphone stream.

    Audio is captured continuously in a dedicated thread
    and distributed through an asyncio queue.

    Consumers can later receive the same frames for:
        - wake-word detection
        - VAD
        - barge-in detection
        - other audio processing
    """

    def __init__(self, queue_size=100):

        self.queue = asyncio.Queue(
            maxsize=queue_size
        )

        self.stream = None

        self.loop = None

        self.thread = None

        self.running = False

        self._stop_event = threading.Event()

    # --------------------------------------------------
    # Start
    # --------------------------------------------------

    async def start(self):

        if self.running:
            return

        self.loop = asyncio.get_running_loop()

        self._stop_event.clear()

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_thread,
            daemon=True,
            name="HasiniAudioCapture",
        )

        self.thread.start()

        print("🎤 Continuous microphone capture started.")

    # --------------------------------------------------
    # Capture thread
    # --------------------------------------------------

    def _capture_thread(self):

        try:

            self.stream = sd.InputStream(
                samplerate=INPUT_SAMPLE_RATE,
                channels=INPUT_CHANNELS,
                dtype="float32",
                device=AUDIO_DEVICE,
                blocksize=VAD_FRAME_SAMPLES,
            )

            self.stream.start()

            while not self._stop_event.is_set():

                audio, overflowed = (
                    self.stream.read(
                        VAD_FRAME_SAMPLES
                    )
                )

                if overflowed:

                    print(
                        "⚠️ Microphone buffer overflow."
                    )

                frame = audio.flatten().copy()

                self._publish(frame)

        except Exception as e:

            print(
                f"\n❌ Audio capture error: {e}"
            )

            self.running = False

        finally:

            if self.stream is not None:

                try:
                    self.stream.stop()
                except Exception:
                    pass

                try:
                    self.stream.close()
                except Exception:
                    pass

                self.stream = None

            self.running = False

    # --------------------------------------------------
    # Publish frame into asyncio
    # --------------------------------------------------

    def _publish(self, frame):

        if self.loop is None:
            return

        self.loop.call_soon_threadsafe(
            self._put_frame,
            frame,
        )

    def _put_frame(self, frame):

        if not self.running:
            return

        if self.queue.full():

            # Drop the oldest frame rather than allowing
            # microphone capture to block indefinitely.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        try:

            self.queue.put_nowait(
                frame
            )

        except asyncio.QueueFull:
            pass

    # --------------------------------------------------
    # Read one frame
    # --------------------------------------------------

    async def get_frame(self):

        return await self.queue.get()

    # --------------------------------------------------
    # Stop
    # --------------------------------------------------

    async def stop(self):

        if not self.running:
            return

        print(
            "🛑 Stopping microphone capture..."
        )

        self.running = False

        self._stop_event.set()

        if self.thread is not None:

            await asyncio.to_thread(
                self.thread.join,
                2.0,
            )

        self.thread = None

        self._clear_queue()

        print(
            "🎤 Microphone capture stopped."
        )

    # --------------------------------------------------
    # Queue cleanup
    # --------------------------------------------------

    def _clear_queue(self):

        while True:

            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break