import asyncio
import logging
import threading

import sounddevice as sd

from .config import (
    INPUT_SAMPLE_RATE,
    INPUT_CHANNELS,
    AUDIO_DEVICE,
    VAD_FRAME_SAMPLES,
)


logger = logging.getLogger(__name__)


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
            logger.debug(
                "Audio capture is already running."
            )
            return

        self.loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_thread,
            daemon=True,
            name="HasiniAudioCapture",
        )

        try:
            self.thread.start()
        except Exception:
            self.running = False
            self.thread = None

            logger.exception(
                "Failed to start microphone capture thread."
            )
            raise

        logger.info(
            "Continuous microphone capture started."
        )

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

            logger.debug(
                "Microphone stream opened successfully."
            )

            while not self._stop_event.is_set():
                audio, overflowed = self.stream.read(
                    VAD_FRAME_SAMPLES
                )

                if overflowed:
                    logger.warning(
                        "Microphone buffer overflow."
                    )

                frame = audio.flatten().copy()

                self._publish(frame)

        except Exception:
            logger.exception(
                "Audio capture thread failed."
            )
            self.running = False

        finally:
            self._close_stream()
            self.running = False

    # --------------------------------------------------
    # Stream cleanup
    # --------------------------------------------------

    def _close_stream(self):
        """Stop and close the microphone stream safely."""

        if self.stream is None:
            return

        try:
            self.stream.stop()
        except Exception:
            logger.warning(
                "Failed to stop microphone stream.",
                exc_info=True,
            )

        try:
            self.stream.close()
        except Exception:
            logger.warning(
                "Failed to close microphone stream.",
                exc_info=True,
            )

        self.stream = None

    # --------------------------------------------------
    # Publish frame into asyncio
    # --------------------------------------------------

    def _publish(self, frame):
        if self.loop is None:
            return

        try:
            self.loop.call_soon_threadsafe(
                self._put_frame,
                frame,
            )
        except RuntimeError:
            logger.debug(
                "Unable to publish audio frame because "
                "the asyncio event loop is no longer running."
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
                logger.debug(
                    "Audio queue was empty while dropping old frame."
                )

        try:
            self.queue.put_nowait(frame)

        except asyncio.QueueFull:
            # Queue may become full between full() and put_nowait().
            logger.debug(
                "Dropped audio frame because the queue is full."
            )

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

        logger.info(
            "Stopping microphone capture..."
        )

        self.running = False
        self._stop_event.set()

        if self.thread is not None:
            try:
                await asyncio.to_thread(
                    self.thread.join,
                    2.0,
                )
            except Exception:
                logger.exception(
                    "Error while waiting for microphone "
                    "capture thread to stop."
                )

        self.thread = None
        self._clear_queue()

        logger.info(
            "Microphone capture stopped."
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