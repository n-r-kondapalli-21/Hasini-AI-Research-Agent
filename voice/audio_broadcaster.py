import asyncio
import logging


logger = logging.getLogger(__name__)


class AudioBroadcaster:
    """
    Broadcast every microphone frame to multiple
    independent consumers.

    Each consumer receives its own queue, so consumers
    never steal frames from one another.
    """

    def __init__(self, capture):
        self.capture = capture
        self.subscribers = []
        self.running = False
        self.task = None

    # --------------------------------------------------
    # Subscribe
    # --------------------------------------------------

    def subscribe(self, queue_size=100):
        queue = asyncio.Queue(maxsize=queue_size)

        self.subscribers.append(queue)

        logger.debug(
            "Audio subscriber added. Total subscribers: %d",
            len(self.subscribers),
        )

        return queue

    # --------------------------------------------------
    # Start
    # --------------------------------------------------

    async def start(self):
        if self.running:
            logger.debug("Audio broadcaster is already running.")
            return

        self.running = True

        try:
            self.task = asyncio.create_task(
                self._broadcast_loop()
            )
        except Exception:
            self.running = False
            logger.exception(
                "Failed to start audio broadcaster."
            )
            raise

        logger.info("Audio broadcaster started.")

    # --------------------------------------------------
    # Broadcast
    # --------------------------------------------------

    async def _broadcast_loop(self):
        try:
            while self.running:
                frame = await self.capture.get_frame()

                if frame is None:
                    logger.debug(
                        "Audio capture returned an empty frame."
                    )
                    continue

                for queue in list(self.subscribers):
                    if queue.full():
                        # Drop the oldest frame so a slow consumer
                        # cannot block the microphone pipeline.
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            logger.debug(
                                "Subscriber queue was empty while "
                                "attempting to drop an old frame."
                            )

                    try:
                        queue.put_nowait(frame.copy())

                    except asyncio.QueueFull:
                        # The queue became full between the full()
                        # check and put_nowait().
                        logger.debug(
                            "Dropped audio frame because subscriber "
                            "queue is full."
                        )

        except asyncio.CancelledError:
            logger.debug("Audio broadcaster task cancelled.")
            raise

        except Exception:
            logger.exception(
                "Audio broadcaster loop failed."
            )
            self.running = False
            raise

    # --------------------------------------------------
    # Stop
    # --------------------------------------------------

    async def stop(self):
        if not self.running:
            return

        self.running = False

        if self.task is not None:
            self.task.cancel()

            try:
                await self.task

            except asyncio.CancelledError:
                logger.debug(
                    "Audio broadcaster task cancelled during shutdown."
                )

            except Exception:
                logger.exception(
                    "Error while stopping audio broadcaster task."
                )

        self.task = None
        self.subscribers.clear()

        logger.info("Audio broadcaster stopped.")