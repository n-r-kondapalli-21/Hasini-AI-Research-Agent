import asyncio


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

        queue = asyncio.Queue(
            maxsize=queue_size
        )

        self.subscribers.append(
            queue
        )

        return queue

    # --------------------------------------------------
    # Start
    # --------------------------------------------------

    async def start(self):

        if self.running:
            return

        self.running = True

        self.task = asyncio.create_task(
            self._broadcast_loop()
        )

        print(
            "📡 Audio broadcaster started."
        )

    # --------------------------------------------------
    # Broadcast
    # --------------------------------------------------

    async def _broadcast_loop(self):

        try:

            while self.running:

                frame = await (
                    self.capture.get_frame()
                )

                for queue in list(
                    self.subscribers
                ):

                    if queue.full():

                        # Drop the oldest frame so
                        # a slow consumer cannot block
                        # the microphone pipeline.

                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                    try:

                        queue.put_nowait(
                            frame.copy()
                        )

                    except asyncio.QueueFull:

                        pass

        except asyncio.CancelledError:

            pass

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
                pass

        self.task = None

        self.subscribers.clear()

        print(
            "📡 Audio broadcaster stopped."
        )