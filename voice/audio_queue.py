import asyncio
import numpy as np

from .audio_io import play_audio


class OrderedAudioQueue:
    """
    Ordered audio playback.

    Playback always occurs in sequence order.
    """

    def __init__(self):

        self.queue = {}

        self.condition = asyncio.Condition()

        self.next_index = 0

        self.closed = False

        self.pending_audio = 0

        self.playing = False

    async def reset(self):
        """
        Reset playback indexing for a new turn.

        `next_index` only ever increases and chunk indices restart
        at 0 for every new voice command (see process_voice_command),
        so without this, the second and every later turn would wait
        forever for an index that will never be `put()` — this must
        be called once at the start of each new command, after the
        previous turn's `wait_until_finished()` has returned.
        """

        async with self.condition:

            self.queue.clear()

            self.next_index = 0

            self.pending_audio = 0

            self.playing = False

            self.condition.notify_all()

    async def put(
        self,
        index: int,
        audio: np.ndarray,
        sample_rate: int,
    ):
        """
        Register a chunk's audio at its index.

        IMPORTANT: even an empty/failed chunk must still occupy its
        slot in `self.queue`. `play_loop` advances strictly in order
        by index — silently dropping an empty chunk here (instead of
        recording an empty placeholder) leaves a permanent hole that
        `next_index` can never move past, deadlocking playback for
        every chunk queued after it.
        """

        if audio is None:
            audio = np.array([], dtype=np.float32)

        async with self.condition:

            self.queue[index] = (
                audio,
                sample_rate,
            )

            self.pending_audio += 1

            self.condition.notify_all()

    async def play_loop(self):
        """
        Play chunks strictly in order.
        """

        while True:

            async with self.condition:

                while (
                    self.next_index not in self.queue
                    and not self.closed
                ):
                    await self.condition.wait()

                if (
                    self.closed
                    and self.next_index not in self.queue
                    and self.pending_audio == 0
                ):
                    break

                if self.next_index not in self.queue:
                    continue

                audio, sample_rate = self.queue.pop(
                    self.next_index
                )

                current_index = self.next_index

                self.next_index += 1

                self.playing = True

            try:

                if audio is not None and len(audio) > 0:

                    print(
                        f"🔊 Playing chunk {current_index + 1}"
                    )

                    await asyncio.to_thread(
                        play_audio,
                        audio,
                        sample_rate,
                    )

            finally:

                async with self.condition:

                    self.pending_audio -= 1

                    self.playing = False

                    self.condition.notify_all()

    async def wait_until_finished(self):
        """
        Wait until every queued audio chunk has
        completely finished playing.
        """

        async with self.condition:

            while (
                self.pending_audio > 0
                or self.playing
                or self.queue
            ):
                await self.condition.wait()

    async def close(self):

        async with self.condition:

            self.closed = True

            self.condition.notify_all()