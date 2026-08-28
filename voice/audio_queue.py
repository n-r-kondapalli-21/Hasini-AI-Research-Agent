import asyncio
import logging

import numpy as np

from .audio_io import play_audio, stop_audio


logger = logging.getLogger(__name__)


class OrderedAudioQueue:
    """
    Ordered, cancellable, generation-aware audio playback.

    Playback is always sequential:

        chunk 0 → chunk 1 → chunk 2 → ...

    A response generation is invalidated when the user interrupts.
    Audio belonging to an old generation is never played.
    """

    def __init__(self):

        self.queue = {}

        self.condition = asyncio.Condition()

        self.next_index = 0

        self.closed = False

        self.pending_audio = 0

        self.playing = False

        self.playback_task = None

        # --------------------------------------------------
        # Generation control
        # --------------------------------------------------

        self.generation = 0

        self.cancel_event = asyncio.Event()

    # ============================================================
    # GENERATION
    # ============================================================

    def is_generation_active(
        self,
        generation: int,
    ) -> bool:

        return (
            generation == self.generation
            and not self.cancel_event.is_set()
        )

    # ============================================================
    # RESET
    # ============================================================

    async def reset(
        self,
        generation: int | None = None,
    ):
        """
        Prepare the queue for a new response.

        If generation is supplied, that generation becomes
        the active response generation.
        """

        # Stop any previous audio.
        stop_audio()

        async with self.condition:

            self.queue.clear()

            self.next_index = 0

            self.closed = False

            self.pending_audio = 0

            self.playing = False

            self.cancel_event.clear()

            if generation is not None:

                self.generation = generation

            self.condition.notify_all()

    # ============================================================
    # PUT
    # ============================================================

    async def put(
        self,
        index: int,
        audio: np.ndarray,
        sample_rate: int,
        generation: int,
    ):
        """
        Add audio to the queue only if it belongs to the
        currently active response generation.
        """

        if audio is None:

            audio = np.array(
                [],
                dtype=np.float32,
            )

        async with self.condition:

            # --------------------------------------------------
            # Reject old response generations.
            # --------------------------------------------------

            if generation != self.generation:

                return

            # --------------------------------------------------
            # Reject cancelled generation.
            # --------------------------------------------------

            if self.cancel_event.is_set():

                return

            # --------------------------------------------------
            # Reject closed queue.
            # --------------------------------------------------

            if self.closed:

                return

            self.queue[index] = (
                audio,
                sample_rate,
                generation,
            )

            self.pending_audio += 1

            self.condition.notify_all()

    # ============================================================
    # START PLAYBACK
    # ============================================================

    async def start_playback(self):

        # --------------------------------------------------
        # Reuse existing playback task if alive.
        # --------------------------------------------------

        if (
            self.playback_task is not None
            and not self.playback_task.done()
        ):

            return

        self.playback_task = asyncio.create_task(
            self.play_loop()
        )

        logger.debug("Audio playback loop started.")

    # ============================================================
    # PLAY LOOP
    # ============================================================

    async def play_loop(self):

        try:

            while True:

                async with self.condition:

                    # --------------------------------------------------
                    # Wait until next required chunk arrives or queue
                    # is closed.
                    # --------------------------------------------------

                    while (
                        self.next_index
                        not in self.queue
                        and not self.closed
                    ):

                        await self.condition.wait()

                    # --------------------------------------------------
                    # Queue closed and nothing left.
                    # --------------------------------------------------

                    if (
                        self.closed
                        and self.next_index
                        not in self.queue
                    ):

                        break

                    # --------------------------------------------------
                    # Required chunk still unavailable.
                    # --------------------------------------------------

                    if (
                        self.next_index
                        not in self.queue
                    ):

                        continue

                    # --------------------------------------------------
                    # Get next chunk.
                    # --------------------------------------------------

                    (
                        audio,
                        sample_rate,
                        generation,
                    ) = self.queue.pop(
                        self.next_index
                    )

                    current_index = (
                        self.next_index
                    )

                    self.next_index += 1

                    self.playing = True

                # ==================================================
                # PLAY OUTSIDE LOCK
                # ==================================================

                try:

                    # --------------------------------------------------
                    # Generation may have been cancelled while we
                    # were waiting for the lock.
                    # --------------------------------------------------

                    if not self.is_generation_active(
                        generation
                    ):

                        continue

                    if (
                        audio is not None
                        and len(audio) > 0
                    ):

                        logger.debug(
                            "Playing audio chunk %d.",
                            current_index + 1,
                        )

                        try:

                            await asyncio.to_thread(
                                play_audio,
                                audio,
                                sample_rate,
                            )

                            # ------------------------------------------------
                            # Only report completion if generation is
                            # still active.
                            # ------------------------------------------------

                            if self.is_generation_active(
                                generation
                            ):

                                logger.debug(
                                    "Finished audio chunk %d.",
                                    current_index + 1,
                                )

                        except Exception as e:

                            # ------------------------------------------------
                            # PortAudio can report an error when sd.stop()
                            # interrupts active playback.
                            #
                            # This is expected during cancellation.
                            # ------------------------------------------------

                            if self.is_generation_active(
                                generation
                            ):

                                logger.warning(
                                    "Audio playback error for chunk %d: %s",
                                    current_index + 1,
                                    e,
                                )

                finally:

                    async with self.condition:

                        self.pending_audio = max(
                            0,
                            self.pending_audio - 1,
                        )

                        self.playing = False

                        self.condition.notify_all()

        except asyncio.CancelledError:
            logger.debug("Audio playback loop cancelled.")
            raise

        finally:

            async with self.condition:

                self.playing = False

                self.condition.notify_all()

        logger.debug(
            "Audio queue reset for generation %s.",
            self.generation,
        )

    # ============================================================
    # WAIT
    # ============================================================

    async def wait_until_finished(
        self,
        generation: int | None = None,
    ):
        """
        Wait until all audio belonging to the active generation
        has finished.
        """

        async with self.condition:

            while True:

                # --------------------------------------------------
                # If this generation is no longer active, stop waiting.
                # --------------------------------------------------

                if (
                    generation is not None
                    and generation != self.generation
                ):

                    return

                if (
                    self.pending_audio == 0
                    and not self.playing
                    and not self.queue
                ):

                    return

                await self.condition.wait()

    # ============================================================
    # CANCEL
    # ============================================================

    async def cancel(
        self,
        generation: int | None = None,
    ):
        """
        Cancel the currently active response.

        IMPORTANT:
        The generation is invalidated immediately so old Piper
        tasks cannot submit audio for the next response.
        """

        logger.info("Cancelling audio playback...")

        async with self.condition:

            # --------------------------------------------------
            # If a newer response generation has already registered,
            # do not cancel the newer response generation.
            # --------------------------------------------------

            if (
                generation is not None
                and generation < self.generation
            ):

                return

            self.generation += 1

            self.cancel_event.set()

            # --------------------------------------------------
            # Immediately stop current speaker playback.
            # --------------------------------------------------

            stop_audio()

            # --------------------------------------------------
            # Remove all queued audio from old response.
            # --------------------------------------------------

            self.queue.clear()

            self.pending_audio = 0

            self.playing = False

            # --------------------------------------------------
            # Do NOT close permanently.
            #
            # The same queue will be reused for the next response.
            # --------------------------------------------------

            self.closed = False

            self.next_index = 0

            self.condition.notify_all()

        logger.debug(
            "Audio playback cancelled. Generation invalidated to %d.",
            self.generation,
        )

    # ============================================================
    # CLOSE
    # ============================================================

    async def close(
        self,
        generation: int | None = None,
    ):
        """
        Close playback for a completed generation.

        The queue can later be reset for a new generation.
        """

        async with self.condition:

            if (
                generation is not None
                and generation != self.generation
            ):

                return

            self.closed = True

            self.condition.notify_all()

        # --------------------------------------------------
        # Wait for playback task to finish.
        # --------------------------------------------------

        if self.playback_task is not None:

            try:

                await self.playback_task

            except asyncio.CancelledError:
                logger.debug(
                    "Audio playback task cancelled during close."
                )

            self.playback_task = None