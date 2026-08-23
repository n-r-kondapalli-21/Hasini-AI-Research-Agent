import asyncio

from .audio_io import play_audio


class CancellableAudioPlayer:
    """
    Plays audio while allowing the playback task to be
    cancelled by the voice controller.
    """

    def __init__(self):

        self.current_task = None

    async def play(
        self,
        audio,
        sample_rate,
    ):

        self.current_task = (
            asyncio.current_task()
        )

        try:

            await asyncio.to_thread(
                play_audio,
                audio,
                sample_rate,
            )

        except asyncio.CancelledError:

            print(
                "🔇 Audio playback cancelled."
            )

            raise

        finally:

            self.current_task = None

    async def cancel(self):

        if (
            self.current_task is not None
            and not self.current_task.done()
        ):

            self.current_task.cancel()

            try:

                await self.current_task

            except asyncio.CancelledError:

                pass

            self.current_task = None