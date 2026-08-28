import asyncio
import logging

from .audio_io import play_audio


logger = logging.getLogger(__name__)


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
        self.current_task = asyncio.current_task()

        try:
            await asyncio.to_thread(
                play_audio,
                audio,
                sample_rate,
            )

        except asyncio.CancelledError:
            logger.info("Audio playback cancelled.")
            raise

        except Exception:
            logger.exception(
                "Audio playback failed."
            )
            raise

        finally:
            self.current_task = None

    async def cancel(self):
        if (
            self.current_task is None
            or self.current_task.done()
        ):
            return

        logger.debug("Cancelling audio playback task.")

        self.current_task.cancel()

        try:
            await self.current_task

        except asyncio.CancelledError:
            logger.debug(
                "Audio playback task cancelled successfully."
            )

        except Exception:
            logger.exception(
                "Error while cancelling audio playback task."
            )

        finally:
            self.current_task = None