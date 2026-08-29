
from __future__ import annotations

import asyncio
import logging
import os
import random
import wave

import numpy as np

from .audio_io import play_audio


logger = logging.getLogger("hasini.voice.exit_acknowledgement")


class ExitAcknowledgement:
    """Play one randomly selected spoken exit acknowledgement."""

    EXIT_FILES = [
        ("end_goodbye.wav", "Goodbye."),
        ("end_goodbye_take_care.wav", "Goodbye, take care."),
        ("end_see_you.wav", "See you later."),
        ("end_session.wav", "Ending the session. Goodbye."),
    ]

    def __init__(self):
        self.sounds = []
        self._load_sounds()

    def _load_sounds(self) -> None:
        """Load available ending acknowledgement WAV files."""

        assets_dir = os.path.join(
            os.path.dirname(__file__),
            "assets",
        )

        for filename, phrase in self.EXIT_FILES:
            filepath = os.path.join(
                assets_dir,
                filename,
            )

            if not os.path.isfile(filepath):
                logger.warning(
                    "Exit acknowledgement asset missing: %s",
                    filepath,
                )
                continue

            try:
                with wave.open(filepath, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(
                        wav_file.getnframes()
                    )

                audio = (
                    np.frombuffer(
                        frames,
                        dtype=np.int16,
                    )
                    .astype(np.float32)
                    / 32768.0
                )

                if audio.size == 0:
                    logger.warning(
                        "Exit acknowledgement is empty: %s",
                        filepath,
                    )
                    continue

                self.sounds.append(
                    (
                        audio,
                        sample_rate,
                        phrase,
                    )
                )

                logger.debug(
                    "Loaded exit acknowledgement: %s",
                    filename,
                )

            except Exception:
                logger.exception(
                    "Failed to load exit acknowledgement: %s",
                    filepath,
                )

    async def play(self) -> None:
        """Play one randomly selected ending acknowledgement."""

        if not self.sounds:
            logger.warning(
                "No exit acknowledgement sounds are available."
            )
            return

        audio, sample_rate, phrase = random.choice(
            self.sounds
        )

        logger.info(
            'Exit acknowledgement: "%s"',
            phrase,
        )

        try:
            await asyncio.to_thread(
                play_audio,
                audio,
                sample_rate,
            )

        except Exception:
            logger.exception(
                "Exit acknowledgement playback failed."
            )
