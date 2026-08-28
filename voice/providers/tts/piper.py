import logging
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from piper import PiperVoice

from ...config import (
    PIPER_MODEL,
    PIPER_OUTPUT_FILE,
)

from .base import TTSProvider


logger = logging.getLogger(__name__)


class PiperTTS(TTSProvider):
    """
    Piper Text-to-Speech provider.

    Supports both file-based synthesis and streaming synthesis.
    """

    def __init__(self):
        self.model = PIPER_MODEL
        self.output_file = PIPER_OUTPUT_FILE
        self.voice = None

        self._prepare_output_directory()
        self._validate_model()
        self._load_voice()

    def _prepare_output_directory(self) -> None:
        """Create the TTS output directory if it does not exist."""
        output_directory = os.path.dirname(self.output_file)

        if not output_directory:
            return

        try:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )
        except OSError:
            logger.exception(
                "Failed to create Piper output directory: %s",
                output_directory,
            )
            raise

    def _validate_model(self) -> None:
        """Verify that the configured Piper model exists."""
        if not os.path.isfile(self.model):
            logger.error(
                "Piper model not found: %s",
                self.model,
            )
            raise FileNotFoundError(
                f"Piper model not found:\n{self.model}"
            )

    def _load_voice(self) -> None:
        """Load the Piper voice model."""
        logger.info("Loading Piper voice...")
        logger.info("Voice: %s", self.model)

        start_time = time.perf_counter()

        try:
            self.voice = PiperVoice.load(self.model)

        except Exception:
            load_time = time.perf_counter() - start_time

            logger.exception(
                "Failed to load Piper voice '%s' after %.2f seconds.",
                self.model,
                load_time,
            )
            raise

        load_time = time.perf_counter() - start_time

        logger.info(
            "Piper voice loaded successfully in %.2f seconds.",
            load_time,
        )

    def synthesize(self, text: str):
        """
        Synthesize text into an audio array.

        Returns:
            tuple:
                (audio, sample_rate)

        Returns (None, None) when the input text is empty.
        """

        if not text or not text.strip():
            logger.debug("Skipping TTS synthesis because text is empty.")
            return None, None

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "piper",
                    "--model",
                    self.model,
                    "--output_file",
                    self.output_file,
                ],
                input=text,
                text=True,
                capture_output=True,
            )

        except OSError:
            logger.exception(
                "Failed to start Piper TTS process."
            )
            raise

        if process.returncode != 0:
            stderr = process.stderr.strip()

            logger.error(
                "Piper TTS process failed with exit code %d. Error: %s",
                process.returncode,
                stderr or "No error output.",
            )

            raise RuntimeError(
                f"Piper TTS failed:\n"
                f"{stderr or 'No error output.'}"
            )

        try:
            audio, sample_rate = sf.read(
                self.output_file,
                dtype="float32",
            )

        except Exception:
            logger.exception(
                "Failed to read generated Piper audio file: %s",
                self.output_file,
            )
            raise

        logger.debug(
            "Piper synthesis completed successfully. "
            "Sample rate: %s",
            sample_rate,
        )

        return audio, sample_rate

    def synthesize_stream(self, text: str):
        """
        Stream synthesized audio chunks from the loaded Piper voice.
        """

        if not text or not text.strip():
            logger.debug(
                "Skipping streaming TTS because text is empty."
            )
            return

        if self.voice is None:
            raise RuntimeError(
                "Piper voice is not initialized."
            )

        try:
            for chunk in self.voice.synthesize(text):
                audio = chunk.audio_float_array

                if audio is None:
                    continue

                if len(audio) == 0:
                    continue

                yield audio, chunk.sample_rate

        except Exception:
            logger.exception(
                "Piper streaming synthesis failed."
            )
            raise

