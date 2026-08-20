import subprocess
import sys
import os
import time

import numpy as np
import soundfile as sf

from piper import PiperVoice

from .config import (
    PIPER_MODEL,
    PIPER_OUTPUT_FILE,
)


class TextToSpeech:
    """
    Local Text-to-Speech using Piper.

    Supports:
    - Existing complete-response synthesis
    - Low-latency streaming synthesis
    """

    def __init__(self):

        self.model = PIPER_MODEL
        self.output_file = PIPER_OUTPUT_FILE

        output_directory = os.path.dirname(
            self.output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        if not os.path.isfile(self.model):
            raise FileNotFoundError(
                f"Piper model not found:\n{self.model}"
            )

        print("\n🔊 Loading Piper voice...")

        start = time.perf_counter()

        self.voice = PiperVoice.load(
            self.model
        )

        load_time = time.perf_counter() - start

        print(
            f"✅ Piper voice loaded in "
            f"{load_time:.2f} seconds."
        )

        print(
            f"Voice: {self.model}"
        )

    def synthesize(self, text: str):
        """
        Existing complete-response synthesis.

        Kept for compatibility with current code.
        """

        if not text or not text.strip():
            return None, None

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

        if process.returncode != 0:
            raise RuntimeError(
                f"Piper TTS failed:\n"
                f"{process.stderr}"
            )

        audio, sample_rate = sf.read(
            self.output_file,
            dtype="float32",
        )

        return audio, sample_rate

    def synthesize_stream(self, text: str):
        """
        Low-latency Piper synthesis.

        PiperVoice stays loaded in memory and
        yields audio chunks directly.
        """

        if not text or not text.strip():
            return

        for chunk in self.voice.synthesize(text):

            audio = chunk.audio_float_array

            if audio is None:
                continue

            if len(audio) == 0:
                continue

            yield audio, chunk.sample_rate