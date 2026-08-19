import subprocess
import sys
import os

import soundfile as sf

from .config import (PIPER_MODEL,PIPER_OUTPUT_FILE,)


class TextToSpeech:
    """
    Local Text-to-Speech using Piper.
    """

    def __init__(self):

        self.model = PIPER_MODEL
        self.output_file = PIPER_OUTPUT_FILE

        # Create output directory if it doesn't exist
        output_directory = os.path.dirname(
            self.output_file
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        # Verify Piper model exists
        if not os.path.isfile(self.model):
            raise FileNotFoundError(
                f"Piper model not found:\n{self.model}"
            )

        print("\n🔊 Piper TTS ready.")
        print(f"Voice: {self.model}")

    def synthesize(self, text: str):

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
                f"Piper TTS failed:\n{process.stderr}"
            )

        audio, sample_rate = sf.read(
            self.output_file,
            dtype="float32",
        )

        return audio, sample_rate