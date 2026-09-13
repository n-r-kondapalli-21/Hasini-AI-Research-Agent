import subprocess
import sys

import sounddevice as sd
import soundfile as sf

from ..config import PIPER_MODEL, PIPER_OUTPUT_FILE_TEST


TEXT = "Hello. I am Hasini, your AI research assistant."


def speak(text: str):

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "piper",
            "--model",
            PIPER_MODEL,
            "--output_file",
            PIPER_OUTPUT_FILE_TEST,
        ],
        input=text,
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        print("❌ Piper error:")
        print(process.stderr)
        return

    audio, sample_rate = sf.read(PIPER_OUTPUT_FILE_TEST)

    sd.play(audio, sample_rate)
    sd.wait()

    print("✅ Speech playback complete.")


if __name__ == "__main__":
    speak(TEXT)