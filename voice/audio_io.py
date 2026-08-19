import numpy as np
import sounddevice as sd

from .config import (
    INPUT_SAMPLE_RATE,
    INPUT_CHANNELS,
    AUDIO_DEVICE,
)


def record_audio(duration: int) -> np.ndarray:
    """
    Record microphone audio for a fixed duration.

    Returns:
        numpy.float32 audio array.
    """

    print(f"\n🎤 Listening for {duration} seconds...")

    audio = sd.rec(
        int(duration * INPUT_SAMPLE_RATE),
        samplerate=INPUT_SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        dtype="float32",
        device=AUDIO_DEVICE,
    )

    sd.wait()

    print("✅ Recording complete.")

    return audio.flatten()


def play_audio(audio: np.ndarray, sample_rate: int):
    """
    Play an audio array through the default speaker.
    """

    if audio is None or len(audio) == 0:
        return

    sd.play(
        audio,
        sample_rate,
        device=AUDIO_DEVICE,
    )

    sd.wait()


def stop_audio():
    """
    Immediately stop audio playback.

    This will become important for Phase 3 barge-in.
    """

    sd.stop()