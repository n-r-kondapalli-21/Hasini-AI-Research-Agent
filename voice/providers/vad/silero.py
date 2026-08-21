import numpy as np
import torch

from silero_vad import load_silero_vad, VADIterator

from ...config import (
    INPUT_SAMPLE_RATE,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
)

from .base import VADProvider


class SileroVAD(VADProvider):
    """
    Silero Voice Activity Detection provider.
    """

    def __init__(self):

        print("\n🧠 Loading Silero VAD...")

        torch.set_num_threads(1)

        self.model = load_silero_vad(
            onnx=False
        )

        self.vad = VADIterator(
            self.model,
            threshold=VAD_THRESHOLD,
            sampling_rate=INPUT_SAMPLE_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

        print("✅ Silero VAD ready.")

    def reset(self):

        self.vad.reset_states()

    def process(self, audio_frame: np.ndarray):

        audio_tensor = torch.from_numpy(
            audio_frame.astype(np.float32)
        )

        return self.vad(audio_tensor)

    def is_speech(self, audio_frame: np.ndarray) -> bool:

        result = self.process(audio_frame)

        return (
            result is not None
            and "start" in result
        )