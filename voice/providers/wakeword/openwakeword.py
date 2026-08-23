from pathlib import Path

import numpy as np

from openwakeword.model import Model

from ...config import (
    WAKEWORD_MODEL,
    WAKEWORD_THRESHOLD,
)

from .base import WakeWordProvider


class OpenWakeWordProvider(WakeWordProvider):
    """
    Local openWakeWord provider.

    All openWakeWord-specific logic is isolated here.
    """

    def __init__(self):

        print("\n🧠 Loading openWakeWord...")

        self.threshold = WAKEWORD_THRESHOLD

        wakeword_path = Path(
            WAKEWORD_MODEL
        )

        if not wakeword_path.is_file():

            raise FileNotFoundError(
                f"Wake word model not found:\n"
                f"{wakeword_path}"
            )

        model_directory = (
            wakeword_path.parent
        )

        self.melspectrogram_model = str(
            model_directory
            / "melspectrogram.onnx"
        )

        self.embedding_model = str(
            model_directory
            / "embedding_model.onnx"
        )

        for name, path in {
            "Mel spectrogram": self.melspectrogram_model,
            "Embedding": self.embedding_model,
        }.items():

            if not Path(path).is_file():

                raise FileNotFoundError(
                    f"{name} model not found:\n{path}"
                )

        self.model = Model(
            wakeword_models=[
                str(wakeword_path)
            ],
            inference_framework="onnx",
            melspec_model_path=self.melspectrogram_model,
            embedding_model_path=self.embedding_model,
        )

        print("✅ openWakeWord ready.")

        print(
            f"Wake word model: "
            f"{wakeword_path}"
        )

    def score_frame(self, audio_frame):

        if audio_frame is None:
            return 0.0

        if len(audio_frame) == 0:
            return 0.0

        audio = np.asarray(
            audio_frame,
            dtype=np.int16,
        )

        predictions = self.model.predict(audio)

        if not predictions:
            return 0.0

        return max(
            float(score)
            for score in predictions.values()
        )

    def is_detected(
        self,
        audio_frame: np.ndarray,
    ) -> bool:

        return (
            self.score_frame(audio_frame)
            >= self.threshold
        )

    def reset(self):

        self.model.reset()