import logging
from pathlib import Path

import numpy as np
from openwakeword.model import Model

from ...config import (
    WAKEWORD_MODEL,
    WAKEWORD_THRESHOLD,
)

from .base import WakeWordProvider


logger = logging.getLogger(__name__)


class OpenWakeWordProvider(WakeWordProvider):
    """
    Local openWakeWord provider.

    All openWakeWord-specific logic is isolated here.
    """

    def __init__(self):
        self.threshold = WAKEWORD_THRESHOLD
        self.model = None

        wakeword_path = Path(WAKEWORD_MODEL)

        logger.info("Loading openWakeWord...")
        logger.info("Wake word model: %s", wakeword_path)

        self._validate_models(wakeword_path)
        self._load_model(
            wakeword_path
        )

        logger.info("openWakeWord ready.")

    def _validate_models(
        self,
        wakeword_path: Path,
    ) -> None:
        """Validate all required openWakeWord model files."""

        if not wakeword_path.is_file():
            logger.error(
                "Wake word model not found: %s",
                wakeword_path,
            )
            raise FileNotFoundError(
                f"Wake word model not found:\n"
                f"{wakeword_path}"
            )

        model_directory = wakeword_path.parent

        self.melspectrogram_model = str(
            model_directory
            / "melspectrogram.onnx"
        )

        self.embedding_model = str(
            model_directory
            / "embedding_model.onnx"
        )

        required_models = {
            "Mel spectrogram": self.melspectrogram_model,
            "Embedding": self.embedding_model,
        }

        for name, path in required_models.items():
            if not Path(path).is_file():
                logger.error(
                    "%s model not found: %s",
                    name,
                    path,
                )
                raise FileNotFoundError(
                    f"{name} model not found:\n"
                    f"{path}"
                )

    def _load_model(
        self,
        wakeword_path: Path,
    ) -> None:
        """Load the openWakeWord model."""

        try:
            self.model = Model(
                wakeword_models=[
                    str(wakeword_path)
                ],
                inference_framework="onnx",
                melspec_model_path=self.melspectrogram_model,
                embedding_model_path=self.embedding_model,
            )

        except Exception:
            logger.exception(
                "Failed to initialize openWakeWord model: %s",
                wakeword_path,
            )
            raise

    def score_frame(
        self,
        audio_frame,
    ) -> float:
        """Return the highest wake-word score for an audio frame."""

        if audio_frame is None:
            return 0.0

        if len(audio_frame) == 0:
            return 0.0

        if self.model is None:
            raise RuntimeError(
                "openWakeWord model is not initialized."
            )

        try:
            audio = np.asarray(
                audio_frame,
                dtype=np.int16,
            )

            predictions = self.model.predict(
                audio
            )

            if not predictions:
                return 0.0

            return max(
                float(score)
                for score in predictions.values()
            )

        except Exception:
            logger.exception(
                "openWakeWord inference failed."
            )
            raise

    def is_detected(
        self,
        audio_frame: np.ndarray,
    ) -> bool:
        """Return whether the wake-word threshold has been reached."""

        return (
            self.score_frame(audio_frame)
            >= self.threshold
        )

    def reset(self) -> None:
        """Reset the openWakeWord model state."""

        if self.model is None:
            logger.warning(
                "Cannot reset openWakeWord: model is not initialized."
            )
            return

        try:
            self.model.reset()
        except Exception:
            logger.exception(
                "Failed to reset openWakeWord model."
            )
            raise