import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ...config import (
    INPUT_SAMPLE_RATE,
    VAD_MODEL,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
)

from .base import VADProvider


logger = logging.getLogger(__name__)


class SileroVAD(VADProvider):
    """
    Local Silero VAD provider.

    Uses the project-local ONNX model and exposes
    speech start/end events independently of the
    underlying VAD implementation.
    """

    def __init__(self):
        self.model_path = Path(VAD_MODEL)

        self.threshold = VAD_THRESHOLD
        self.min_silence_ms = VAD_MIN_SILENCE_MS
        self.speech_pad_ms = VAD_SPEECH_PAD_MS

        self.session = None

        logger.info("Loading Silero VAD...")
        logger.info("Model: %s", self.model_path)

        self._validate_model()
        self._load_model()

        self.reset()

        logger.info("Silero VAD ready.")

    def _validate_model(self) -> None:
        """Validate that the configured VAD model exists."""
        if not self.model_path.is_file():
            logger.error(
                "Silero VAD model not found: %s",
                self.model_path,
            )
            raise FileNotFoundError(
                f"Silero VAD model not found:\n"
                f"{self.model_path}"
            )

    def _load_model(self) -> None:
        """Create the ONNX Runtime inference session."""
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        except Exception:
            logger.exception(
                "Failed to load Silero VAD model: %s",
                self.model_path,
            )
            raise

    def reset(self):
        """
        Reset model state and speech tracking.
        """

        self.h = np.zeros(
            (2, 1, 64),
            dtype=np.float32,
        )

        self.c = np.zeros(
            (2, 1, 64),
            dtype=np.float32,
        )

        self.speech_started = False
        self.silence_samples = 0
        self.last_probability = 0.0

    def _predict(self, audio_frame):
        """
        Run one inference step.
        """

        if self.session is None:
            raise RuntimeError(
                "Silero VAD session is not initialized."
            )

        try:
            audio = np.asarray(
                audio_frame,
                dtype=np.float32,
            )

            if audio.size == 0:
                return 0.0

            audio = audio.reshape(
                1,
                -1,
            )

            outputs = self.session.run(
                None,
                {
                    "input": audio,
                    "sr": np.array(
                        INPUT_SAMPLE_RATE,
                        dtype=np.int64,
                    ),
                    "h": self.h,
                    "c": self.c,
                },
            )

            probability = float(
                outputs[0][0][0]
            )

            self.h = outputs[1]
            self.c = outputs[2]

            self.last_probability = probability

            return probability

        except Exception:
            logger.exception(
                "Silero VAD inference failed."
            )
            raise

    def process(
        self,
        audio_frame,
        min_silence_ms: float | None = None,
    ):
        """
        Process an audio frame.

        Returns:

            {"start": ...}
                when speech starts

            {"end": ...}
                when speech ends

            {"speech": ...}
                while speech continues

            None
                when nothing changes
        """

        probability = self._predict(
            audio_frame
        )

        is_speech = (
            probability >= self.threshold
        )

        frame_samples = len(
            audio_frame
        )

        effective_min_silence = (
            min_silence_ms
            if min_silence_ms is not None
            else self.min_silence_ms
        )

        # ------------------------------------------
        # Speech started
        # ------------------------------------------

        if (
            is_speech
            and not self.speech_started
        ):
            self.speech_started = True
            self.silence_samples = 0

            logger.debug(
                "Speech started. Probability: %.3f",
                probability,
            )

            return {
                "start": True,
                "probability": probability,
            }

        # ------------------------------------------
        # Speech continues
        # ------------------------------------------

        if (
            is_speech
            and self.speech_started
        ):
            self.silence_samples = 0

            return {
                "speech": True,
                "probability": probability,
            }

        # ------------------------------------------
        # Silence while speaking
        # ------------------------------------------

        if (
            not is_speech
            and self.speech_started
        ):
            self.silence_samples += (
                frame_samples
            )

            silence_ms = (
                self.silence_samples
                * 1000
                / INPUT_SAMPLE_RATE
            )

            if (
                silence_ms
                >= effective_min_silence
            ):
                self.speech_started = False
                self.silence_samples = 0

                logger.debug(
                    "Speech ended. Probability: %.3f",
                    probability,
                )

                return {
                    "end": True,
                    "probability": probability,
                }

        return None

    def is_speech(
        self,
        audio_frame,
    ) -> bool:
        """
        Return whether the current frame contains
        speech according to the VAD threshold.
        """

        probability = self._predict(
            audio_frame
        )

        return (
            probability
            >= self.threshold
        )

    def process_probability(
        self,
        audio_frame: np.ndarray,
    ) -> float:
        """
        Return the raw Silero speech probability.

        Unlike process(), this does not apply the normal
        VAD_THRESHOLD. It is intended for the more
        conservative barge-in detector.
        """

        return self._predict(
            audio_frame
        )
