import logging
import math
import time

from faster_whisper import WhisperModel

from ...config import (
    WHISPER_MODEL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_CPU_THREADS,
)

from .base import STTProvider


logger = logging.getLogger(__name__)


class FasterWhisperSTT(STTProvider):
    """
    faster-whisper Speech-to-Text provider.

    Loads the configured Whisper model and provides transcription
    with an optional confidence score.
    """

    def __init__(self):
        self.model = None

        logger.info("Loading Whisper model...")
        logger.info("Model: %s", WHISPER_MODEL)

        start_time = time.perf_counter()

        try:
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=WHISPER_CPU_THREADS,
                num_workers=1,
            )

        except Exception:
            load_time = time.perf_counter() - start_time

            logger.exception(
                "Failed to load Whisper model '%s' after %.2f seconds.",
                WHISPER_MODEL,
                load_time,
            )

            raise

        load_time = time.perf_counter() - start_time

        logger.info(
            "Whisper loaded successfully in %.2f seconds.",
            load_time,
        )

    def transcribe_with_confidence(self, audio) -> dict:
        """
        Transcribe audio and return text with an average confidence score.

        Returns:
            {
                "text": str,
                "confidence": float,
            }

        Raises:
            RuntimeError: If the Whisper model is not initialized.
            Exception: If transcription fails.
        """

        if self.model is None:
            raise RuntimeError(
                "Whisper model is not initialized."
            )

        try:
            segments, info = self.model.transcribe(
                audio,
                language="en",
                beam_size=8,
                best_of=5,
                temperature=0.0,
                initial_prompt=(
                    "Hasini, Python, LangChain, LangGraph, MCP, "
                    "Model Context Protocol, GitHub, Git, "
                    "OpenRouter, Ollama, FastAPI, filesystem, "
                    "AI agent, research agent, tool, tools, "
                    "Windows, PowerShell, "
                    "search, execute, create, delete, read, write, quit, exit."
                ),
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                },
                without_timestamps=True,
            )

            seg_list = list(segments)

            if not seg_list:
                logger.debug("Whisper returned no speech segments.")
                return {
                    "text": "",
                    "confidence": 0.0,
                }

            text = " ".join(
                segment.text.strip()
                for segment in seg_list
            ).strip()

            confidences = []

            for segment in seg_list:
                if (
                    hasattr(segment, "avg_logprob")
                    and segment.avg_logprob is not None
                ):
                    confidence = math.exp(segment.avg_logprob)
                    confidences.append(confidence)

            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            )

            avg_confidence = max(
                0.0,
                min(1.0, float(avg_confidence)),
            )

            logger.debug(
                "Transcription completed: %d segment(s), confidence=%.3f",
                len(seg_list),
                avg_confidence,
            )

            return {
                "text": text,
                "confidence": avg_confidence,
            }

        except Exception:
            logger.exception("Whisper transcription failed.")
            raise

    def transcribe(self, audio) -> str:
        """
        Transcribe audio and return only the recognized text.
        """

        result = self.transcribe_with_confidence(audio)
        return result["text"]