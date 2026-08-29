from __future__ import annotations

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

    Optimized for low-latency voice commands while keeping the configured
    Whisper model/device/compute type unchanged.

    Important:
        The command listener already performs speech segmentation with VAD.
        Therefore Whisper-side VAD is disabled by default here to avoid doing
        a second VAD pass over already-segmented command audio.
    """

    def __init__(self):
        self.model = None

        logger.info("Loading Whisper model...")
        logger.info("Model: %s", WHISPER_MODEL)
        logger.info("Device: %s", WHISPER_DEVICE)
        logger.info("Compute type: %s", WHISPER_COMPUTE_TYPE)
        logger.info("CPU threads: %s", WHISPER_CPU_THREADS)

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

        The decoding configuration is intentionally optimized for interactive
        voice commands:

            beam_size=1
                Greedy decoding is considerably faster than beam search.

            best_of is omitted
                It is unnecessary for the fixed temperature=0.0 path and
                avoids additional decoding work/configuration.

            condition_on_previous_text=False
                Prevents previous decoded text from influencing the command.

            vad_filter=False
                CommandListener already performs VAD and supplies a completed
                speech segment. Running another VAD pass is redundant.

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

        transcription_started_at = time.perf_counter()

        try:
            segments, info = self.model.transcribe(
                audio,
                language="en",

                # ------------------------------------------------------
                # LOW-LATENCY DECODING
                # ------------------------------------------------------
                beam_size=1,
                temperature=0.0,

                # Do not carry previous transcription context into a
                # separate voice command.
                condition_on_previous_text=False,

                # ------------------------------------------------------
                # Domain vocabulary
                # ------------------------------------------------------
                initial_prompt=(
                    "Hasini, Python, LangChain, LangGraph, MCP, "
                    "Model Context Protocol, GitHub, Git, "
                    "OpenRouter, Ollama, FastAPI, filesystem, "
                    "AI agent, research agent, tool, tools, "
                    "Windows, PowerShell, "
                    "search, execute, create, delete, read, write, "
                    "quit, exit."
                ),

                # ------------------------------------------------------
                # The CommandListener has already detected speech start/end
                # and supplied a bounded command segment. Avoid a second VAD
                # pass here.
                # ------------------------------------------------------
                vad_filter=False,

                without_timestamps=True,
            )

            seg_list = list(segments)

            transcription_time = (
                time.perf_counter() - transcription_started_at
            )

            if not seg_list:
                logger.debug(
                    "Whisper returned no speech segments in %.3f seconds.",
                    transcription_time,
                )

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

            audio_duration = getattr(
                info,
                "duration",
                None,
            )

            if audio_duration is not None:
                logger.info(
                    "Whisper transcription completed in %.3fs "
                    "(audio=%.3fs, RTF=%.2fx, segments=%d, confidence=%.3f).",
                    transcription_time,
                    float(audio_duration),
                    (
                        transcription_time / float(audio_duration)
                        if float(audio_duration) > 0
                        else 0.0
                    ),
                    len(seg_list),
                    avg_confidence,
                )
            else:
                logger.info(
                    "Whisper transcription completed in %.3fs "
                    "(segments=%d, confidence=%.3f).",
                    transcription_time,
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
