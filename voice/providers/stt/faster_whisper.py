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

# Tune this from your logs — segments below this are treated as
# "didn't catch that" rather than dispatched to the agent.
LOW_CONFIDENCE_THRESHOLD = 0.4

# Segments with a no_speech_prob above this are likely noise/silence/
# breath picked up by the VAD, not real speech.
NO_SPEECH_PROB_THRESHOLD = 0.6


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

        Decoding configuration:

            beam_size=5, temperature fallback ladder
                Greedy decoding with a fixed temperature=0.0 has no recovery
                path when the top hypothesis is wrong — it just commits.
                Beam search plus a short temperature fallback lets Whisper
                retry with more randomness when its own confidence is low,
                which meaningfully cuts down on confident-but-wrong output.
                This costs some latency; for short command-length audio on
                CPU it's normally well under the cost of misrecognition.

            condition_on_previous_text=False
                Prevents previous decoded text from influencing the command.

            vad_filter=False
                CommandListener already performs VAD and supplies a completed
                speech segment. Running another VAD pass is redundant.

        Returns:
            {
                "text": str,
                "confidence": float,
                "no_speech_prob": float,
                "is_reliable": bool,
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
                # DECODING
                # ------------------------------------------------------
                beam_size=5,
                temperature=[0.0, 0.2, 0.4],

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
                    "quit, exit ,pause ,halt ,goodbye ,stop,  "
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
                    "no_speech_prob": 1.0,
                    "is_reliable": False,
                }

            text = " ".join(
                segment.text.strip()
                for segment in seg_list
            ).strip()

            confidences = []
            no_speech_probs = []

            for segment in seg_list:
                if (
                    hasattr(segment, "avg_logprob")
                    and segment.avg_logprob is not None
                ):
                    confidence = math.exp(segment.avg_logprob)
                    confidences.append(confidence)

                if (
                    hasattr(segment, "no_speech_prob")
                    and segment.no_speech_prob is not None
                ):
                    no_speech_probs.append(segment.no_speech_prob)

            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            )

            avg_confidence = max(
                0.0,
                min(1.0, float(avg_confidence)),
            )

            max_no_speech_prob = (
                max(no_speech_probs) if no_speech_probs else 0.0
            )

            is_reliable = (
                bool(text)
                and avg_confidence >= LOW_CONFIDENCE_THRESHOLD
                and max_no_speech_prob < NO_SPEECH_PROB_THRESHOLD
            )

            audio_duration = getattr(
                info,
                "duration",
                None,
            )

            if audio_duration is not None:
                logger.info(
                    "Whisper transcription completed in %.3fs "
                    "(audio=%.3fs, RTF=%.2fx, segments=%d, "
                    "confidence=%.3f, no_speech_prob=%.3f, reliable=%s).",
                    transcription_time,
                    float(audio_duration),
                    (
                        transcription_time / float(audio_duration)
                        if float(audio_duration) > 0
                        else 0.0
                    ),
                    len(seg_list),
                    avg_confidence,
                    max_no_speech_prob,
                    is_reliable,
                )
            else:
                logger.info(
                    "Whisper transcription completed in %.3fs "
                    "(segments=%d, confidence=%.3f, no_speech_prob=%.3f, "
                    "reliable=%s).",
                    transcription_time,
                    len(seg_list),
                    avg_confidence,
                    max_no_speech_prob,
                    is_reliable,
                )

            if not is_reliable:
                logger.warning(
                    "Low-confidence or noisy transcription rejected: "
                    "text=%r confidence=%.3f no_speech_prob=%.3f",
                    text,
                    avg_confidence,
                    max_no_speech_prob,
                )

            return {
                "text": text,
                "confidence": avg_confidence,
                "no_speech_prob": max_no_speech_prob,
                "is_reliable": is_reliable,
            }

        except Exception:
            logger.exception("Whisper transcription failed.")
            raise

    def transcribe(self, audio) -> str:
        """
        Transcribe audio and return only the recognized text.

        Returns an empty string if the transcription is judged unreliable
        (low confidence or likely non-speech), so the caller can prompt the
        user to repeat instead of dispatching bad text to the agent.
        """
        result = self.transcribe_with_confidence(audio)

        if not result["is_reliable"]:
            return ""

        return result["text"]