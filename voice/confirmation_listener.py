"""
Voice confirmation listener.

Handles confirmation prompts, recording, and transcription while reusing
the application's shared STT and TTS providers.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .audio_io import play_audio, record_until_silence
from .config import MAX_RECORD_SECONDS
from .providers.factory import create_tts_provider


logger = logging.getLogger("hasini.voice.confirmation")


class VoiceConfirmationListener:
    """Handle voice confirmation prompts, recording, and transcription."""

    def __init__(self, vad, stt, tts=None) -> None:
        self.vad = vad
        self.stt = stt

        if self.stt is None:
            raise ValueError(
                "A shared STT provider is required for voice confirmation."
            )

        if tts is not None:
            self.tts = tts
        else:
            try:
                self.tts = create_tts_provider()
            except Exception:
                logger.exception(
                    "Failed to initialize confirmation TTS provider."
                )
                raise

        logger.debug(
            "Voice confirmation listener initialized using shared STT provider."
        )

    async def speak(self, text: str) -> None:
        """Synthesize and play a confirmation prompt."""
        if not text or not text.strip():
            logger.debug("Skipping empty confirmation prompt.")
            return

        if self.tts is None:
            logger.warning(
                "Cannot speak confirmation: TTS provider is unavailable."
            )
            return

        def _synth() -> tuple[np.ndarray | None, int | None]:
            parts = []
            sample_rate = 22050

            try:
                for chunk, sr in self.tts.synthesize_stream(text):
                    if chunk is not None and len(chunk) > 0:
                        parts.append(chunk)
                        sample_rate = sr
            except Exception:
                logger.exception(
                    "TTS synthesis failed for confirmation prompt."
                )
                raise

            if not parts:
                return None, None

            return np.concatenate(parts), sample_rate

        try:
            audio, sample_rate = await asyncio.to_thread(_synth)

            if audio is None or len(audio) == 0:
                logger.warning(
                    "TTS produced no audio for confirmation prompt."
                )
                return

            await asyncio.to_thread(
                play_audio,
                audio,
                sample_rate,
            )

        except asyncio.CancelledError:
            logger.debug("Confirmation speech cancelled.")
            raise

        except Exception:
            logger.exception(
                "Failed to speak confirmation prompt."
            )

    async def listen(self) -> dict:
        """Record and transcribe the user's confirmation response."""
        logger.info("Listening for confirmation...")

        try:
            audio = await asyncio.to_thread(
                record_until_silence,
                self.vad,
                MAX_RECORD_SECONDS,
            )

        except asyncio.CancelledError:
            logger.debug("Confirmation recording cancelled.")
            raise

        except Exception:
            logger.exception(
                "Failed to record confirmation."
            )
            return {
                "text": "",
                "confidence": None,
            }

        if audio is None or len(audio) == 0:
            logger.warning(
                "No confirmation speech detected."
            )
            return {
                "text": "",
                "confidence": None,
            }

        try:
            if hasattr(
                self.stt,
                "transcribe_with_confidence",
            ):
                result = await asyncio.to_thread(
                    self.stt.transcribe_with_confidence,
                    audio,
                )
            else:
                result = await asyncio.to_thread(
                    self.stt.transcribe,
                    audio,
                )

        except asyncio.CancelledError:
            logger.debug(
                "Confirmation transcription cancelled."
            )
            raise

        except Exception:
            logger.exception(
                "Failed to transcribe confirmation."
            )
            return {
                "text": "",
                "confidence": None,
            }

        # Current STT interface may return plain text or a result dictionary.
        if isinstance(result, str):
            text = result.strip()

            logger.info(
                "Confirmation transcription: %s",
                text if text else "<empty>",
            )

            return {
                "text": text,
                "confidence": None,
            }

        if isinstance(result, dict):
            text = str(
                result.get("text", "") or ""
            ).strip()

            confidence = result.get("confidence")

            logger.info(
                "Confirmation transcription: %s",
                text if text else "<empty>",
            )

            return {
                "text": text,
                "confidence": confidence,
            }

        text = str(result).strip()

        logger.info(
            "Confirmation transcription: %s",
            text if text else "<empty>",
        )

        return {
            "text": text,
            "confidence": None,
        }
