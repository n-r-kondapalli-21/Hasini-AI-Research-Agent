"""
Voice provider factory.

Creates the configured STT, TTS, VAD, and wake-word providers.
Provider selection behavior is preserved while failures are logged
with useful context before being propagated to the caller.
"""

from __future__ import annotations

import logging

from .stt.faster_whisper import FasterWhisperSTT
from .tts.piper import PiperTTS
from .vad.silero import SileroVAD
from .wakeword.openwakeword import OpenWakeWordProvider

from ..config import (
    STT_PROVIDER,
    TTS_PROVIDER,
    VAD_PROVIDER,
    WAKEWORD_PROVIDER,
)


logger = logging.getLogger("voice.providers.factory")


def create_stt_provider():
    """Create and return the configured STT provider."""
    logger.info(
        "Initializing STT provider: %s",
        STT_PROVIDER,
    )

    if STT_PROVIDER != "faster_whisper":
        logger.error(
            "Unsupported STT provider configured: %s",
            STT_PROVIDER,
        )
        raise ValueError(
            f"Unsupported STT provider: {STT_PROVIDER}"
        )

    try:
        provider = FasterWhisperSTT()
    except Exception:
        logger.exception(
            "Failed to initialize STT provider: %s",
            STT_PROVIDER,
        )
        raise

    logger.info(
        "STT provider ready: %s",
        STT_PROVIDER,
    )
    return provider


def create_tts_provider():
    """Create and return the configured TTS provider."""
    logger.info(
        "Initializing TTS provider: %s",
        TTS_PROVIDER,
    )

    if TTS_PROVIDER != "piper":
        logger.error(
            "Unsupported TTS provider configured: %s",
            TTS_PROVIDER,
        )
        raise ValueError(
            f"Unsupported TTS provider: {TTS_PROVIDER}"
        )

    try:
        provider = PiperTTS()
    except Exception:
        logger.exception(
            "Failed to initialize TTS provider: %s",
            TTS_PROVIDER,
        )
        raise

    logger.info(
        "TTS provider ready: %s",
        TTS_PROVIDER,
    )
    return provider


def create_vad_provider():
    """Create and return the configured VAD provider."""
    logger.info(
        "Initializing VAD provider: %s",
        VAD_PROVIDER,
    )

    if VAD_PROVIDER != "silero":
        logger.error(
            "Unsupported VAD provider configured: %s",
            VAD_PROVIDER,
        )
        raise ValueError(
            f"Unsupported VAD provider: {VAD_PROVIDER}"
        )

    try:
        provider = SileroVAD()
    except Exception:
        logger.exception(
            "Failed to initialize VAD provider: %s",
            VAD_PROVIDER,
        )
        raise

    logger.info(
        "VAD provider ready: %s",
        VAD_PROVIDER,
    )
    return provider


def create_wakeword_provider():
    """Create and return the configured wake-word provider."""
    logger.info(
        "Initializing wake-word provider: %s",
        WAKEWORD_PROVIDER,
    )

    if WAKEWORD_PROVIDER != "openwakeword":
        logger.error(
            "Unsupported wake-word provider configured: %s",
            WAKEWORD_PROVIDER,
        )
        raise ValueError(
            f"Unsupported wake-word provider: {WAKEWORD_PROVIDER}"
        )

    try:
        provider = OpenWakeWordProvider()
    except Exception:
        logger.exception(
            "Failed to initialize wake-word provider: %s",
            WAKEWORD_PROVIDER,
        )
        raise

    logger.info(
        "Wake-word provider ready: %s",
        WAKEWORD_PROVIDER,
    )
    return provider
