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


def create_stt_provider():

    if STT_PROVIDER == "faster_whisper":
        return FasterWhisperSTT()

    raise ValueError(
        f"Unsupported STT provider: {STT_PROVIDER}"
    )


def create_tts_provider():

    if TTS_PROVIDER == "piper":
        return PiperTTS()

    raise ValueError(
        f"Unsupported TTS provider: {TTS_PROVIDER}"
    )


def create_vad_provider():

    if VAD_PROVIDER == "silero":
        return SileroVAD()

    raise ValueError(
        f"Unsupported VAD provider: {VAD_PROVIDER}"
    )


def create_wakeword_provider():

    if WAKEWORD_PROVIDER == "openwakeword":
        return OpenWakeWordProvider()

    raise ValueError(
        f"Unsupported wake-word provider: "
        f"{WAKEWORD_PROVIDER}"
    )