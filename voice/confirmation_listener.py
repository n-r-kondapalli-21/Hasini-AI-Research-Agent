import asyncio
import numpy as np

from .audio_io import record_until_silence, play_audio
from .config import MAX_RECORD_SECONDS
from .providers.factory import create_stt_provider, create_tts_provider


class VoiceConfirmationListener:

    def __init__(self, vad, tts=None):

        self.vad = vad
        self.stt = create_stt_provider()
        self.tts = tts or create_tts_provider()

    async def speak(self, text: str):
        if not text or not text.strip() or self.tts is None:
            return

        def _synth():
            parts = []
            sample_rate = 22050
            for chunk, sr in self.tts.synthesize_stream(text):
                if chunk is not None and len(chunk) > 0:
                    parts.append(chunk)
                    sample_rate = sr
            if parts:
                return np.concatenate(parts), sample_rate
            return None, None

        audio, sample_rate = await asyncio.to_thread(_synth)
        if audio is not None and len(audio) > 0:
            await asyncio.to_thread(play_audio, audio, sample_rate)

    async def listen(self):

        print(
            "\n🎤 Listening for confirmation..."
        )

        audio = await asyncio.to_thread(
            record_until_silence,
            self.vad,
            MAX_RECORD_SECONDS,
        )

        if audio is None or len(audio) == 0:

            return {
                "text": "",
                "confidence": None,
            }

        if hasattr(self.stt, "transcribe_with_confidence"):
            result = await asyncio.to_thread(
                self.stt.transcribe_with_confidence,
                audio,
            )
        else:
            result = await asyncio.to_thread(
                self.stt.transcribe,
                audio,
            )

        # Your current STT interface returns text.
        #
        # Until the provider exposes a confidence value,
        # HIGH-risk confirmations will be rejected rather
        # than pretending we have confidence information.

        if isinstance(result, str):

            return {
                "text": result,
                "confidence": None,
            }

        if isinstance(result, dict):

            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence"),
            }

        return {
            "text": str(result),
            "confidence": None,
        }