import logging
import os
import wave

import numpy as np

from voice.providers.factory import create_tts_provider


logger = logging.getLogger("hasini.voice.acknowledgements")


PHRASES = {
    "ack_yes.wav": "Yes?",
    "ack_listening.wav": "I'm listening.",
    "ack_go_ahead.wav": "Go ahead.",
    "ack_yes_listening.wav": "Yes, I'm listening.",
}

OUTPUT_DIR = os.path.join("voice", "assets")


def _configure_logging() -> None:
    """Configure basic logging when the project has not configured it."""
    root = logging.getLogger()

    if root.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _generate_audio(tts, text: str):
    """Generate a complete audio array from the TTS stream."""
    audio_parts = []
    sample_rate = 22050

    try:
        for audio, rate in tts.synthesize_stream(text):
            if audio is not None and len(audio) > 0:
                audio_parts.append(audio)
                sample_rate = rate
    except Exception:
        logger.exception("TTS synthesis failed for phrase: %r", text)
        raise

    if not audio_parts:
        return None, None

    try:
        return np.concatenate(audio_parts), sample_rate
    except Exception:
        logger.exception(
            "Failed to combine generated audio for phrase: %r",
            text,
        )
        raise


def _save_wav(filepath: str, audio: np.ndarray, sample_rate: int) -> int:
    """Convert normalized audio to PCM16 and save it as a WAV file."""
    try:
        pcm_data = (
            np.clip(audio, -1.0, 1.0)
            * 32767.0
        ).astype(np.int16).tobytes()

        with wave.open(filepath, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        return len(pcm_data)

    except Exception:
        logger.exception("Failed to save WAV file: %s", filepath)
        raise


def main() -> int:
    """Generate all predefined voice acknowledgement WAV files."""
    _configure_logging()

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("Generating voice acknowledgement files...")

        try:
            tts = create_tts_provider()
        except Exception:
            logger.exception("Failed to initialize TTS provider.")
            return 1

        generated_count = 0

        for filename, phrase in PHRASES.items():
            filepath = os.path.join(OUTPUT_DIR, filename)

            logger.info(
                "Generating %s: %r",
                filename,
                phrase,
            )

            try:
                audio, sample_rate = _generate_audio(
                    tts,
                    phrase,
                )

                if audio is None or len(audio) == 0:
                    logger.warning(
                        "No audio generated for %s.",
                        filename,
                    )
                    continue

                byte_count = _save_wav(
                    filepath,
                    audio,
                    sample_rate,
                )

                generated_count += 1

                logger.info(
                    "Generated %s (%d bytes).",
                    filepath,
                    byte_count,
                )

            except Exception:
                logger.exception(
                    "Failed to generate acknowledgement: %s",
                    filename,
                )

        if generated_count == len(PHRASES):
            logger.info(
                "All %d voice acknowledgements generated successfully.",
                len(PHRASES),
            )
            return 0

        logger.warning(
            "Generated %d of %d voice acknowledgements.",
            generated_count,
            len(PHRASES),
        )
        return 1

    except KeyboardInterrupt:
        logger.info("Voice acknowledgement generation interrupted.")
        return 130

    except Exception:
        logger.exception(
            "Unexpected error while generating voice acknowledgements."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
