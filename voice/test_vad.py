from .providers.factory import create_vad_provider
from .audio_io import record_until_silence
from .config import MAX_RECORD_SECONDS


def main():

    print("=" * 50)
    print("🎤 Phase 2.1 - VAD Test")
    print("=" * 50)

    vad = create_vad_provider()

    audio = record_until_silence(
        vad,
        MAX_RECORD_SECONDS
    )

    print(
        f"\nRecorded samples: {len(audio)}"
    )

    if len(audio) == 0:

        print("❌ No speech detected.")

    else:

        duration = len(audio) / 16000

        print(
            f"✅ Recorded duration: {duration:.2f} seconds"
        )


if __name__ == "__main__":
    main()