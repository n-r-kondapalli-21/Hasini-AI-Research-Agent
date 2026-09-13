import numpy as np
import time

from ..audio_io import audio_stream
from ..providers.factory import create_wakeword_provider


TARGET_SAMPLES = 1280


def main():

    print("=" * 50)
    print("🎤 Hasini Wake Word Calibration")
    print("=" * 50)

    wakeword = create_wakeword_provider()

    print("\nSay:")
    print("    Hey Jarvis")
    print("\nThen wait for the result.")
    print("Press Ctrl+C to stop.\n")

    buffer = np.array(
        [],
        dtype=np.float32,
    )

    highest_score = 0.0
    last_activity = time.monotonic()

    try:

        for frame in audio_stream():

            buffer = np.concatenate(
                (
                    buffer,
                    frame,
                )
            )

            while len(buffer) >= TARGET_SAMPLES:

                chunk = buffer[
                    :TARGET_SAMPLES
                ]

                buffer = buffer[
                    TARGET_SAMPLES:
                ]

                audio = np.clip(
                    chunk * 32767,
                    -32768,
                    32767,
                ).astype(
                    np.int16
                )

                score = wakeword.score_frame(
                    audio
                )

                highest_score = max(
                    highest_score,
                    score,
                )

                # Show current maximum
                print(
                    f"\rCurrent: {score:.3f} | "
                    f"Maximum: {highest_score:.3f}",
                    end="",
                    flush=True,
                )

                # Detection
                if score >= wakeword.threshold:

                    print(
                        f"\n\n🟢 HEY JARVIS DETECTED"
                        f" — {score:.3f}\n"
                    )

                    wakeword.reset()

                    highest_score = 0.0

                    last_activity = (
                        time.monotonic()
                    )

    except KeyboardInterrupt:

        print(
            "\n\n🛑 Calibration stopped."
        )


if __name__ == "__main__":
    main()