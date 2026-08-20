import numpy as np
import sounddevice as sd

from .config import (INPUT_SAMPLE_RATE, INPUT_CHANNELS, AUDIO_DEVICE, VAD_FRAME_SAMPLES)


def audio_stream():
    """
    Continuously yield microphone frames.

    Each frame is approximately VAD_FRAME_MS long.
    """

    stream = sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        dtype="float32",
        device=AUDIO_DEVICE,
        blocksize=VAD_FRAME_SAMPLES,
    )

    try:

        stream.start()

        while True:

            audio, overflowed = stream.read(
                VAD_FRAME_SAMPLES
            )

            if overflowed:
                print("⚠️ Microphone buffer overflow.")

            yield audio.flatten()

    finally:

        stream.stop()
        stream.close()


def record_until_silence(vad, max_duration):
    """
    Wait for speech and record until the user stops speaking.

    Returns:
        numpy.float32 audio array.
    """

    print("\n🎤 Listening...")

    vad.reset()

    recorded_frames = []

    speech_started = False

    max_frames = int(
        max_duration
        * INPUT_SAMPLE_RATE
        / VAD_FRAME_SAMPLES
    )

    frame_count = 0

    for frame in audio_stream():

        frame_count += 1

        result = vad.process(frame)

        # ----------------------------------------------
        # Speech started
        # ----------------------------------------------

        if result is not None and "start" in result:

            if not speech_started:

                speech_started = True

                print("🗣️ Speech detected.")

        # ----------------------------------------------
        # Save audio after speech starts
        # ----------------------------------------------

        if speech_started:

            recorded_frames.append(frame.copy())

        # ----------------------------------------------
        # Speech ended
        # ----------------------------------------------

        if (
            speech_started
            and result is not None
            and "end" in result
        ):

            print("✅ Speech complete.")

            break

        # ----------------------------------------------
        # Safety timeout
        # ----------------------------------------------

        if frame_count >= max_frames:

            if speech_started:
                print("⏱️ Maximum recording time reached.")
            else:
                print("⏱️ No speech detected.")

            break

    if not recorded_frames:

        return np.array([], dtype=np.float32)

    return np.concatenate(recorded_frames)


def play_audio(audio: np.ndarray, sample_rate: int):
    """
    Play audio through the speaker.
    """

    if audio is None or len(audio) == 0:
        return

    sd.play(
        audio,
        sample_rate,
        device=AUDIO_DEVICE,
    )

    sd.wait()


def stop_audio():
    """
    Immediately stop audio playback.
    """

    sd.stop()