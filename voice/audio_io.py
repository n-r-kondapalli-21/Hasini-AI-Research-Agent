import threading

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


# ============================================================
# PLAYBACK
#
# A single persistent OutputStream is reused across chunks
# within a response instead of opening/closing a stream per
# chunk (sd.play()/sd.wait()). Reopening a stream per chunk
# left a small gap where sd.wait() could return before the
# OS mixer had actually finished draining the previous
# chunk's tail, causing the next chunk to start writing while
# the previous one was still audibly playing.
#
# stream.write() blocks until the given audio has been fully
# accepted into the stream's buffer, so chunk N+1 cannot be
# written until chunk N's write() call returns.
# ============================================================


_stream = None
_stream_lock = threading.Lock()
_stream_sample_rate = None
_cancel_event = threading.Event()

# ~20ms per chunk keeps the lock hold time short so stop_audio()
# can acquire it almost immediately instead of racing a long write().
_PLAYBACK_CHUNK_FRAMES = 1024


def _get_stream_locked(sample_rate: int, channels: int = 1):
    """Must be called with _stream_lock already held."""
    global _stream, _stream_sample_rate

    if _stream is None or _stream_sample_rate != sample_rate:

        if _stream is not None:
            try:
                _stream.stop()
                _stream.close()
            except Exception:
                pass

        _stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=AUDIO_DEVICE,
        )
        _stream.start()
        _stream_sample_rate = sample_rate

    return _stream


def play_audio(audio: np.ndarray, sample_rate: int):
    """
    Play audio through the speaker, in small chunks, checking for
    cancellation between chunks so stop_audio() never has to abort/close
    the stream while a write() is in flight on another thread.
    """

    if audio is None or len(audio) == 0:
        return

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    _cancel_event.clear()

    pos = 0
    n = len(audio)

    while pos < n:

        if _cancel_event.is_set():
            break

        end = min(pos + _PLAYBACK_CHUNK_FRAMES, n)
        chunk = audio[pos:end]

        with _stream_lock:

            if _cancel_event.is_set():
                break

            try:
                stream = _get_stream_locked(sample_rate, channels=audio.shape[1])
                stream.write(chunk)
            except Exception:
                # Stream was aborted/closed concurrently by stop_audio()
                break

        pos = end


def stop_audio():
    """
    Immediately stop audio playback and discard any buffered audio
    (used on cancel/reset, e.g. barge-in).
    """

    global _stream, _stream_sample_rate

    _cancel_event.set()

    with _stream_lock:

        if _stream is not None:
            try:
                _stream.abort()
                _stream.close()
            except Exception:
                pass

            _stream = None
            _stream_sample_rate = None