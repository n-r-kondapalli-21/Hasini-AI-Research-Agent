# ============================================================
# Audio
# ============================================================

INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1

AUDIO_DEVICE = None

MAX_RECORD_SECONDS = 10


# ============================================================
# Piper TTS
# ============================================================

PIPER_MODEL = r"D:\Hasini_Ai_Research_Agent\models\piper\en_US-lessac-medium.onnx"

PIPER_OUTPUT_FILE = r"D:\Hasini_Ai_Research_Agent\runtime\audio\tts_output.wav"

PIPER_OUTPUT_FILE_TEST = r"D:\Hasini_Ai_Research_Agent\models\piper\test_output.wav"


# ============================================================
# Whisper STT
# ============================================================

# WHISPER_MODEL = r"D:\Hasini_Ai_Research_Agent\models\whisper_models\whisper-base.en"

WHISPER_MODEL = r"D:\Hasini_Ai_Research_Agent\models\whisper_models\whisper-small.en"

WHISPER_COMPUTE_TYPE = "int8"

WHISPER_DEVICE = "cpu"

WHISPER_CPU_THREADS = 2


# --------------------------------------------------
# Phase 2 - Voice Activity Detection
# --------------------------------------------------

VAD_THRESHOLD = 0.5

# End speech after this much silence
VAD_MIN_SILENCE_MS = 500

# Keep a small amount of audio around speech boundaries
VAD_SPEECH_PAD_MS = 80

# Silero works with 16 kHz
VAD_FRAME_MS = 32

VAD_FRAME_SAMPLES = int(INPUT_SAMPLE_RATE * VAD_FRAME_MS / 1000)