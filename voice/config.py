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

WHISPER_MODEL = r"D:\Hasini_Ai_Research_Agent\models\whisper_models\whisper-small.en"

WHISPER_COMPUTE_TYPE = "int8"

WHISPER_DEVICE = "cpu"

WHISPER_CPU_THREADS = 2


# ============================================================
# Voice Activity Detection
# ============================================================

VAD_MODEL = r"D:\Hasini_Ai_Research_Agent\models\vad\silero_vad.onnx"

VAD_THRESHOLD = 0.5

# End speech after this much silence
VAD_MIN_SILENCE_MS = 500

# Keep a small amount of audio around speech boundaries
VAD_SPEECH_PAD_MS = 80

# Silero works with 16 kHz
VAD_FRAME_MS = 32

VAD_FRAME_SAMPLES = int(INPUT_SAMPLE_RATE * VAD_FRAME_MS / 1000)

# ============================================================
# Phase 3 - Wake Word
# ============================================================

WAKEWORD_MODEL = r"D:\Hasini_Ai_Research_Agent\models\wakeword\hey_jarvis_v0.1.onnx"

WAKEWORD_THRESHOLD = 0.5





# ============================================================
# Phase 3 - Barge-In Detection
# ============================================================

# Barge-in is intentionally stricter than normal command VAD.
BARGE_IN_VAD_THRESHOLD = 0.85

# Number of consecutive speech frames required.
#
# VAD_FRAME_MS = 32 ms
# 12 frames = 384 ms
BARGE_IN_REQUIRED_SPEECH_FRAMES = 12

# Ignore VAD immediately after TTS starts.
# This helps avoid playback/onset artifacts.
BARGE_IN_GRACE_PERIOD_MS = 400

# Ignore extremely quiet microphone signals.
# RMS below this value is treated as background noise.
BARGE_IN_MIN_RMS = 0.008



# ============================================================
# Voice Providers
# ============================================================

STT_PROVIDER = "faster_whisper"

TTS_PROVIDER = "piper"

VAD_PROVIDER = "silero"

WAKEWORD_PROVIDER = "openwakeword"



#voice test using pre-define responce to avoid repeated requested to llm while testing 
VOICE_TEST_MODE = False              #True for enable Test mode  and False for disable 
  

