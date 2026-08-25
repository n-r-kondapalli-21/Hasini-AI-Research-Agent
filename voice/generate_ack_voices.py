import os
import sys
import wave
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, ".")
from voice.providers.factory import create_tts_provider

phrases = {
    "ack_yes.wav": "Yes?",
    "ack_listening.wav": "I'm listening.",
    "ack_go_ahead.wav": "Go ahead.",
    "ack_yes_listening.wav": "Yes, I'm listening.",
}

os.makedirs("voice/assets", exist_ok=True)
tts = create_tts_provider()

for fname, text in phrases.items():
    audio_parts = []
    sample_rate = 22050
    for audio, rate in tts.synthesize_stream(text):
        if audio is not None and len(audio) > 0:
            audio_parts.append(audio)
            sample_rate = rate
    
    if audio_parts:
        full_audio = np.concatenate(audio_parts)
        pcm_data = (np.clip(full_audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        filepath = os.path.join("voice/assets", fname)
        with wave.open(filepath, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm_data)
        print(f"✅ Generated {filepath} ({len(pcm_data)} bytes)")

print("🎉 All 4 voice acknowledgements generated successfully!")
