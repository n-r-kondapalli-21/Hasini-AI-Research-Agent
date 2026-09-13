from kokoro import KModel, KPipeline
import soundfile as sf


MODEL = r"models\kokoro\kokoro-v1_0.pth"
CONFIG = r"models\kokoro\config.json"
VOICE = r"models\kokoro\voices\af_heart.pt"
OUTPUT = r"models\kokoro\test_output.wav"


print("Loading Kokoro model...")

model = KModel(
    config=CONFIG,
    model=MODEL,
)

print("Creating Kokoro pipeline...")

pipeline = KPipeline(
    lang_code="a",
    model=model,
    device="cpu",
)

text = "Hello, I am Hasini. This is a test of the Kokoro text to speech system."

print("Generating speech...")

for result in pipeline(
    text,
    voice=VOICE,
):
    if result.audio is None:
        continue

    audio = result.audio.detach().cpu().numpy()

    sample_rate = getattr(
        result,
        "sr",
        24000,
    )

    sf.write(
        OUTPUT,
        audio,
        sample_rate,
    )

    print(
        f"Audio generated successfully: {OUTPUT}"
    )
    print(f"Sample rate: {sample_rate}")
    print(f"Samples: {len(audio)}")
    break

print("Test completed.")