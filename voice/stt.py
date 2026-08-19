import time

from faster_whisper import WhisperModel

from .config import (
    WHISPER_MODEL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_CPU_THREADS,
)


class SpeechToText:
    """
    Local Speech-to-Text using faster-whisper.
    """

    def __init__(self):

        print("\nLoading Whisper model...")
        print(f"Model: {WHISPER_MODEL}")

        start_time = time.perf_counter()

        self.model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_CPU_THREADS,
        num_workers=1,
        )

        load_time = time.perf_counter() - start_time

        print(
            f"✅ Whisper loaded in {load_time:.2f} seconds."
        )

    def transcribe(self, audio) -> str:

        segments, info = self.model.transcribe(
            audio,

            language="en",

            beam_size=8,
            best_of=5,

            temperature=0.0,

            initial_prompt=(
                "Hasini, Python, LangChain, LangGraph, MCP, "
                "Model Context Protocol, GitHub, Git, "
                "OpenRouter, Ollama, FastAPI, filesystem, "
                "AI agent, research agent, tool, tools, "
                "Windows, PowerShell, "
                "search, execute, create, delete, read, write, quit, exit."
            ),

            condition_on_previous_text=False,

            vad_filter=True,

            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
            },

            without_timestamps=True,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return text.strip()