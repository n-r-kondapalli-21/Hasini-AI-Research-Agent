import math
import time

from faster_whisper import WhisperModel

from ...config import (
    WHISPER_MODEL,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_CPU_THREADS,
)

from .base import STTProvider


class FasterWhisperSTT(STTProvider):
    """
    faster-whisper Speech-to-Text provider.
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

    def transcribe_with_confidence(self, audio) -> dict:

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

        seg_list = list(segments)
        if not seg_list:
            return {"text": "", "confidence": 0.0}

        text = " ".join(
            s.text.strip()
            for s in seg_list
        ).strip()

        confidences = []
        for s in seg_list:
            if hasattr(s, "avg_logprob") and s.avg_logprob is not None:
                conf = math.exp(s.avg_logprob)
                confidences.append(conf)

        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        avg_confidence = max(0.0, min(1.0, float(avg_confidence)))

        return {
            "text": text,
            "confidence": avg_confidence,
        }

    def transcribe(self, audio) -> str:
        res = self.transcribe_with_confidence(audio)
        return res["text"]