from __future__ import annotations

import logging
import os
import threading
import time
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch
import warnings

# Known Kokoro/PyTorch/phonemizer warnings that do not
# affect generated speech quality in our configuration.
warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    # NOTE: torch's actual message is wrapped in backticks
    # ("`torch.nn.utils.weight_norm` is deprecated..."), and
    # filterwarnings anchors `message` at the start of the string
    # via re.match — a pattern starting with `torch\.` (no leading
    # backtick) never matches. Match loosely instead so this also
    # survives future torch wording changes.
    message=r".*weight_norm.*deprecated.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"words count mismatch.*",
    category=UserWarning,
    module=r"phonemizer.*",
)

# Suppress phonemizer logging warnings about word count mismatches
logging.getLogger("phonemizer").setLevel(logging.ERROR)

from kokoro import KModel, KPipeline

from ...config import (
    KOKORO_MODEL,
    KOKORO_CONFIG,
    KOKORO_VOICE,
    KOKORO_OUTPUT_FILE,
)

from .base import TTSProvider


logger = logging.getLogger(__name__)


class KokoroTTS(TTSProvider):
    """
    Kokoro Text-to-Speech provider.

    Uses the locally stored Kokoro model and voice.

    Supports:
        - Complete synthesis
        - Streaming synthesis (with barge-in cancellation)
    """

    REPO_ID = "hexgrad/Kokoro-82M"
    LANG_CODE = "a"
    SAMPLE_RATE = 24000

    def __init__(self, warm_up: bool = True):
        self.model_path = KOKORO_MODEL
        self.config_path = KOKORO_CONFIG
        self.voice_path = KOKORO_VOICE
        self.output_file = KOKORO_OUTPUT_FILE

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None
        self.pipeline = None

        # Guards concurrent synthesis / writes to self.output_file.
        # Matters once barge-in can interrupt an in-flight synthesize()
        # while a new one starts for the fresh turn.
        self._lock = threading.Lock()

        self._prepare_output_directory()
        self._validate_files()
        self._load_model()

        if warm_up:
            self._warm_up()

    def _prepare_output_directory(self) -> None:
        """Create the TTS output directory if required."""

        output_directory = os.path.dirname(self.output_file)

        if not output_directory:
            return

        try:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )
        except OSError:
            logger.exception(
                "Failed to create Kokoro output directory: %s",
                output_directory,
            )
            raise

    def _validate_files(self) -> None:
        """Verify that all required Kokoro files exist."""

        required_files = {
            "Kokoro model": self.model_path,
            "Kokoro config": self.config_path,
            "Kokoro voice": self.voice_path,
        }

        for name, path in required_files.items():
            if not os.path.isfile(path):
                logger.error(
                    "%s not found: %s",
                    name,
                    path,
                )
                raise FileNotFoundError(
                    f"{name} not found:\n{path}"
                )

    def _load_model(self) -> None:
        """Load the local Kokoro model and pipeline."""

        logger.info("Loading Kokoro TTS...")
        logger.info("Model: %s", self.model_path)
        logger.info("Config: %s", self.config_path)
        logger.info("Voice: %s", self.voice_path)
        logger.info("Device: %s", self.device)

        start_time = time.perf_counter()

        try:
            self.model = KModel(
                repo_id=self.REPO_ID,
                config=self.config_path,
                model=self.model_path,
            ).to(self.device).eval()

            # Note: device placement is handled by the .to(self.device)
            # call above on self.model — KPipeline itself takes no
            # device kwarg in the reference usage for this model.
            self.pipeline = KPipeline(
                lang_code=self.LANG_CODE,
                repo_id=self.REPO_ID,
                model=self.model,
            )

        except Exception:
            load_time = time.perf_counter() - start_time

            logger.exception(
                "Failed to load Kokoro TTS after %.2f seconds.",
                load_time,
            )
            # Don't leave a half-initialized model sitting around.
            self.model = None
            self.pipeline = None
            raise

        load_time = time.perf_counter() - start_time

        logger.info(
            "Kokoro TTS loaded successfully in %.2f seconds.",
            load_time,
        )

    def _warm_up(self) -> None:
        """
        Run a throwaway synthesis right after load so that first-call
        latency (kernel/lazy-init warm-up) doesn't land on the user's
        first real turn. Failures here are logged, not fatal.
        """

        try:
            start_time = time.perf_counter()

            for _ in self.synthesize_stream("Hello."):
                pass

            logger.info(
                "Kokoro warm-up completed in %.2f seconds.",
                time.perf_counter() - start_time,
            )
        except Exception:
            logger.warning(
                "Kokoro warm-up synthesis failed; continuing anyway.",
                exc_info=True,
            )

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        output_file: Optional[str] = None,
        interrupt_event: Optional[threading.Event] = None,
    ):
        """
        Synthesize complete text and write it to disk.

        Args:
            text: Text to synthesize.
            speed: Kokoro speech-rate multiplier.
            output_file: Optional override path; defaults to
                self.output_file. Pass a unique path per call if you
                may have overlapping synthesize() calls (e.g. a new
                turn starting right after a barge-in interrupt).
            interrupt_event: Optional event checked between chunks;
                when set, synthesis stops early and whatever audio
                was generated so far is still returned/written.

        Returns:
            (audio, sample_rate), or (None, None) for empty input or
            if interrupted before any audio was produced.
        """

        if not text or not text.strip():
            logger.debug(
                "Skipping Kokoro synthesis because text is empty."
            )
            return None, None

        target_file = output_file or self.output_file

        audio_chunks = []
        sample_rate = self.SAMPLE_RATE

        try:
            for audio, current_sample_rate in self.synthesize_stream(
                text,
                speed=speed,
                interrupt_event=interrupt_event,
            ):
                audio_chunks.append(audio)
                sample_rate = current_sample_rate

        except Exception:
            logger.exception(
                "Kokoro synthesis failed."
            )
            raise

        if not audio_chunks:
            logger.warning(
                "Kokoro produced no audio for the supplied text."
            )
            return None, None

        audio = np.concatenate(
            audio_chunks
        ).astype(
            np.float32,
            copy=False,
        )

        with self._lock:
            try:
                sf.write(
                    target_file,
                    audio,
                    sample_rate,
                )
            except Exception:
                logger.exception(
                    "Failed to write Kokoro audio file: %s",
                    target_file,
                )
                raise

        logger.debug(
            "Kokoro synthesis completed. "
            "Sample rate=%s, samples=%d",
            sample_rate,
            len(audio),
        )

        return audio, sample_rate

    def synthesize_stream(
        self,
        text: str,
        speed: float = 1.0,
        interrupt_event: Optional[threading.Event] = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """
        Stream Kokoro-generated audio, chunk by chunk.

        Args:
            text: Text to synthesize.
            speed: Kokoro speech-rate multiplier.
            interrupt_event: Optional threading.Event. Checked before
                each chunk is yielded; if set, the generator stops
                immediately without producing further chunks. This is
                the hook the barge-in path should set when the user
                starts talking over playback, so Kokoro doesn't keep
                synthesizing the rest of a reply nobody will hear.

        Yields:
            (audio, sample_rate)
        """

        if not text or not text.strip():
            logger.debug(
                "Skipping Kokoro streaming synthesis "
                "because text is empty."
            )
            return

        if self.pipeline is None:
            raise RuntimeError(
                "Kokoro pipeline is not initialized."
            )

        try:
            with torch.inference_mode():
                results = self.pipeline(
                    text,
                    voice=self.voice_path,
                    speed=speed,
                )

                for result in results:
                    if interrupt_event is not None and interrupt_event.is_set():
                        logger.debug(
                            "Kokoro streaming synthesis interrupted "
                            "(barge-in)."
                        )
                        return

                    audio = getattr(
                        result,
                        "audio",
                        None,
                    )

                    if audio is None:
                        continue

                    if hasattr(audio, "detach"):
                        audio = (
                            audio.detach()
                            .cpu()
                            .numpy()
                        )

                    audio = np.asarray(
                        audio,
                        dtype=np.float32,
                    )

                    if audio.size == 0:
                        continue

                    sample_rate = getattr(
                        result,
                        "sr",
                        self.SAMPLE_RATE,
                    )

                    yield audio, int(sample_rate)

        except Exception:
            logger.exception(
                "Kokoro streaming synthesis failed."
            )
            raise