import logging
import time


logger = logging.getLogger("hasini.voice.response_segmenter")


class ResponseSegmenter:
    """
    Converts an LLM token stream into TTS-ready text chunks.
    """

    def __init__(
        self,
        min_chunk_length: int = 25,
        max_buffer_time: float = 3.0,
    ):
        if min_chunk_length < 1:
            raise ValueError("min_chunk_length must be at least 1.")

        if max_buffer_time <= 0:
            raise ValueError("max_buffer_time must be greater than 0.")

        self.min_chunk_length = min_chunk_length
        self.max_buffer_time = max_buffer_time

        self.buffer = ""
        self.start_time: float | None = None

        logger.debug(
            "Response segmenter initialized "
            "(min_chunk_length=%d, max_buffer_time=%.2fs).",
            min_chunk_length,
            max_buffer_time,
        )

    def reset(self) -> None:
        """Clear the current buffered response text."""
        self.buffer = ""
        self.start_time = None

    def add(self, token: str):
        """
        Add a streamed token.

        Returns:
            A completed text chunk, or None.
        """
        if not token:
            return None

        try:
            if self.start_time is None:
                self.start_time = time.monotonic()

            self.buffer += str(token)

            stripped = self.buffer.strip()

            if not stripped:
                return None

            # ----------------------------------------------
            # 1. Sentence boundary
            # ----------------------------------------------
            if stripped.endswith((".", "!", "?")):
                return self._flush()

            # ----------------------------------------------
            # 2. Clause boundary
            # ----------------------------------------------
            if (
                len(stripped) >= self.min_chunk_length
                and stripped.endswith((",", ";", ":"))
            ):
                return self._flush()

            # ----------------------------------------------
            # 3. Maximum buffering time
            # ----------------------------------------------
            elapsed = time.monotonic() - self.start_time

            if elapsed >= self.max_buffer_time:
                return self._flush()

            return None

        except Exception:
            logger.exception("Failed to process streamed response token.")
            raise

    def flush(self):
        """Flush any remaining buffered text."""
        if not self.buffer.strip():
            return None

        return self._flush()

    def _flush(self):
        """Return the current buffer and reset the segmenter."""
        chunk = self.buffer.strip()

        self.buffer = ""
        self.start_time = None

        logger.debug(
            "Response segment flushed: %r",
            chunk,
        )

        return chunk
