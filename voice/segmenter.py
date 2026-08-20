import time


class ResponseSegmenter:
    """
    Converts an LLM token stream into TTS-ready text chunks.
    """

    def __init__(
        self,
        min_chunk_length: int = 25,
        max_buffer_time: float = 3.0,
    ):
        self.min_chunk_length = min_chunk_length
        self.max_buffer_time = max_buffer_time

        self.buffer = ""
        self.start_time = None

    def reset(self):
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

        if self.start_time is None:
            self.start_time = time.monotonic()

        self.buffer += token

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

        elapsed = (
            time.monotonic() - self.start_time
        )

        if elapsed >= self.max_buffer_time:

            return self._flush()

        return None

    def flush(self):
        """
        Flush any remaining text.
        """

        if not self.buffer.strip():
            return None

        return self._flush()

    def _flush(self):
        """
        Return current buffer and reset.
        """

        chunk = self.buffer.strip()

        self.buffer = ""
        self.start_time = None

        return chunk