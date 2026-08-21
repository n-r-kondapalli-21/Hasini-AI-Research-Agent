from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """
    Base interface for all Text-to-Speech providers.
    """

    @abstractmethod
    def synthesize(self, text: str):
        """
        Synthesize complete text.

        Returns:
            (audio, sample_rate)
        """
        raise NotImplementedError

    @abstractmethod
    def synthesize_stream(self, text: str):
        """
        Stream synthesized audio chunks.

        Yields:
            (audio, sample_rate)
        """
        raise NotImplementedError