from abc import ABC, abstractmethod


class STTProvider(ABC):
    """
    Base interface for all Speech-to-Text providers.
    """

    @abstractmethod
    def transcribe(self, audio) -> str:
        """
        Convert audio into text.
        """
        raise NotImplementedError