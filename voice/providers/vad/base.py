from abc import ABC, abstractmethod


class VADProvider(ABC):
    """
    Base interface for Voice Activity Detection.
    """

    @abstractmethod
    def reset(self):
        """
        Reset internal VAD state.
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, audio_frame):
        """
        Process an audio frame.

        Returns:
            Provider-specific speech state.
        """
        raise NotImplementedError

    @abstractmethod
    def is_speech(self, audio_frame) -> bool:
        """
        Determine whether speech is present.
        """
        raise NotImplementedError