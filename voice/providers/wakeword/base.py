from abc import ABC, abstractmethod


class WakeWordProvider(ABC):

    @abstractmethod
    def score_frame(self, audio_frame) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def is_detected(
        self,
        audio_frame,
    ) -> bool:

        return (
            self.score_frame(audio_frame)
            >= self.threshold
        )