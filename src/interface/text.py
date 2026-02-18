from typing import List, Annotated, Tuple, Optional, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch

AvailableLanguage = Literal["en", "ja"]

@dataclass
class AlignedPhoneme:
    phoneme: str
    start: Annotated[float, "sec"]
    end: Annotated[float, "sec"]

class TextProcessor(ABC):
    def __init__(self, lang: AvailableLanguage, n_phonemes: int, n_prosodies: Optional[int]):
        self.lang = lang
        self.n_phonemes = n_phonemes
        self.n_prosodies = n_prosodies

    @abstractmethod
    def text_to_phoneme_and_prosody(self, text: str) -> Tuple[str, Optional[str]]:
        raise NotImplementedError

    @abstractmethod
    def phoneme_to_id(self, phoneme: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def prosody_to_id(self, prosody: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def force_alignment(self, phoneme: str, wav: torch.Tensor, sample_rate: int) -> List[AlignedPhoneme]:
        raise NotImplementedError