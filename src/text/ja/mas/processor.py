from typing import List, Tuple, Optional
import os
from jaconv import jaconv
import torch

from interface.text import TextProcessor, AlignedPhoneme, AvailableLanguage
from ..g2p import G2P
from .symbols import PHONEME_MAP_JP

class JapaneseTextProcessorForMAS(TextProcessor):
    def __init__(self, lang: AvailableLanguage, n_phonemes: int, n_prosodies: int):
        self.lang: AvailableLanguage = lang
        self.n_phonemes = n_phonemes
        self.n_prosodies = n_prosodies
        
    def text_to_phoneme_and_prosody(self, text: str) -> Tuple[str, Optional[str]]:
        text = jaconv.normalize(text)
        phoneme_and_prosody = G2P.from_grapheme(text, drop_unvoiced_vowels=False)
        return phoneme_and_prosody, None
                
    def phoneme_to_id(self, phoneme: str) -> List[int]:
        phoneme_ids = [PHONEME_MAP_JP[p] for p in phoneme.split()]
        return phoneme_ids
        
    def prosody_to_id(self, prosody: str) -> List[int]:
        raise ValueError("Prosody is not supported.")
        
    def force_alignment(self, phoneme: str, wav: torch.Tensor, sample_rate: int) -> List[AlignedPhoneme]:
        raise ValueError("Force alignment is not supported.")