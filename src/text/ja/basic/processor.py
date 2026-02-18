from typing import List, Tuple, Optional
import os
import torch
from dsp_board.preprocesses import resample
import pydomino

from interface.text import TextProcessor, AlignedPhoneme, AvailableLanguage
from ..g2p import G2P
from .symbols import PHONEME_JP, PHONEME_MAP_JP, PROSODY_MAP_JP
from ..cleaner import japanese_text_cleaner

class JapaneseTextProcessor(TextProcessor):
    def __init__(self, lang: AvailableLanguage, n_phonemes: int, n_prosodies: int, aligner_path: str):
        self.aligner = pydomino.Aligner(aligner_path)
        self.lang: AvailableLanguage = lang
        self.n_phonemes = n_phonemes
        self.n_prosodies = n_prosodies
    
    def text_to_phoneme_and_prosody(self, text: str) -> Tuple[str, Optional[str]]:
        text = japanese_text_cleaner(text)
        phoneme_and_prosody = G2P.from_grapheme(text, drop_unvoiced_vowels=False)
        phoneme = []
        prosody = []
        for p in phoneme_and_prosody.split():
            if p in ["^", "$", "?", "_"]:
                phoneme += ["pau"]
                prosody += [p]
            elif p in PHONEME_JP:
                phoneme += [p]
                prosody += ["_"]
            else:
                prosody[-1] = p
        return " ".join(phoneme), " ".join(prosody)
                
    def phoneme_to_id(self, phoneme: str) -> List[int]:
        phoneme_ids = [PHONEME_MAP_JP[p] for p in phoneme.split()]
        return phoneme_ids
        
    def prosody_to_id(self, prosody: str) -> List[int]:
        prosody_ids = [PROSODY_MAP_JP[p] for p in prosody.split()]
        return prosody_ids
        
    def force_alignment(self, phoneme: str, wav: torch.Tensor, sample_rate: int) -> List[AlignedPhoneme]:
        # 強制アライメントにはpydominoを使用．
        # https://github.com/DwangoMediaVillage/pydomino
        # https://dwangomediavillage.github.io/pydomino/
        
        wav_for_align = resample(wav, sample_rate, 16_000).squeeze().numpy()
        alignment = [
            AlignedPhoneme(phoneme, start, end) 
            for start, end, phoneme in self.aligner.align(wav_for_align, phoneme, 3)
        ]
            
        return alignment