from typing import List
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from interface.corpus import Utterance
from interface.preprocess import UtteranceListMaker
from utils.io.yaml import save_as_yaml
from utils.environment import seed_everything

EXCLUDE_CHARS = ["（", "）", "(", ")", "「", "」"]

class HiFiCaptainUtteranceListMaker(UtteranceListMaker):
    def _load_transcripts(self, split) -> List[List[str]]:
        transcript_path = os.path.join(self.corpus.root, "text", f"{split}.txt")
        with open(transcript_path) as f:
            transcripts = [s.rstrip().split(" ", maxsplit=1) for s in f.readlines()] # list[[filename, text]]
        return transcripts
    
    def _is_valid_text(self, text: str) -> bool:
        for char in EXCLUDE_CHARS:
            if char in text:
                return False
        return True
        
    def make(self):
        transcripts_train = self._load_transcripts("train_parallel")
        utt_list_train = [
            Utterance(
                wav_path=os.path.join(self.corpus.root, "wav", "train_parallel", f"{name}.wav"), 
                filename=name, 
                speaker_id=0, 
                text=text
            ) for name, text in transcripts_train if self._is_valid_text(text)
        ]
        os.makedirs(os.path.dirname(self.corpus.utterance_list_path.train), exist_ok=True)
        save_as_yaml(path=self.corpus.utterance_list_path.train, data=utt_list_train)
        
        transcripts_valid = self._load_transcripts("dev")
        utt_list_valid = [
            Utterance(
                wav_path=os.path.join(self.corpus.root, "wav", "dev", f"{name}.wav"), 
                filename=name, 
                speaker_id=0, 
                text=text
            ) for name, text in transcripts_valid if self._is_valid_text(text)
        ]
        os.makedirs(os.path.dirname(self.corpus.utterance_list_path.valid), exist_ok=True)
        save_as_yaml(path=self.corpus.utterance_list_path.valid, data=utt_list_valid)
        
        transcripts_test = self._load_transcripts("eval")
        utt_list_test = [
            Utterance(
                wav_path=os.path.join(self.corpus.root, "wav", "eval", f"{name}.wav"), 
                filename=name, 
                speaker_id=0, 
                text=text
            ) for name, text in transcripts_test if self._is_valid_text(text)
        ]
        os.makedirs(os.path.dirname(self.corpus.utterance_list_path.test), exist_ok=True)
        save_as_yaml(path=self.corpus.utterance_list_path.test, data=utt_list_test)
