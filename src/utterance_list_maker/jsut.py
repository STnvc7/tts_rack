from typing import List
import os

from interface.corpus import Utterance
from interface.preprocess import UtteranceListMaker
from utils.io.yaml import save_as_yaml


class JSUTUtteranceListMaker(UtteranceListMaker):
    def _load_jsut_transcripts(self) -> List[List[str]]:
        transcript_path = f"{self.corpus.root}/transcript_utf8.txt"
        with open(transcript_path) as f:
            transcripts = [s.rstrip().split(":", maxsplit=1) for s in f.readlines()] # list[[filename, text]]
        return transcripts
    
        
    def make(self):
        transcripts = self._load_jsut_transcripts()
        utt_list = [
            Utterance(
                wav_path=os.path.join(self.corpus.root, "wav", f"{filename}.wav"), 
                filename=filename, 
                speaker_id=0, 
                text=text
            ) for filename, text in transcripts
        ]

        train_list = utt_list[:int(len(utt_list)*0.8)]
        os.makedirs(os.path.dirname(self.corpus.utterance_list_path.train), exist_ok=True)
        save_as_yaml(path=self.corpus.utterance_list_path.train, data=train_list)

        valid_list = utt_list[int(len(utt_list)*0.8):int(len(utt_list)*0.9)]
        os.makedirs(os.path.dirname(self.corpus.utterance_list_path.valid), exist_ok=True)
        save_as_yaml(path=self.corpus.utterance_list_path.valid, data=valid_list)

        test_list = utt_list[int(len(utt_list)*0.9):]
        os.makedirs(os.path.dirname(self.corpus.utterance_list_path.test), exist_ok=True)
        save_as_yaml(path=self.corpus.utterance_list_path.test, data=test_list)
        