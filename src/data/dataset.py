from typing import List, Dict, Optional, Literal
import math
import random
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from dsp_board.preprocesses import resample

from interface.data import DatasetOutput
from interface.text import TextProcessor
from interface.feature import AcousticFeature
from interface.corpus import Utterance
from utils.io.yaml import load_from_yaml
from dsp import DSPProcessor

class TTSDataset(Dataset):
    def __init__(
        self,
        corpus_root: str,
        utterance_list_path: str,
        dsp_processor: DSPProcessor,
        text_processor: TextProcessor,
        features_to_extract: List[AcousticFeature],
        require_duration: bool,
        segment_size: Optional[int],
        random_onset: bool = True,
        trim: Optional[Literal["forward", "backward", "both"]] = None,
        normalize: Optional[Literal["peak", "loudness"]] = None
    ):
        super().__init__()
        
        self.corpus_root = corpus_root
        self.utterance_list = load_from_yaml(utterance_list_path, List[Utterance])
        self.dsp_processor = dsp_processor
        self.sample_rate = dsp_processor.sample_rate
        self.hop_size = dsp_processor.hop_size
        self.text_processor = text_processor
        self.features_to_extract = features_to_extract
        self.require_duration = require_duration
        self.segment_size = segment_size
        self.random_onset = random_onset
        self.trim: Optional[Literal["forward", "backward", "both"]] = trim
        self.normalize = normalize

    def __len__(self):
        return len(self.utterance_list)

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _process_wav(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if sr != self.dsp_processor.sample_rate:
            wav = resample(wav, sr, self.sample_rate)
        if self.trim is not None:
            wav = self.dsp_processor.trim(wav, direction=self.trim)
        if self.normalize is not None:
            if self.normalize == "peak":
                wav = self.dsp_processor.peak_normalize(wav)
            elif self.normalize == "loudness":
                wav = self.dsp_processor.loudness_normalize(wav)
        return wav
        
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def _extract_feature(self, wav: torch.Tensor) -> Dict[AcousticFeature, torch.Tensor]:
        features: Dict[AcousticFeature, torch.Tensor] = {}
        for func_name in self.features_to_extract:
            dsp_fn = getattr(self.dsp_processor, func_name)
            _feature = dsp_fn(wav)
            features[func_name] = _feature
        return features
        
    def _calculate_duration(self, phoneme: str, wav: torch.Tensor):
        alignment = self.text_processor.force_alignment(phoneme, wav, self.sample_rate)
        duration = [
            (a.end*self.sample_rate)//self.hop_size - (a.start*self.sample_rate)//self.hop_size
            for a in alignment
        ]
        duration = torch.Tensor(duration).int()
        return duration

    def __getitem__(self, idx: int):
        try:
            return self.process_item(idx)
        except Exception as e:
            print(f"Error processing utterance {self.utterance_list[idx].filename}: {e}")
            return None
        
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def process_item(self, idx: int):
        utterance = self.utterance_list[idx]
        filename = utterance.filename
        speaker_id = torch.Tensor([utterance.speaker_id]).int()
        text = utterance.text
        phoneme, prosody = self.text_processor.text_to_phoneme_and_prosody(text)
        phoneme_id = torch.Tensor(self.text_processor.phoneme_to_id(phoneme)).int()
        if prosody is not None:
            prosody_id = torch.Tensor(self.text_processor.prosody_to_id(prosody)).int()
        else:
            prosody_id = None
        
        wav = self._process_wav(utterance.wav_path)
        if self.segment_size and wav.shape[-1] <= self.segment_size:
            wav = F.pad(wav, (0, self.segment_size - wav.shape[-1]))
        features = self._extract_feature(wav.squeeze(0))
            
        if self.require_duration:
            duration = self._calculate_duration(phoneme, wav)
            diff = duration.sum() - math.ceil(wav.shape[-1] / self.hop_size)
            if diff != 0:
                duration[-1] = duration[-1] - diff
        else:
            duration = None
        
        segment_id_wav = None
        segment_id_feats = None
        if self.segment_size:
            segment_size_frame = math.ceil(self.segment_size / self.hop_size)
            if self.random_onset:
                onset = random.randint(0, wav.shape[-1] - self.segment_size)
            else:
                onset = 0
            offset = onset + self.segment_size
            onset_frame = math.ceil(onset/self.hop_size)
            offset_frame = onset_frame + segment_size_frame
            segment_id_wav = torch.arange(onset, offset)
            segment_id_feats = torch.arange(onset_frame, offset_frame)
            
        return DatasetOutput(
            filename=filename,
            speaker_id=speaker_id,
            text=text,
            phoneme=phoneme,
            phoneme_id=phoneme_id,
            prosody=prosody,
            prosody_id=prosody_id,
            duration=duration,
            wav=wav,
            features=features,
            segment_id_wav=segment_id_wav,
            segment_id_feats=segment_id_feats
        )
