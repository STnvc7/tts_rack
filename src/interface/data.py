from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from interface.feature import AcousticFeature

@dataclass
class DatasetOutput:
    filename: str
    speaker_id: torch.Tensor
    text: str
    phoneme: str
    phoneme_id: torch.Tensor
    prosody: Optional[str]
    prosody_id: Optional[torch.Tensor]
    duration: Optional[torch.Tensor]
    wav: torch.Tensor
    features: Dict[AcousticFeature, torch.Tensor]
    segment_id_wav: Optional[torch.Tensor]
    segment_id_feats: Optional[torch.Tensor]

@dataclass
class DataLoaderOutput:
    filename: List[str]
    speaker_id: torch.Tensor
    text: List[str]
    phoneme: List[str]
    phoneme_id: torch.Tensor
    phoneme_id_mask: torch.Tensor
    prosody: Optional[List[str]]
    prosody_id: Optional[torch.Tensor]
    duration: Optional[torch.Tensor]
    wav: torch.Tensor
    features: Dict[AcousticFeature, torch.Tensor]
    feature_mask: torch.Tensor
    segment_id_wav: Optional[torch.Tensor]
    segment_id_feats: Optional[torch.Tensor]
