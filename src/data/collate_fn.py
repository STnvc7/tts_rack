from typing import List, Dict
import torch

from interface.data import DatasetOutput, DataLoaderOutput
from interface.feature import AcousticFeature
from utils.tensor import fix_length
from engine._common.tensor import create_mask_from_lengths

class TTSCollateFn:
    def __init__(self):
        pass
    def __call__(self, batch: List[DatasetOutput]):
        batch = [b for b in batch if b is not None]
        if batch == []:
            return None

        id_lengths = [b.phoneme_id.shape[-1] for b in batch]
        phoneme_id_mask = create_mask_from_lengths(id_lengths)
        phoneme_id = torch.stack([
            fix_length(b.phoneme_id, max(id_lengths), dim=-1) 
            for b in batch
        ])
        
        if all(b.prosody is not None and b.prosody_id is not None for b in batch):
            prosody = [b.prosody for b in batch if b.prosody is not None]
            prosody_id = torch.stack([
                fix_length(b.prosody_id, max(id_lengths), dim=-1) 
                for b in batch if b.prosody_id is not None
            ])
        else:
            prosody = None
            prosody_id = None
        
        
        if all(b.duration is not None for b in batch):
            duration = torch.stack([
                fix_length(b.duration, max(id_lengths), dim=-1) 
                for b in batch if b.duration is not None
            ])
        else:
            duration = None
    
        max_wav_length = max([b.wav.shape[-1] for b in batch])
        wav = torch.stack([
            fix_length(b.wav, max_wav_length, dim=-1) 
            for b in batch
        ])
        
        keys: List[AcousticFeature] = [k for k in batch[0].features.keys()]
        feat_lengths = [b.features[keys[0]].shape[-1] for b in batch]
        max_feat_length = max(feat_lengths)
        feat_mask = create_mask_from_lengths(feat_lengths)
        features: Dict[AcousticFeature, torch.Tensor] = {}
        for feat_name in batch[0].features.keys():
            features[feat_name] = torch.stack([
                fix_length(b.features[feat_name], max_feat_length, dim=-1) 
                for b in batch
            ])
        
        # segment ids ----------------------------------
        if all(b.segment_id_wav is not None for b in batch):
            segment_id_wav = torch.stack([
                b.segment_id_wav
                for b in batch if b.segment_id_wav is not None
            ])
        else:
            segment_id_wav = None
            
        if all(b.segment_id_feats is not None for b in batch):
            segment_id_feats = torch.stack([
                b.segment_id_feats
                for b in batch if b.segment_id_feats is not None
            ])
        else:
            segment_id_feats = None
        
        return DataLoaderOutput(
            filename=[b.filename for b in batch],
            speaker_id=torch.stack([b.speaker_id for b in batch]),
            text=[b.text for b in batch],
            phoneme=[b.phoneme for b in batch],
            phoneme_id=phoneme_id,
            phoneme_id_mask=phoneme_id_mask,
            prosody=prosody,
            prosody_id=prosody_id,
            duration=duration,
            wav=wav,
            features=features,
            feature_mask=feat_mask,
            segment_id_wav=segment_id_wav,
            segment_id_feats=segment_id_feats,
        )