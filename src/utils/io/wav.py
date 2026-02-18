from typing import Optional
import torch
import torchaudio

def load_wav(path: str, sample_rate: Optional[int]=None):
    wav, sr = torchaudio.load(path)
    if sample_rate is not None and sample_rate != sr:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    return wav, sr
    
def save_wav(wav: torch.Tensor, path: str, sample_rate: int):
    wav = wav.detach().cpu()
    if wav.ndim == 1:   # (Time) -> (1, Time)
        wav = wav.unsqueeze(0)
    elif wav.ndim > 2:  # (Batch, ..., Time) -> (Time) -> (1, Time) or (C, T)
        wav = wav.squeeze()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"Waveform must be 2D (Channel, Time), but got shape {wav.shape}")
    
    torchaudio.save(
        uri=path,
        src=wav.to(torch.float32),
        sample_rate=sample_rate,
        format="wav",
        bits_per_sample=32,
    )