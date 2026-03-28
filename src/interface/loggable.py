from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal, List, Callable
import wandb
from dsp_board.features import log_spectrogram

import torch
import numpy as np

from utils.tensor import to_numpy
from utils import plot
from utils.io.wav import save_wav

class Loggable(ABC):
    @abstractmethod
    def to_wandb_media(self) -> float | wandb.Audio | wandb.Image | wandb.Video:
        raise NotImplementedError
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

@dataclass
class Audio(Loggable):
    data: torch.Tensor
    sample_rate: int
    def to_wandb_media(self) -> wandb.Audio:
        data = self.data.squeeze()
        assert data.ndim == 1, "Audio data must be 1-dim tensor"
        return wandb.Audio(to_numpy(data, np.float32), self.sample_rate)
    def save(self, path: str):
        save_wav(self.data, path, self.sample_rate)

@dataclass
class Heatmap(Loggable):
    data: torch.Tensor
    label: Optional[str] = None
    origin: Literal["upper", "lower"] = "lower"
    def to_wandb_media(self) -> wandb.Image:
        data = self.data.squeeze()
        assert data.ndim==2, "Heatmap data must be 2-dim tensor"
        return wandb.Image(plot.imshow(data, title=self.label, origin=self.origin))
    def save(self, path: str):
        data = self.data.squeeze()
        assert data.ndim==2, "Heatmap data must be 2-dim tensor"
        fig = plot.imshow(data, title=self.label, origin=self.origin)
        fig.savefig(fname=path)

@dataclass
class Heatmap3D(Loggable):
    data: torch.Tensor
    label: Optional[str] = None
    anchor: int = 5
    def to_wandb_media(self) -> wandb.Image:
        data = self.data.squeeze()
        assert data.ndim==2, "Heatmap data must be 2-dim tensor"
        return wandb.Image(plot.heatmap_3d(data, title=self.label, anchor=self.anchor))
    def save(self, path: str):
        data = self.data.squeeze()
        assert data.ndim==2, "Heatmap data must be 2-dim tensor"
        fig = plot.heatmap_3d(data, title=self.label, anchor=self.anchor)
        fig.savefig(fname=path)

@dataclass
class Sequence(Loggable):
    data: torch.Tensor
    label: Optional[str] = None
    def to_wandb_media(self) -> wandb.Image:
        data = self.data.squeeze()
        assert data.ndim==1, "Sequence data must be 1-dim tensor"
        return wandb.Image(plot.plot(data, title=self.label))
    def save(self, path: str):
        data = self.data.squeeze()
        assert data.ndim==1, "Sequence data must be 1-dim tensor"
        fig = plot.plot(data, title=self.label)
        fig.savefig(fname=path)

@dataclass
class Scalar(Loggable):
    data: torch.Tensor
    def to_wandb_media(self) -> float:
        return self.data.cpu().detach().clone().item()
    def save(self, path: str):
        return
        
@dataclass
class Spectrogram(Loggable):
    data: torch.Tensor # wav data!
    fft_size: int = 1024
    hop_size: int = 256
    label: Optional[str] = None

    def to_wandb_media(self):
        wav = self.data.squeeze()
        assert wav.ndim==1, "Spectrogram data must be 1-dim wav tensor"
        spc = log_spectrogram(wav, self.fft_size, self.hop_size)
        return wandb.Image(plot.imshow(spc, title=self.label, origin="lower"))
    
    def save(self, path: str):
        wav = self.data.squeeze()
        assert wav.ndim==1, "Spectrogram data must be 1-dim wav tensor"
        spc = log_spectrogram(wav, self.fft_size, self.hop_size)
        fig = plot.imshow(spc, title=self.label, origin="lower")
        fig.savefig(fname=path)

@dataclass
class Duration(Loggable):
    phoneme: List[str]
    duration: torch.Tensor
    spectrogram: torch.Tensor
    label: Optional[str] = None
    def to_wandb_media(self) -> wandb.Image:
        assert self.duration.shape[0] == len(self.phoneme), "Duration and phoneme length mismatch"
        return wandb.Image(plot.duration(self.duration, self.phoneme, self.spectrogram, self.label))
    def save(self, path: str):
        assert self.duration.shape[0] == len(self.phoneme), "Duration and phoneme length mismatch"
        fig = plot.duration(self.duration, self.phoneme, self.spectrogram, self.label)
        fig.savefig(fname=path)
        
        
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def _default_loggable_factory(
    data: torch.Tensor, 
    label: Optional[str] = None
) -> Loggable:
    data = data.squeeze()
    if data.ndim == 1:
        return Sequence(data, label)
    elif data.ndim == 2:
        return Heatmap(data, label, origin="lower")
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}")

def tensor_to_loggable(
    data: torch.Tensor, 
    label: Optional[str]=None, 
    factory: Callable[..., Loggable]=_default_loggable_factory
) -> Loggable:
    return factory(data, label)