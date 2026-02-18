from typing import Dict, Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from interface.model import GeneratorOutput, Generator
from interface.feature import AcousticFeature
from utils.tensor import fix_length
from .diffusion import Diffusion
from .module import DiffWaveGenerator

class DiffWave(Generator):
    def __init__(
        self,
        n_mels,
        upsample_rates,
        residual_channels,
        residual_layers,
        dilation_cycle_length,
        beta_min,
        beta_max,
        beta_step,
        inference_schedule,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.inference_schedule = torch.tensor(inference_schedule)
        model = DiffWaveGenerator(
            n_mels=n_mels,
            upsample_rates=upsample_rates,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
            n_steps=beta_step,
        )
        self.diffusion = Diffusion(
            model=model,
            upsample_rate=math.prod(upsample_rates),
            beta_min=beta_min,
            beta_max=beta_max,
            beta_step=beta_step,
        )
        
    def scale_melspectrogram(self, mel):
        mel = mel.exp()
        mel = 20 * torch.log10(mel.clip(min=1e-5)) - 20
        mel = (mel + 100) / 100
        mel = mel.clip(min=0, max=1)
        return mel
    
    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        if not self.training:
            return self.inference(input_feature)
        
        assert wav is not None
        mel = input_feature["mel_spectrogram"]
        mel = self.scale_melspectrogram(mel)
        
        upsample_rate = math.prod(self.upsample_rates)
        if wav.shape[-1] % upsample_rate != 0:
            mod = wav.shape[-1] % upsample_rate
            length = wav.shape[-1] + upsample_rate - mod
            wav = fix_length(wav, length, dim=-1)
            mel = fix_length(mel, length//upsample_rate, dim=-1)
        
        loss = self.diffusion.compute_loss(wav, mel)
        pred = torch.zeros_like(wav)
        return GeneratorOutput(pred=pred, outputs={"loss": loss})
    
    def inference(
        self,
        input_feature: Dict[AcousticFeature, torch.Tensor],
        control: Optional[Any]=None
    ) -> GeneratorOutput:
        mel = input_feature["mel_spectrogram"]
        mel = self.scale_melspectrogram(mel)
        
        schedule = self.inference_schedule.to(mel.device)
        pred = self.diffusion(mel, schedule)
        return GeneratorOutput(pred=pred)
        