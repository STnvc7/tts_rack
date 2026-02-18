from typing import Dict, Optional, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from interface.model import GeneratorOutput, Generator
from interface.feature import AcousticFeature
from utils.tensor import fix_length
from .diffusion import Diffusion
from .module import WaveGradGenerator

class WaveGrad(Generator):
    def __init__(
        self,
        n_mels,
        upsample_rate=[4,4,4,2,2],
        upsample_channels=[(768, 512), (512, 512), (512, 256), (256, 128), (128, 128)],
        upsample_dilations=[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
        downsample_channels=[(1, 32), (32, 128), (128, 128), (128, 256), (256, 512)],
        beta_min=1e-6,
        beta_max=0.01,
        beta_step=1000,
        inference_steps=25,
    ):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.inference_steps = inference_steps
        model = WaveGradGenerator(
            n_mels=n_mels,
            upsample_rate=upsample_rate,
            upsample_channels=upsample_channels,
            upsample_dilations=upsample_dilations,
            downsample_channels=downsample_channels,
        )
        self.diffusion = Diffusion(
            model=model,
            upsample_rate=math.prod(upsample_rate),
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
        
        upsample_rate = math.prod(self.upsample_rate)
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
        
        schedule = torch.linspace(self.beta_min, self.beta_max, 100)
        schedule = schedule.to(mel.device)
        pred = self.diffusion(mel, schedule)
        return GeneratorOutput(pred=pred)
        