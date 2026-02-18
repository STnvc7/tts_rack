from typing import Dict, Optional
import torch

from .backbones import VocosBackbone
from .heads import WaveNextHead

from interface.feature import AcousticFeature
from interface.model import GeneratorOutput, Generator

class WaveNeXtGenerator(Generator):
    def __init__(
        self,
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        fft_size,
        hop_size,
    ):
        super(WaveNeXtGenerator, self).__init__()

        self.backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            layer_scale_init_value=None,
            adanorm_num_embeddings=None,
        )
        self.head = WaveNextHead(
             dim=dim, 
             n_fft=fft_size, 
             hop_length=hop_size, 
        )

    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        
        x = input_feature["mel_spectrogram"]
        
        x = self.backbone(x)
        y_hat = self.head(x)
        y_hat = y_hat.unsqueeze(1)

        return GeneratorOutput(pred=y_hat)