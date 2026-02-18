from typing import Dict, Optional
import torch

from .backbones import VocosBackbone
from .heads import ISTFTHead

from interface.feature import AcousticFeature
from interface.model import GeneratorOutput, Generator
from interface.loggable import Heatmap

class VocosGenerator(Generator):
    def __init__(
        self,
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        fft_size,
        hop_size,
    ):
        super(VocosGenerator, self).__init__()

        self.backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            layer_scale_init_value=None,
            adanorm_num_embeddings=None,
        )
        self.head = ISTFTHead(
             dim=dim, 
             n_fft=fft_size, 
             hop_length=hop_size, 
             padding="same"
        )


    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        
        x = input_feature["mel_spectrogram"]
        x = self.backbone(x)
        y_hat, S = self.head(x)

        y_hat = y_hat.unsqueeze(1)
        spectrogram = torch.log(torch.sqrt(S.real**2 + S.imag**2)).squeeze()

        output = GeneratorOutput(
            pred=y_hat,
            loggable_outputs={
                "spectrogram": Heatmap(data=spectrogram, origin="lower")
            }
        )
        return output