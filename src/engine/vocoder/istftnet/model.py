from typing import Dict, Optional
import torch

from .backbone import ISTFTNetBackbone
from .stft import TorchSTFT

from interface.model import GeneratorOutput, Generator
from interface.feature import AcousticFeature
from interface.loggable import Heatmap

class iSTFTNetGenerator(Generator):
    def __init__(
        self,
        input_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
    ):
        super(iSTFTNetGenerator, self).__init__()
        self.backbone = ISTFTNetBackbone(
            input_channel=input_channel,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
        )
        self.head = TorchSTFT(
            filter_length=gen_istft_n_fft, 
            hop_length=gen_istft_hop_size, 
            win_length=gen_istft_n_fft, 
        )
    
    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        
        mel = input_feature["mel_spectrogram"]

        spec, phase = self.backbone(mel)
        y = self.head.inverse(spec, phase)

        output = GeneratorOutput(
            pred=y,
            loggable_outputs={
                "mag": Heatmap(data=spec.squeeze()),
                "phase": Heatmap(data=phase.squeeze())
            }
        )
        return output