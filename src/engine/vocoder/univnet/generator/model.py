from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from interface.model import GeneratorOutput, Generator
from interface.feature import AcousticFeature
from .lvcnet import LVCBlock

MAX_WAV_VALUE = 32768.0

class UnivNetGenerator(Generator):
    """UnivNet Generator"""
    def __init__(
        self,
        n_mel_channels,
        noise_dim,
        channel_size,
        dilations,
        strides,
        lReLU_slope,
        kpnet_conv_size,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        channel_size = channel_size
        kpnet_conv_size = kpnet_conv_size

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    n_mel_channels,
                    stride=stride,
                    dilations=dilations,
                    lReLU_slope=lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )
        
        self.conv_pre = weight_norm(nn.Conv1d(
            noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        c = input_feature["mel_spectrogram"]
        B, C, T = c.shape
        z = torch.randn(B, self.noise_dim, T).to(c.device)

        z = self.conv_pre(z)                # (B, c_g, L)

        for res_block in self.res_stack:
            z = res_block(z, c)             # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)

        output = GeneratorOutput(pred=z)

        return output

    def remove_weight_norm(self):
        print('Removing weight norm...')

        remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                remove_weight_norm(layer)

        for res_block in self.res_stack:
            remove_weight_norm(res_block)