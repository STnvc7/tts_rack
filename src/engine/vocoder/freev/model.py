from typing import Dict, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
import torchaudio
from dsp_board.transforms import istft

from interface.model import GeneratorOutput, Generator
from interface.feature import AcousticFeature
from interface.loggable import Heatmap

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def inverse_mel(mel, sample_rate, fft_size, n_mels) -> torch.Tensor:
    mel_filter = torchaudio.functional.melscale_fbanks(
        n_freqs=fft_size//2+1,
        f_min=0,
        f_max=sample_rate // 2,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=None,
        mel_scale='htk'
    ).to(mel.device)

    inv_filter = mel_filter.pinverse().T
    pseudo_spc =  torch.matmul(inv_filter, mel.exp())
    return pseudo_spc


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x
    

class FreeVGenerator(Generator):
    def __init__(
        self,
        sample_rate,
        n_mels,
        fft_size,
        hop_size,
        psp_channel,
        psp_input_conv_kernel_size,
        psp_output_real_conv_kernel_size,
        psp_output_imag_conv_kernel_size,
        convnext_intermediate_dim,
        asp_convnext_n_layers,
        psp_convnext_n_layers,
    ):
        super(FreeVGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fft_size = fft_size
        self.hop_size = hop_size

        self.asp_convnext = nn.ModuleList([
                ConvNeXtBlock(
                    dim=fft_size//2+1,
                    intermediate_dim=convnext_intermediate_dim,
                )for _ in range(asp_convnext_n_layers)
        ])
        
        self.psp_input_conv = Conv1d(
            n_mels, psp_channel, psp_input_conv_kernel_size, padding=get_padding(psp_input_conv_kernel_size)
        )
        self.psp_input_norm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.psp_convnext = nn.ModuleList([
                ConvNeXtBlock(
                    dim=psp_channel,
                    intermediate_dim=convnext_intermediate_dim,
                ) for _ in range(psp_convnext_n_layers)
        ])

        self.psp_output_real_conv = Conv1d(
            psp_channel, fft_size//2+1, psp_output_real_conv_kernel_size, padding=get_padding(psp_output_real_conv_kernel_size)
        )
        self.psp_output_imag_conv = Conv1d(
            psp_channel, fft_size//2+1, psp_output_imag_conv_kernel_size, padding=get_padding(psp_output_imag_conv_kernel_size)
        )
        self.psp_final_norm = nn.LayerNorm(psp_channel, eps=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        
        mel = input_feature["mel_spectrogram"]
        
        inv_mel = inverse_mel(mel, self.sample_rate, self.fft_size, self.n_mels) 
        logamp = inv_mel.abs().clip(min=1e-5).log()   
        for m in self.asp_convnext:
            logamp = m(logamp)

        pha = self.psp_input_conv(mel)
        pha = self.psp_input_norm(pha.transpose(1, 2)).transpose(1, 2)
        for m in self.psp_convnext:
            pha = m(pha)
        pha = self.psp_final_norm(pha.transpose(1, 2)).transpose(1, 2)
        real = self.psp_output_real_conv(pha)
        imag = self.psp_output_imag_conv(pha)

        pha = torch.atan2(imag, real)

        real = torch.exp(logamp) * torch.cos(pha)
        imag = torch.exp(logamp) * torch.sin(pha)

        spc = torch.complex(real, imag)
        audio = istft(spc, self.fft_size, self.hop_size)
        
        output = GeneratorOutput(
            pred=audio.unsqueeze(1),
            outputs={
                "real": real,
                "imag": imag,
                "logamp": logamp,
                "phase": pha,
            },
            loggable_outputs={
                "mag": Heatmap(logamp.squeeze()),
                "phase": Heatmap(pha.squeeze()),
            }
        )
        return output
