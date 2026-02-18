from typing import Optional
import torch
from torch import nn


class WaveNextHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        l_fft = n_fft + 2
        l_shift = hop_length
        self.linear_1 = torch.nn.Linear(dim, l_fft)
        self.linear_2 = torch.nn.Linear(l_fft, l_shift, bias=False)

        # W init
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.shape
        x = self.linear_1(x)
        x = self.linear_2(x)
        audio = x.view(B,-1)
        audio = torch.clip(audio, min=-1.0, max=1.0)
        return audio
