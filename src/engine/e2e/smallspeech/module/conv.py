import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import AdaptiveNorm1d
from .act import Activation

class ConvNeXtBlock(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        kernel_size,
        n_layers,
        n_groups=1,
        act="gelu"
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            ConvNeXtLayer(
                channels=channels, 
                h_channels=h_channels, 
                kernel_size=kernel_size, 
                n_groups=n_groups,
                act=act,
            )
            for _ in range(n_layers)
        ])
    
    def forward(self, x, x_mask, task_emb):
        for layer in self.layers:
            x = layer(x, task_emb) * x_mask
        return x
        
class Shuffle1d(nn.Module):
    def __init__(self, n_groups):
        super().__init__()
        self.n_groups = n_groups
    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B, self.n_groups, C//self.n_groups, L)
        x = x.permute(0,2,1,3)
        x = x.reshape(B,C,L)
        return x
        
class ConvNeXtLayer(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        kernel_size,
        n_groups,
        act,
    ):
        super().__init__()
        self.norm = AdaptiveNorm1d(channels)
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1, groups=n_groups)
        self.shuffle = Shuffle1d(n_groups)
        self.act = Activation(act)
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1, groups=n_groups)
        
    def forward(self, x, task_emb):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x, task_emb.norm_scale, task_emb.norm_shift)
        x = self.pw_conv1(x)
        x = self.shuffle(x)
        x = self.act(x, task_emb.act_scale, task_emb.act_shift)
        x = self.pw_conv2(x)
        x = res + x
        return x