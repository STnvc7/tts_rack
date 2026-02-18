import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalize import LayerNorm1d, AdaptiveNorm1d, CouplingNorm1d
from .shuffle import Shuffle1d

class ConvNeXtBlock1d(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        kernel_size,
        n_groups,
        scale, 
    ):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=channels)
        self.norm = LayerNorm1d(channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1, groups=n_groups)
        self.shuffle = Shuffle1d(n_groups)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1, groups=n_groups)
        self.scale = nn.Parameter(torch.full(size=(1, channels, 1), fill_value=scale), requires_grad=True)
    
    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.shuffle(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = res + x
        return x
        
class AdaptiveConvNeXtBlock1d(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        kernel_size,
        n_groups,
        scale, 
    ):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=channels)
        self.norm = AdaptiveNorm1d(channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1, groups=n_groups)
        self.shuffle = Shuffle1d(n_groups)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1, groups=n_groups)
        self.scale = nn.Parameter(torch.full(size=(1, channels, 1), fill_value=scale), requires_grad=True)
    
    def forward(self, x, c):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x, c)
        x = self.pw_conv1(x)
        x = self.shuffle(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = res + x
        return x
        
        
class CouplingBlock1d(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        kernel_size,
        n_groups,
        scale, 
    ):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=channels)
        self.norm = CouplingNorm1d(channels)
        self.pw_conv1 = nn.Conv1d(channels, h_channels, 1, groups=n_groups)
        self.shuffle = Shuffle1d(n_groups)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv1d(h_channels, channels, 1, groups=n_groups)
        self.scale = nn.Parameter(torch.full(size=(1, channels, 1), fill_value=scale), requires_grad=True)
    
    def forward(self, x, c=None):
        res = x
        x = self.dw_conv(x)
        x, loss = self.norm(x, c)
        x = self.pw_conv1(x)
        x = self.shuffle(x)
        x = self.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = res + x
        return x, loss