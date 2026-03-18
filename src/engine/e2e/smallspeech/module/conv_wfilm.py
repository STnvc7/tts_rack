import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import AdaptiveNorm1d
from .act import Activation

class WeightFiLMConvNeXtBlock(nn.Module):
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
    
class DWConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.channels = channels
        self.padding = (kernel_size - 1) // 2
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
    def forward(self, x, w_scale, w_shift):
        """
        x: [B, C, T]
        w_scale: [C, 1, K]
        w_shift: [C, 1, K]
        """
        weight = self.weight * w_scale + w_shift
        return F.conv1d(x, weight, bias=None, padding=self.padding, groups=self.channels)

class PWConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups):
        super().__init__()
        self.n_groups = n_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels//n_groups, 1))
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
    def forward(self, x, w_scale, w_shift):
        """
        x: [B, C, T]
        w_scale: [1, C, 1]
        w_shift: [1, C, 1]
        """
        weight = self.weight * w_scale + w_shift
        return F.conv1d(x, weight, bias=None, groups=self.n_groups)

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
        self.dw_conv = DWConv1d(channels, kernel_size)
        self.pw_conv1 = PWConv1d(channels, h_channels, n_groups)
        self.pw_conv2 = PWConv1d(h_channels, channels, n_groups)
        self.shuffle = Shuffle1d(n_groups)
        self.act = Activation(act)
    
    def forward(self, x, task_emb):
        res = x
        x = self.dw_conv(x, task_emb.dw_conv_scale, task_emb.dw_conv_shift)
        x = self.norm(x, task_emb.norm_scale, task_emb.norm_shift)
        x = self.pw_conv1(x, task_emb.pw_conv1_scale, task_emb.pw_conv1_shift)
        x = self.shuffle(x)
        x = self.act(x, task_emb.act_scale, task_emb.act_shift)
        x = self.pw_conv2(x, task_emb.pw_conv2_scale, task_emb.pw_conv2_shift)
        x = res + x
        return x