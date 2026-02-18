import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x
        
class AdaptiveNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        
    def forward(self, x, c):
        assert isinstance(c, (tuple, list))
        scale, shift = c
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x * scale + shift
        return x
        
class CouplingNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.conv_c = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, channels*2, 1),
        )
        self.conv_x = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, channels*2, 1),
        )
        
    def forward(self, x, c=None):
        scale_x, shift_x = self.conv_x(x).chunk(2, dim=1)
        if c is not None and self.training:
            scale_c, shift_c = self.conv_c(c).chunk(2, dim=1)
            loss = F.l1_loss(scale_c.detach(), scale_x) + F.l1_loss(shift_c.detach(), shift_x)
            scale, shift = scale_c, shift_c
        else:
            scale, shift = scale_x, shift_x
            loss = torch.tensor(0.0, device=x.device)
            
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x * scale + shift
        return x, loss
        