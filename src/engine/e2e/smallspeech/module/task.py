import torch
import torch.nn as nn

class TaskEmbedding(nn.Module):
    def __init__(self, channels, h_channels):
        super().__init__()
        self.norm_shift = nn.Parameter(torch.zeros(1,channels, 1))
        self.norm_scale = nn.Parameter(torch.ones(1, channels, 1))
        
        self.act_shift = nn.Parameter(torch.zeros(1, h_channels, 1))
        self.act_scale = nn.Parameter(torch.ones(1, h_channels, 1))