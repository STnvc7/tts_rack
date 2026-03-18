import torch
import torch.nn as nn

class TaskEmbedding(nn.Module):
    def __init__(self, channels, h_channels, kernel_size, n_groups):
        super().__init__()
        self.norm_shift = nn.Parameter(torch.zeros(1,channels, 1))
        self.norm_scale = nn.Parameter(torch.ones(1, channels, 1))
        
        self.act_shift = nn.Parameter(torch.zeros(1, h_channels, 1))
        self.act_scale = nn.Parameter(torch.ones(1, h_channels, 1))
        
        self.dw_conv_shift = nn.Parameter(torch.zeros(channels, 1, kernel_size))
        self.dw_conv_scale = nn.Parameter(torch.ones(channels, 1, kernel_size))
        self.pw_conv1_shift = nn.Parameter(torch.zeros(1, channels//n_groups, 1))
        self.pw_conv1_scale = nn.Parameter(torch.ones(1, channels//n_groups, 1))
        self.pw_conv2_shift = nn.Parameter(torch.zeros(1, h_channels//n_groups, 1))
        self.pw_conv2_scale = nn.Parameter(torch.ones(1, h_channels//n_groups, 1))
