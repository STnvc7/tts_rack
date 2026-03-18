import math
import torch
from torch import nn
from torch.nn import functional as F

from .wavenet import WN
from .attentions import Encoder

class TextEncoder(nn.Module):
    def __init__(self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
    
        self.emb = nn.Embedding(n_vocab+1, hidden_channels, padding_idx=0)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
    
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )
        self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # self.proj.weight.data.zero_() # for training stability
        # if self.proj.bias is not None:
        #     self.proj.bias.data.zero_()
    
    def forward(self, x, x_mask):
        x = self.emb(x)
        x = x * math.sqrt(self.hidden_channels) # [b, t, h]
        x = torch.transpose(x, 1, -1) # [b, h, t]
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
    
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs


class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
    
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # self.proj.weight.data.zero_() # for training stability
        # if self.proj.bias is not None:
        #     self.proj.bias.data.zero_()
        
    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs
    
