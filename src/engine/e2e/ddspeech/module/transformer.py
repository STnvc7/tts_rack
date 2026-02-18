import math
import torch
from torch import nn
from torch.nn import functional as F

from .normalize import LayerNorm1d
from .rope import RoPE
from .shuffle import Shuffle1d

class TransformerLayer(nn.Module):
    def __init__(
        self,
        channels, 
        h_channels, 
        n_heads, 
        kernel_size=1, 
        p_dropout=0., 
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            channels=channels,
            out_channels=channels, 
            n_heads=n_heads,
            p_dropout=p_dropout, 
        )
        self.norm1 = LayerNorm1d(channels)
        self.ffn = FFN(
            in_channels=channels, 
            out_channels=channels, 
            h_channels=h_channels, 
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        self.norm2 = LayerNorm1d(channels)
        self.drop = nn.Dropout(p_dropout)
    
    def forward(self, x, x_mask=None, attn_mask=None):
        res = x
        x = self.attention(x, x, attn_mask)
        x = self.drop(x)
        x = x + res
        x = self.norm1(x)
        
        res = x
        x = self.ffn(x, x_mask)
        x = self.drop(x)
        x = x + res
        x = self.norm2(x)
        return x

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        n_heads,
        n_layers, 
        kernel_size=1, 
        p_dropout=0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                channels=channels, 
                h_channels=h_channels, 
                n_heads=n_heads, 
                kernel_size=kernel_size, 
                p_dropout=p_dropout, 
            ) for _ in range(n_layers)
        ])
    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for l in self.layers:
            x = l(x, x_mask, attn_mask)
        x = x * x_mask
        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        group=True
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        n_groups = n_heads if group else 1
        self.conv_q = nn.Conv1d(channels, channels, 1, groups=n_groups)
        self.conv_k = nn.Conv1d(channels, channels, 1, groups=n_groups)
        self.conv_v = nn.Conv1d(channels, channels, 1, groups=n_groups)
        self.rope = RoPE(channels // n_heads, partial=True)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        B, C, T = query.shape
        C_K = C // self.n_heads
        query = query.view(B, self.n_heads, C_K, T)
        key = key.view(B, self.n_heads, C_K, T)
        value = value.view(B, self.n_heads, C_K, T)

        query = self.rope(query)
        key = self.rope(key)
        
        attn = torch.matmul(query.transpose(2, 3), key) / math.sqrt(C_K)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        output = torch.matmul(value, attn)
        output = output.view(B, C, T)
        return output


class FFN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        h_channels, 
        kernel_size,
        p_dropout=0., 
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, h_channels, kernel_size, padding=(kernel_size-1)//2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(h_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x * x_mask)
        return x * x_mask
      
class FFN2(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        h_channels, 
        n_groups,
        p_dropout=0., 
    ):
        super().__init__()
        self.pw1 = nn.Conv1d(in_channels, h_channels, 1, groups=n_groups)
        self.shuffle = Shuffle1d(n_groups)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(h_channels, out_channels, 1, groups=n_groups)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = x * x_mask
        x = self.pw1(x)
        x = self.shuffle(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        return x * x_mask