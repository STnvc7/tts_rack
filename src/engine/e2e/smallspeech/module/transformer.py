import math
import torch
from torch import nn
from torch.nn import functional as F

from .norm import LayerNorm1d

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        channels, 
        h_channels, 
        n_heads,
        n_layers, 
        p_dropout=0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                channels=channels, 
                h_channels=h_channels, 
                n_heads=n_heads, 
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

class TransformerLayer(nn.Module):
    def __init__(
        self,
        channels, 
        h_channels, 
        n_heads,
        p_dropout=0.,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            channels=channels,
            out_channels=channels, 
            n_heads=n_heads,
            p_dropout=p_dropout, 
        )
        self.ffn = FFN(
            in_channels=channels, 
            out_channels=channels, 
            h_channels=h_channels, 
            p_dropout=p_dropout,
        )
        self.drop = nn.Dropout(p_dropout)
        
        self.norm1 = LayerNorm1d(channels)
        self.norm2 = LayerNorm1d(channels)

    def forward(self, x, x_mask, attn_mask):
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

class RoPE(nn.Module):
    def __init__(self, channel, base: int = 10_000, partial=True, cache_length=200):
        super().__init__()
        self.base = base
        self.partial = partial
        self.rope_channel = channel // 2 if partial else channel
        self.cache_length = cache_length
        dummy = torch.zeros(1, 1, self.rope_channel, cache_length)
        cos, sin = self._emb(dummy)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
    def _emb(self, x: torch.Tensor):
        # x: [B, H, C, T]
        # cos: [1, 1, C, T], sin: [1, 1, C, T]
        
        k = torch.arange(0, self.rope_channel, 2, device=x.device).float()
        theta = 1.0 / (self.base ** (k / self.rope_channel))
        frame_idx = torch.arange(x.shape[-1], device=x.device).float()
        angles = theta.unsqueeze(1) * frame_idx.unsqueeze(0) # [C//2, T]
        angles = torch.cat([angles, angles], dim=0) # [C, T]
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
        
    def _neg_half(self, x: torch.Tensor):
        _, _, C, _ = x.shape
        return torch.cat([-x[:, :, C//2:, :], x[:, :, :C//2, :]], dim=2)

    def forward(self, x: torch.Tensor):
        """
        x: (b, n_head, channel//n_head, frame)
        """
        _, _, C, T = x.shape
        if T > self.cache_length:
            cos, sin = self._emb(x)
        else:
            cos, sin = self.cos[:, :, :, :T], self.sin[:, :, :, :T] # type: ignore
        
        if self.partial:
            x_rope, x_pass = x[:, :, :C//2, :], x[:, :, C//2:, :]
            neg_half_x = self._neg_half(x_rope)
            x_rope = (x_rope * cos) + (neg_half_x * sin)
            x = torch.cat([x_rope, x_pass], dim=2)
        else:
            neg_half_x = self._neg_half(x)
            x = x * cos + neg_half_x * sin
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        group=False
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
        p_dropout=0.,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, h_channels, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(h_channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x * x_mask)
        return x * x_mask