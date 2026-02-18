import torch
import torch.nn as nn

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

