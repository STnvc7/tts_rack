import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import AdaptiveConvNeXtBlock1d

class SinusoidalPosEmb(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        assert self.channel % 2 == 0, "SinusoidalPosEmb requires channel to be even"

    def forward(self, x, scale=1000):
        x = x[:, None]
        device = x.device
        half_ch = self.channel // 2
        emb = math.log(10000) / (half_ch - 1)
        emb = torch.exp(torch.arange(half_ch, device=device).float() * -emb)
        emb = scale * x * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Estimator1d(nn.Module):
    def __init__(
        self,
        n_bins,
        channels,
        h_channels,
        kernel_size,
        n_layers,
        n_groups,
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels*2),
        )
        self.pre_conv = nn.Conv1d(n_bins, channels, 1)
        self.conv = nn.ModuleList([
            AdaptiveConvNeXtBlock1d(
                channels=channels,
                h_channels=h_channels,
                kernel_size=kernel_size,
                n_groups=n_groups,
                scale=1./n_layers
            ) for _ in range(n_layers)
        ])
        self.post_conv = nn.Conv1d(channels, n_bins, 1)

    def forward(self, z, c, t):
        z = self.pre_conv(z)
        z = z + c
        
        t = self.time_emb(t)[:, :, None].expand(-1, -1, z.shape[-1])
        scale, shift = t.chunk(2, dim=1)
        for conv in self.conv:
            z = conv(z, c=(scale, shift))
        z = self.post_conv(z)
        return z

class CFM1d(nn.Module):
    def __init__(
        self,
        n_bins,
        channels,
        h_channels,
        kernel_size,
        n_layers,
        n_groups,
    ):
        super().__init__()
        self.estimator = Estimator1d(
            n_bins,
            channels,
            h_channels,
            kernel_size,
            n_layers,
            n_groups,
        )
        
    @torch.inference_mode()
    def forward(self, x0, c, n_timesteps, mask):
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=x0.device)
        return self.solve_euler(x0, c, t_span, mask)

    def solve_euler(self, x0, c, t_span, mask):
        z = x0
        for i in range(len(t_span) - 1):
            t_curr = t_span[i]
            t_next = t_span[i + 1]
            dt = t_next - t_curr

            t_input = t_curr.view(1).repeat(z.shape[0])
            dphi_dt = self.estimator(z, c, t_input) * mask
            z = z + dt * dphi_dt
        return z

    def compute_loss(self, target, x0, c, mask=None):
        B, _, _ = target.shape
        device = target.device
        dtype = target.dtype

        t = torch.rand(B, device=device, dtype=dtype)
        z = (1 - t[:, None, None]) * x0 + t[:, None, None] * target
        y = target - x0
        pred = self.estimator(z, c, t) * mask
        
        if mask is None:
            mask = torch.ones_like(y)
        pred_flatten = pred.masked_select(mask)
        y_flatten = y.masked_select(mask)
        loss = F.mse_loss(pred_flatten, y_flatten)
        return loss
