import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def osc(wavetable: torch.Tensor, f0: torch.Tensor, sample_rate: int, hop_size: int):
    """
    Memory efficient oscillator with bilinear interpolation (Time x Phase).
    
    Args:
        wavetable (torch.Tensor): (batch, n_sample, n_frames)
        f0 (torch.Tensor): (batch, 1, n_frames)
    """
    B, N, L = wavetable.shape
    f0 = F.interpolate(f0, scale_factor=float(hop_size), mode="linear", align_corners=True)
    _, _, T = f0.shape

    idx_delta = (N * f0) / sample_rate
    phase = torch.cumsum(idx_delta.double(), dim=-1).float() % N

    batch_idx = torch.arange(B, device=f0.device).view(B, 1, 1)
    time_idx = torch.arange(T, device=f0.device).float() / hop_size
    time_idx = time_idx.view(1, 1, T).expand(B, -1, -1).clip(min=0, max=L-1)

    p_floor = phase.long().clip(min=0, max=N-1)
    p_ceil = (p_floor + 1) % N
    p_frac = phase - p_floor.float()

    t_floor = time_idx.long().clip(min=0, max=L-1)
    t_ceil = (t_floor + 1).clip(min=0, max=L-1)
    t_frac = time_idx - t_floor.float()

    v00 = wavetable[batch_idx, p_floor, t_floor] # (Time: floor, Phase: floor)
    v01 = wavetable[batch_idx, p_floor, t_ceil]  # (Time: ceil,  Phase: floor)
    v10 = wavetable[batch_idx, p_ceil, t_floor]  # (Time: floor, Phase: ceil)
    v11 = wavetable[batch_idx, p_ceil, t_ceil]   # (Time: ceil,  Phase: ceil)

    # 時間軸方向の補間 (Wavetable morphing)
    v0 = v00 + t_frac * (v01 - v00)
    v1 = v10 + t_frac * (v11 - v10)

    # 位相軸方向の補間 (Wavetable lookup)
    output = v0 + p_frac * (v1 - v0)

    return output

@torch.jit.script
def anti_alias(magnitude: torch.Tensor, f0: torch.Tensor, sample_rate: int, n_sample: int):
    """remove the harmonics above nyquist frequency
    Args:
        mag (torch.Tensor): (batch, bin, frame)
        f0 (torch.Tensor): (batch, 1, frame)

    Returns:
        torch.Tensor: (batch, bin, frame)
    """
    nyquist = sample_rate / 2
    n_harmonics = torch.floor(nyquist /  f0.clamp(min=1e-5))
    
    B, N, T = magnitude.shape
    harmonic_indices = torch.arange(N).reshape(1, N, 1).expand(B, N, T).to(magnitude.device)
    mask = harmonic_indices > n_harmonics

    magnitude = magnitude.masked_fill(mask, 0)

    return magnitude