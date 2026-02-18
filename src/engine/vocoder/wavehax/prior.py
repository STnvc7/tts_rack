# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Functions for generating prior waveforms."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class WavehaxPriorGenerator(nn.Module):
    def __init__(
        self,
        hop_length: int,
        sample_rate: int,
        noise_amplitude: Optional[float] = 0.01,
        random_init_phase: Optional[bool] = True,
        power_factor: Optional[float] = 0.1,
        max_frequency: Optional[float] = None,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.noise_amplitude = noise_amplitude
        self.random_init_phase = random_init_phase
        self.power_factor = power_factor
        self.max_frequency = max_frequency

    @torch.inference_mode()
    def forward(self, f0: Tensor) -> Tensor:
        batch, _, frames = f0.size()
        device = f0.device
        noise = self.noise_amplitude * torch.randn(
            (batch, 1, frames * self.hop_length), device=device
        )
        if torch.all(f0 == 0.0):
            return noise

        vuv = f0 > 0
        min_f0_value = torch.min(f0[f0 > 0]).item()
        max_frequency = self.max_frequency if self.max_frequency is not None else self.sample_rate / 2
        max_n_harmonics = int(max_frequency / min_f0_value)
        n_harmonics = torch.ones_like(f0, dtype=torch.float)
        n_harmonics[vuv] = self.sample_rate / 2.0 / f0[vuv]

        indices = torch.arange(1, max_n_harmonics + 1, device=device).reshape(1, -1, 1)
        harmonic_f0 = f0 * indices

        # Compute harmonic mask
        harmonic_mask = harmonic_f0 <= (self.sample_rate / 2.0)
        harmonic_mask = torch.repeat_interleave(harmonic_mask, self.hop_length, dim=2)

        # Compute harmonic amplitude
        harmonic_amplitude = vuv * self.power_factor * torch.sqrt(2.0 / n_harmonics)
        harmocic_amplitude = torch.repeat_interleave(harmonic_amplitude, self.hop_length, dim=2)

        # Generate sinusoids
        f0 = torch.repeat_interleave(f0, self.hop_length, dim=2)
        radious = f0.to(torch.float64) / self.sample_rate
        if self.random_init_phase:
            radious[..., 0] += torch.rand((1, 1), device=device)
        radious = torch.cumsum(radious, dim=2)
        harmonic_phase = 2.0 * torch.pi * radious * indices
        harmonics = torch.sin(harmonic_phase).to(torch.float32)

        # Multiply coefficients to the harmonic signal
        harmonics = harmonic_mask * harmonics
        harmonics = harmocic_amplitude * torch.sum(harmonics, dim=1, keepdim=True)

        return harmonics + noise