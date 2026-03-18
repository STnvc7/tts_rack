# Copyright 2024 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Wavehax generator modules."""

from typing import Dict, Optional
import torch
from torch import nn
from dsp_board.transforms import stft, istft

from interface.feature import AcousticFeature
from interface.model import GeneratorOutput, Generator
from interface.loggable import Audio, Heatmap

from .norm import LayerNorm2d
from .convnext import ConvNeXtBlock2d
from .stft import STFT, to_log_magnitude_and_phase, to_real_imaginary
from .prior import WavehaxPriorGenerator

class WavehaxGenerator(Generator):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        drop_prob: float = 0.0,
        use_logmag_phase: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.use_logmag_phase = use_logmag_phase

        # Prior waveform generator
        self.prior_generator = WavehaxPriorGenerator(
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

        # STFT layer
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

        # Input projection layers
        n_bins = n_fft // 2 + 1
        self.prior_proj = nn.Conv1d(
            n_bins, n_bins, 7, padding=3, padding_mode="reflect"
        )
        self.cond_proj = nn.Conv1d(
            in_channels, n_bins, 7, padding=3, padding_mode="reflect"
        )

        # Input normalization and projection layers
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ConvNeXtBlock2d(
                channels,
                mult_channels,
                kernel_size,
                drop_prob=drop_prob,
                layer_scale_init_value=1 / num_blocks,
            )
            self.blocks += [block]

        # Output normalization and projection layers
        self.output_norm = LayerNorm2d(channels)
        self.output_proj = nn.Conv2d(channels, 2, 1)

        self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """
        Initialize weights of the module.

        Args:
            m (Any): Module to initialize.
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
    def forward(
        self, 
        input_feature: Dict[AcousticFeature, torch.Tensor],
        wav: Optional[torch.Tensor] = None
    ) -> GeneratorOutput:
        cond = input_feature["mel_spectrogram"]
        f0 = input_feature["pitch"]

        """
        Calculate forward propagation.

        Args:
            cond (Tensor): Conditioning features with shape (batch, in_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
            Tensor: Generated prior waveforms with shape (batch, 1, frames * hop_length).
        """
        # Generate prior waveform and compute spectrogram
        with torch.no_grad():
            prior = self.prior_generator(f0)
            # real, imag = self.stft(prior)
            prior_spc = stft(prior.squeeze(1), self.n_fft, self.hop_length)
            real, imag = prior_spc.real, prior_spc.imag
            if self.use_logmag_phase:
                prior1, prior2 = to_log_magnitude_and_phase(real, imag)
            else:
                prior1, prior2 = real, imag

        # Apply input projection
        prior1_proj = self.prior_proj(prior1)
        prior2_proj = self.prior_proj(prior2)
        cond = self.cond_proj(cond)

        # Convert to 2d representation
        x = torch.stack([prior1, prior2, prior1_proj, prior2_proj, cond], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply residual blocks
        for f in self.blocks:
            x = f(x)

        # Apply output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Apply iSTFT followed by overlap and add
        if self.use_logmag_phase:
            real, imag = to_real_imaginary(x[:, 0], x[:, 1])
        else:
            real, imag = x[:, 0], x[:, 1]
        # x = self.stft.inverse(real, imag)
        x = istft(torch.complex(real, imag), self.n_fft, self.hop_length).unsqueeze(1)

        
        return GeneratorOutput(
            pred=x,    
            loggable_outputs={
                "prior": Audio(data=prior.squeeze(), sample_rate=self.sample_rate),
                "spectrogram": Heatmap(data=torch.complex(real, imag).abs().clip(min=1e-8).log().squeeze(), label="spectrogram", origin="lower"),
            }
        )