# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Source regularization loss modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from dsp_board.features import spectrogram
from .cheaptrick import CheapTrick


class ResidualLoss(nn.Module):
    """The regularization loss of hn-uSFGAN."""

    def __init__(
        self,
        sample_rate=24000,
        fft_size=2048,
        hop_size=120,
        f0_floor=100,
        f0_ceil=840,
        n_mels=80,
        fmin=0,
        fmax=None,
        power=False,
        elim_0th=True,
    ):
        """Initialize ResidualLoss module.

        Args:
            sample_rate (int): Sampling rate.
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            f0_floor (int): Minimum F0 value.
            f0_ceil (int): Maximum F0 value.
            n_mels (int): Number of Mel basis.
            fmin (int): Minimum frequency for Mel.
            fmax (int): Maximum frequency for Mel.
            power (bool): Whether to use power or magnitude spectrogram.
            elim_0th (bool): Whether to exclude 0th cepstrum in CheapTrick.
                If set to true, power is estimated by source-network.

        """
        super(ResidualLoss, self).__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.cheaptrick = CheapTrick(
            sample_rate=sample_rate,
            hop_size=hop_size,
            fft_size=fft_size,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        self.win_length = fft_size
        self.register_buffer("window", torch.hann_window(self.win_length))

        # define mel-filter-bank
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2

        self.mel_filter = torchaudio.transforms.MelScale(
            n_mels=n_mels, 
            sample_rate=sample_rate,
            f_min=fmin,
            f_max=fmax,
            n_stft=(fft_size//2+1),
            norm='slaney',
            mel_scale='slaney'
        )

        self.power = power
        self.elim_0th = elim_0th

    def forward(self, s, y, f):
        """Calculate forward propagation.

        Args:
            s (Tensor): Predicted source excitation signal (B, 1, T).
            y (Tensor): Ground truth signal (B, 1, T).
            f (Tensor): F0 sequence (B, 1, T // hop_size).

        Returns:
            Tensor: Loss value.

        """
        s, y, f = s.squeeze(1), y.squeeze(1), f.squeeze(1)
        mel_filter = self.mel_filter.to(s.device)

        with torch.no_grad():
            # calculate log power (or magnitude) spectrograms
            e = self.cheaptrick.forward(y, f, self.power, self.elim_0th)
            y = spectrogram(
                y,
                self.fft_size,
                self.hop_size,
                self.win_length,
            ).permute(0,2,1)
            # adjust length, (B, T', C)
            minlen = min(e.size(1), y.size(1))
            e, y = e[:, :minlen, :], y[:, :minlen, :]

            # calculate mean power (or magnitude) of y
            if self.elim_0th:
                y_mean = y.mean(dim=-1, keepdim=True)

            # calculate target of output source signal
            y = torch.log(torch.clamp(y, min=1e-7))
            t = (y - e).exp()
            if self.elim_0th:
                t_mean = t.mean(dim=-1, keepdim=True)
                t = y_mean / t_mean * t

            # apply mel-filter-bank and log
            t = mel_filter(t.permute(0,2,1))
            t = torch.log(torch.clamp(t, min=1e-7))

        # calculate power (or magnitude) spectrogram
        s = spectrogram(
            s,
            self.fft_size,
            self.hop_size,
            self.win_length,
        ).permute(0,2,1)
        # adjust length, (B, T', C)
        minlen = min(minlen, s.size(1))
        s, t = s[:, :minlen, :], t[:, :minlen, :]

        # apply mel-filter-bank and log
        s = mel_filter(s.permute(0,2,1))
        s = torch.log(torch.clamp(s, min=1e-7))

        loss = F.l1_loss(s, t.detach())

        return loss