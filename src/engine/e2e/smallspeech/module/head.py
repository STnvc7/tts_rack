import torch
import torch.nn as nn
import torch.nn.functional as F
from dsp_board.transforms import stft, istft
from .oscillator import wavetable_osc, harmonic_osc, anti_alias

class DDSPHead(nn.Module):
    def __init__(self, channels, sample_rate, fft_size, hop_size, n_harmonics, n_bands):
        super().__init__()
        self.periodic = AdditiveOscillator(channels, n_harmonics, sample_rate, hop_size)
        self.aperiodic = LTVFilter(channels, n_bands, fft_size, hop_size)
    def forward(self, z, f0):
        periodic, _ = self.periodic(z, f0)
        aperiodic, _ = self.aperiodic(z)
        y = periodic + aperiodic
        return y, periodic, aperiodic
        
class WTNetHead(nn.Module):
    def __init__(self, channels, sample_rate, fft_size, hop_size, n_samples, n_bands):
        super().__init__()
        self.periodic = WavetableOscillator(channels, n_samples, sample_rate, hop_size)
        self.aperiodic = LTVFilter(channels, n_bands, fft_size, hop_size)
    def forward(self, z, f0):
        periodic, _ = self.periodic(z, f0)
        aperiodic, _ = self.aperiodic(z)
        y = periodic + aperiodic
        return y, periodic, aperiodic

class WavetableOscillator(nn.Module):
    def __init__(self, channels, n_samples, sample_rate, hop_size):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.proj_amp = nn.Conv1d(channels, n_samples//2+1, 1, bias=True)
        self.proj_phase = nn.Conv1d(channels, n_samples//2+1, 1, bias=False)
    
    def forward(self, z, f0):
        amp = self.proj_amp(z)
        phase = self.proj_phase(z)
        
        amp = amp.clip(max=5.).exp()
        amp = anti_alias(amp, f0, self.sample_rate)
        S = amp * (phase.cos() + 1j*phase.sin())
        wt = torch.fft.irfft(S, dim=1)
        harmonic = wavetable_osc(wt, f0, self.sample_rate, self.hop_size)
        return harmonic, wt
        
class AdditiveOscillator(nn.Module):
    def __init__(self, channels, n_harmonics, sample_rate, hop_size):
        super().__init__()
        self.sample_rate = sample_rate
        self.proj_amp = nn.Conv1d(channels, n_harmonics, 1, bias=True)
        self.upsample = nn.Upsample(scale_factor=hop_size, mode="linear", align_corners=True)
    
    def forward(self, z, f0):
        amp = self.proj_amp(z)
        amp = F.softplus(amp)
        amp = anti_alias(amp, f0, self.sample_rate)
        amp = self.upsample(amp)
        f0 = self.upsample(f0)
        harmonic = harmonic_osc(amp, f0, self.sample_rate)
        return harmonic, amp
        
class LTVFilter(nn.Module):
    def __init__(self, channels, n_bands, fft_size, hop_size):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.proj_amp = nn.Conv1d(channels, n_bands, 1)
        self.upsample = nn.Upsample(size=fft_size//2+1, mode="linear", align_corners=True)
        
    def forward(self, z):
        amp = self.proj_amp(z)
        amp = amp.exp().clip(max=100)
        amp = self.upsample(amp.transpose(1, 2)).transpose(1, 2)
        H = torch.complex(amp, torch.zeros_like(amp))
        
        B, _, T = z.shape
        x = torch.rand(B, T*self.hop_size, device=z.device) * 2 - 1
        X = stft(x, self.fft_size, self.hop_size)
        Y = X * H
        y = istft(Y, self.fft_size, self.hop_size).unsqueeze(1)
        
        return y, amp
        